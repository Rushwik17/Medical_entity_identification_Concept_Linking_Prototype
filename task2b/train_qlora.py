# task2b/train_qlora.py
from __future__ import annotations

import argparse
import json
import inspect
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="outputs/biomistral_data/train.jsonl")
    p.add_argument("--eval", default="outputs/biomistral_data/eval.jsonl")
    p.add_argument("--model", default="BioMistral/BioMistral-7B")
    p.add_argument("--out-dir", default="outputs/biomistral_qlora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ---- Load prepared data ----
    train_records = load_jsonl(args.train)
    eval_records = load_jsonl(args.eval)
    print(f"Train records: {len(train_records):,} | Eval records: {len(eval_records):,}")

    def encode(rec):
        prompt_ids = tok(rec["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tok(rec["target"], add_special_tokens=False)["input_ids"]
        total = len(prompt_ids) + len(target_ids)
        if total > args.max_length:
            overflow = total - args.max_length
            prompt_ids = prompt_ids[overflow:]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {"input_ids": input_ids, "labels": labels}

    train_ds = Dataset.from_list(train_records).map(
        encode, remove_columns=["prompt", "target"]
    )
    eval_ds = Dataset.from_list(eval_records).map(
        encode, remove_columns=["prompt", "target"]
    )

    def collate(features):
        max_len = min(max(len(x["input_ids"]) for x in features), args.max_length)
        input_ids, labels, attn = [], [], []
        for x in features:
            ids = x["input_ids"][:max_len]
            lab = x["labels"][:max_len]
            pad = max_len - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    # ---- Quantized base model ----
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- Training args ----
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * args.grad_accum))
    eval_steps = max(5, steps_per_epoch // 2)
    save_steps = eval_steps * 2

    print(f"steps_per_epoch ≈ {steps_per_epoch} | eval_steps={eval_steps} | save_steps={save_steps}")

    ta = dict(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        eval_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    supported = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    ta = {k: v for k, v in ta.items() if k in supported}

    # ---- Trainer with version-compatible tokenizer kwarg ----
    trainer_kwargs = dict(
        model=model,
        args=TrainingArguments(**ta),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok

    trainer = Trainer(**trainer_kwargs)

    print("Starting QLoRA training...")
    trainer.train()

    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"Saved adapters to {args.out_dir}")


if __name__ == "__main__":
    main()
