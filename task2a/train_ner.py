from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from src.data_utils import load_notes, spans_by_note, valid_annotations

LABELS = ["O", "B-MEDICAL", "I-MEDICAL"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
_TRAINER_PARAMS = set(inspect.signature(Trainer.__init__).parameters.keys())

def _ta_kwargs(**kw):
    return {k: v for k, v in kw.items() if k in _TA_PARAMS}

def build_examples(
    data_dir: str,
    tokenizer,
    max_length: int,
    stride: int,
    max_chunks_per_note: int | None = None,
) -> Dataset:
    """
    Chunk each note into overlapping windows whose TOTAL token count
    (CLS + chunk + SEP) is <= max_length.
    """
    notes = load_notes(data_dir)
    annotations = valid_annotations(data_dir)
    note_spans = spans_by_note(annotations)

    chunk_size = max_length - 2
    if chunk_size <= 0:
        raise ValueError(f"max_length={max_length} too small; need >= 3.")

    rows = []
    for row in tqdm(
        notes.itertuples(),
        total=len(notes),
        desc="Building chunked BIO examples",
        unit="note",
    ):
        text = row.text
        note_id = str(row.note_id)
        spans = note_spans.get(note_id, [])

        spans = sorted(spans, key=lambda x: (x[1] - x[0]), reverse=True)
        full = tokenizer(
            text,
            truncation=False,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = full["input_ids"]
        offsets = full["offset_mapping"]
        n = len(input_ids)

        if n == 0:
            continue

        step = max(1, chunk_size - stride)
        starts = list(range(0, n, step))
        chunks = []
        for s in starts:
            e = min(s + chunk_size, n)
            if e - s < 8:
                continue
            chunks.append((s, e))
        if chunks and chunks[-1][1] < n:
            chunks.append((max(0, n - chunk_size), n))

        if max_chunks_per_note is not None and len(chunks) > max_chunks_per_note:
            idx = np.linspace(0, len(chunks) - 1, max_chunks_per_note).astype(int)
            chunks = [chunks[i] for i in idx]

        for s, e in chunks:
            ids = input_ids[s:e]
            offs = offsets[s:e]

            labels = []
            for start, end in offs:
                label = LABEL_TO_ID["O"]
                for span_start, span_end, _ in spans:
                    if start < span_end and end > span_start:
                        if start <= span_start < end:
                            label = LABEL_TO_ID["B-MEDICAL"]
                        else:
                            label = LABEL_TO_ID["I-MEDICAL"]
                        break
                labels.append(label)

            cls_id = tokenizer.cls_token_id
            sep_id = tokenizer.sep_token_id

            input_ids_wrapped = [cls_id] + ids + [sep_id]
            attn = [1] * len(input_ids_wrapped)
            labels_wrapped = [-100] + labels + [-100]

            assert len(input_ids_wrapped) <= max_length, (
                f"Chunk length {len(input_ids_wrapped)} > max_length {max_length}"
            )

            rows.append(
                {
                    "input_ids": input_ids_wrapped,
                    "attention_mask": attn,
                    "labels": labels_wrapped,
                }
            )

    return Dataset.from_list(rows)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_preds, true_labels = [], []
    for p, l in zip(preds, labels):
        tp, tl = [], []
        for pi, li in zip(p, l):
            if li == -100:
                continue
            tp.append(ID_TO_LABEL[pi])
            tl.append(ID_TO_LABEL[li])
        true_preds.append(tp)
        true_labels.append(tl)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--output-dir", default="../outputs/clinical-bert-ner")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--max-chunks-per-note", type=int, default=12)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--eval-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    args = parser.parse_args()

    max_chunks = None if args.max_chunks_per_note < 0 else args.max_chunks_per_note

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token or tokenizer.eos_token

    dataset = build_examples(
        args.data_dir,
        tokenizer,
        args.max_length,
        args.stride,
        max_chunks_per_note=max_chunks,
    )
    print(f"Total chunks: {len(dataset)}")

    dist = Counter()
    for labels in dataset["labels"]:
        for l in labels:
            if l != -100:
                dist[ID_TO_LABEL[l]] += 1
    total_tok = sum(dist.values())
    print(
        "Label distribution:",
        {k: f"{v} ({100*v/total_tok:.2f}%)" for k, v in dist.items()},
    )

    indices = np.arange(len(dataset))
    train_idx, eval_idx = train_test_split(
        indices, test_size=args.eval_size, random_state=args.seed
    )
    train_dataset = dataset.select(train_idx.tolist())
    eval_dataset = dataset.select(eval_idx.tolist())
    print(f"Train chunks: {len(train_dataset)} | Eval chunks: {len(eval_dataset)}")

    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        **_ta_kwargs(
            output_dir=args.output_dir,
            eval_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            warmup_steps=500,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            report_to="none",
            seed=args.seed,
            fp16=True,
            gradient_checkpointing=True,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            ddp_find_unused_parameters=False,
        )
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-100
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if "processing_class" in _TRAINER_PARAMS:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "labels.json").write_text(json.dumps(LABELS, indent=2))

if __name__ == "__main__":
    main()