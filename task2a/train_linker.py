from __future__ import annotations
import argparse
import inspect
import json
import random
import numpy as np
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
_TRAINER_PARAMS = set(inspect.signature(Trainer.__init__).parameters.keys())

def _ta_kwargs(**kw):
    return {k: v for k, v in kw.items() if k in _TA_PARAMS}

def load_records(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading candidates", unit="entity"):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def build_pairs(records, num_random_negatives: int = 2, rng: random.Random | None = None):
    if rng is None:
        rng = random.Random(42)

    concept_pool = []
    for r in records:
        for c in r["candidates"]:
            concept_pool.append(c["concept_text"])
    concept_pool = list(set(concept_pool))

    rows = []
    for record in records:
        gold = str(record["concept_id"])
        seen = set()
        for candidate in record["candidates"]:
            cid = str(candidate["concept_id"])
            if cid in seen:
                continue
            seen.add(cid)
            rows.append(
                {
                    "entity": record["entity_text"],
                    "concept": candidate["concept_text"],
                    "label": int(cid == gold),
                }
            )
        for _ in range(num_random_negatives):
            if not concept_pool:
                break
            neg = rng.choice(concept_pool)
            rows.append(
                {"entity": record["entity_text"], "concept": neg, "label": 0}
            )
    return Dataset.from_list(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="../outputs/sapbert_candidates_v2.jsonl")
    parser.add_argument(
        "--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    parser.add_argument("--output-dir", default="../outputs/sapbert-linker")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    records = load_records(args.candidates)
    print(f"Total records: {len(records)}")

    note_ids = sorted(set(str(r["note_id"]) for r in records))
    rng = random.Random(args.seed)
    rng.shuffle(note_ids)
    n_test = max(1, int(0.1 * len(note_ids)))
    test_notes = set(note_ids[:n_test])
    train_notes = set(note_ids[n_test:])

    train_records = [r for r in records if str(r["note_id"]) in train_notes]
    test_records = [r for r in records if str(r["note_id"]) in test_notes]

    train_data = build_pairs(train_records, num_random_negatives=2, rng=rng)
    test_data = build_pairs(test_records, num_random_negatives=0, rng=rng)

    def tokenize(batch):
        return tokenizer(
            batch["entity"],
            batch["concept"],
            truncation=True,
            max_length=args.max_length,
        )

    train_data = train_data.map(tokenize, batched=True).remove_columns(
        ["entity", "concept"]
    )
    test_data = test_data.map(tokenize, batched=True).remove_columns(
        ["entity", "concept"]
    )

    print(f"Notes: {len(note_ids)} | train: {len(train_notes)} | test: {len(test_notes)}")
    print(f"Train pairs: {len(train_data)} | Test pairs: {len(test_data)}")
    print(
        f"Train pos ratio: {np.mean(train_data['label']):.4f} | "
        f"Test pos ratio: {np.mean(test_data['label']):.4f}"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    )

    training_args = TrainingArguments(
        **_ta_kwargs(
            output_dir=args.output_dir,
            eval_strategy="epoch",
            evaluation_strategy="epoch",       # old name
            save_strategy="epoch",
            learning_rate=args.lr,
            warmup_ratio=0.1,
            warmup_steps=500,                  # old fallback
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            num_train_epochs=args.epochs,
            report_to="none",
            fp16=True,
            gradient_checkpointing=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=100,
            ddp_find_unused_parameters=False,
        )
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        _tokenizer=tokenizer,
    )
    if "processing_class" in _TRAINER_PARAMS:
        trainer_kwargs["processing_class"] = trainer_kwargs.pop("_tokenizer")
    else:
        trainer_kwargs["tokenizer"] = trainer_kwargs.pop("_tokenizer")

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()