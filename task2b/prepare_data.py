# task2b/prepare_data.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notes", default="Data/training-notes.csv")
    p.add_argument("--annotations", default="Data/train-annotations.csv")
    p.add_argument("--tokenizer", default="BioMistral/BioMistral-7B")
    p.add_argument("--out-dir", default="outputs/biomistral_data")
    p.add_argument("--window", type=int, default=768)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--max-answer-tokens", type=int, default=1400)
    p.add_argument("--max-total-tokens", type=int, default=2800)
    p.add_argument("--max-entities-per-chunk", type=int, default=60,
                   help="Drop chunks with more entities than this (keeps answers bounded).")
    p.add_argument("--eval-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    notes = pd.read_csv(args.notes)
    annots = pd.read_csv(args.annotations)
    print(f"Notes: {len(notes)} | Annotations: {len(annots)}")

    ann_by_note: dict[str, list[dict]] = {}
    for r in annots.itertuples():
        ann_by_note.setdefault(str(r.note_id), []).append({
            "start": int(r.start),
            "end": int(r.end),
            "concept_id": str(r.concept_id),
        })

    note_ids = sorted(ann_by_note.keys())
    rng.shuffle(note_ids)
    n_eval = max(1, int(args.eval_frac * len(note_ids)))
    eval_ids = set(note_ids[:n_eval])
    train_ids = set(note_ids[n_eval:])
    print(f"Train notes: {len(train_ids)} | Eval notes: {len(eval_ids)}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    INSTRUCTION = (
        "You are a medical entity extraction and SNOMED CT linking system. "
        "Given a clinical note excerpt, extract all clinically relevant medical "
        "entities (diseases, findings, symptoms, procedures, medications, anatomical "
        "sites) and link each to its SNOMED CT concept ID.\n\n"
        "Return ONLY valid JSON in exactly this format, no explanations:\n"
        '{"entities":[{"text":"ENTITY","snomed_ct_id":"SNOMED_ID"}]}\n'
        "Use the exact surface text from the note. Output only the JSON."
    )

    def make_prompt(chunk_text):
        return f"<s>[INST] {INSTRUCTION}\n\nClinical note:\n{chunk_text} [/INST]"

    def make_target(entities):
        return json.dumps(
            {"entities": entities},
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "</s>"

    def chunk_note(text, window, overlap):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        n = len(ids)
        if n == 0:
            return []
        step = max(1, window - overlap)
        chunks = []
        for s in range(0, n, step):
            e = min(s + window, n)
            if e - s < 32:
                continue
            chunks.append((offs[s][0], offs[e - 1][1]))
            if e == n:
                break
        return chunks

    def write_split(note_id_set, out_path):
        records = []
        dropped_entity_cap = 0
        dropped_len = 0
        kept = 0
        for r in notes.itertuples():
            nid = str(r.note_id)
            if nid not in note_id_set:
                continue
            text = str(r.text)
            spans = ann_by_note.get(nid, [])
            if not spans:
                continue
            for (cs, ce) in chunk_note(text, args.window, args.overlap):
                inside = []
                for s in spans:
                    es, ee = s["start"], s["end"]
                    ov = max(0, min(ee, ce) - max(es, cs))
                    ent_len = max(1, ee - es)
                    if ov / ent_len >= 0.8:
                        inside.append({
                            "start": max(es, cs),
                            "end": min(ee, ce),
                            "concept_id": s["concept_id"],
                        })
                if not inside:
                    continue
                if len(inside) > args.max_entities_per_chunk:
                    dropped_entity_cap += 1
                    continue
                ents = [{"text": text[s["start"]:s["end"]],
                         "snomed_ct_id": s["concept_id"]} for s in inside]
                prompt = make_prompt(text[cs:ce])
                target = make_target(ents)
                pt = tok(prompt, add_special_tokens=False)["input_ids"]
                tt = tok(target, add_special_tokens=False)["input_ids"]
                if len(tt) > args.max_answer_tokens:
                    dropped_len += 1
                    continue
                if len(pt) + len(tt) > args.max_total_tokens:
                    dropped_len += 1
                    continue
                records.append({"prompt": prompt, "target": target})
                kept += 1
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  kept: {kept} | dropped (entity cap): {dropped_entity_cap} | dropped (len): {dropped_len}")
        print(f"Wrote {len(records):,} records -> {out_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Train split:")
    write_split(train_ids, out_dir / "train.jsonl")
    print("Eval split:")
    write_split(eval_ids, out_dir / "eval.jsonl")


if __name__ == "__main__":
    main()
