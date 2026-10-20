# task2b/predict.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def parse_first_json(text: str) -> dict | None:
    """Balanced-brace JSON extractor. Returns the first complete JSON object."""
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                snippet = text[start:i + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    start = None
    return None


def align_entity(text: str, ent_text: str, search_start: int, search_end: int):
    """Find ent_text in text[search_start:search_end]; multi-stage fallback."""
    # 1. exact
    pos = text.find(ent_text, search_start, search_end)
    if pos == -1:
        pos = text.find(ent_text)
    if pos != -1:
        return pos, pos + len(ent_text)

    # 2. case-insensitive
    low = text.lower()
    ent_low = ent_text.lower()
    pos = low.find(ent_low, search_start, search_end)
    if pos == -1:
        pos = low.find(ent_low)
    if pos != -1:
        return pos, pos + len(ent_text)

    # 3. whitespace-normalized
    ent_norm = re.sub(r"\s+", " ", ent_text).strip()
    if not ent_norm:
        return None
    L = len(ent_norm)
    for wlen in (L, L + 5, L + 10, L - 5, L - 10, len(ent_text)):
        if wlen <= 0:
            continue
        for i in range(max(0, search_start - 20),
                       min(len(text), search_end)):
            sub = text[i:i + wlen]
            if re.sub(r"\s+", " ", sub).strip().lower() == ent_norm.lower():
                return i, i + wlen

    # 4. whole note fallback
    for i in range(0, len(text) - L + 1):
        sub = text[i:i + L]
        if re.sub(r"\s+", " ", sub).strip().lower() == ent_norm.lower():
            return i, i + L

    return None


FEW_SHOT_EXAMPLES = [
    {
        "note": (
            "The patient was diagnosed with pneumonia and treated with antibiotics. "
            "Blood pressure was elevated."
        ),
        "json": (
            '{"entities":['
            '{"text":"pneumonia","snomed_ct_id":"233604007"},'
            '{"text":"elevated blood pressure","snomed_ct_id":"24184005"}'
            ']}'
        ),
    },
    {
        "note": (
            "History of type 2 diabetes mellitus. Patient denies chest pain. "
            "Underwent laparoscopic cholecystectomy."
        ),
        "json": (
            '{"entities":['
            '{"text":"type 2 diabetes mellitus","snomed_ct_id":"44054006"},'
            '{"text":"chest pain","snomed_ct_id":"29857009"},'
            '{"text":"laparoscopic cholecystectomy","snomed_ct_id":"45595009"}'
            ']}'
        ),
    },
]


def build_prompt(mode: str, chunk_text: str) -> str:
    instruction = (
        "You are a medical entity extraction and SNOMED CT linking system. "
        "Given a clinical note excerpt, extract all clinically relevant medical "
        "entities (diseases, findings, symptoms, procedures, medications, anatomical "
        "sites) and link each to its SNOMED CT concept ID.\n\n"
        "Return ONLY valid JSON in exactly this format, no explanations:\n"
        '{"entities":[{"text":"ENTITY","snomed_ct_id":"SNOMED_ID"}]}\n'
        "Use the exact surface text from the note. Output only the JSON."
    )
    if mode == "few-shot":
        shots = ""
        for ex in FEW_SHOT_EXAMPLES:
            shots += f"\n\nClinical note:\n{ex['note']}\n\nAnswer:\n{ex['json']}"
        return (
            f"<s>[INST] {instruction}{shots}\n\n"
            f"Clinical note:\n{chunk_text} [/INST]"
        )
    return f"<s>[INST] {instruction}\n\nClinical note:\n{chunk_text} [/INST]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["few-shot", "qlora"], required=True)
    p.add_argument("--test-notes", default="Data/test-notes.csv")
    p.add_argument("--model", default="BioMistral/BioMistral-7B")
    p.add_argument("--adapter", default="outputs/biomistral_qlora")
    p.add_argument("--output", default="")
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--limit", type=int, default=-1)
    args = p.parse_args()

    if not args.output:
        args.output = f"outputs/biomistral_preds/{args.mode}.jsonl"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading base model (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )

    if args.mode == "qlora":
        print(f"Attaching LoRA adapters from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    notes = pd.read_csv(args.test_notes)
    if args.limit > 0:
        notes = notes.head(args.limit)
    print(f"Notes: {len(notes)}")

    results = []
    for row in tqdm(notes.itertuples(), total=len(notes), desc=args.mode):
        note_id = str(row.note_id)
        text = str(row.text)

        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        n = len(ids)
        if n == 0:
            results.append({"note_id": note_id,
                            "prediction": {"entities": []}})
            continue

        step = args.window - args.overlap
        chunks = []
        for s in range(0, n, step):
            e = min(s + args.window, n)
            if e - s < 32:
                continue
            chunks.append((s, e, offs[s][0], offs[e - 1][1]))
            if e == n:
                break

        predictions = []
        for (_, _, cs, ce) in chunks:
            prompt = build_prompt(args.mode, text[cs:ce])
            input_ids = tok(prompt, add_special_tokens=False,
                            return_tensors="pt")["input_ids"].to("cuda")
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )
            generated = out[0][input_ids.shape[1]:]
            decoded = tok.decode(generated, skip_special_tokens=True)

            obj = parse_first_json(decoded)
            if not obj or "entities" not in obj:
                continue

            for ent in obj["entities"]:
                if not isinstance(ent, dict):
                    continue
                et = ent.get("text")
                cid = ent.get("snomed_ct_id")
                if not et or not cid:
                    continue
                cid = re.sub(r"[^0-9]", "", str(cid).strip())
                if not cid:
                    continue
                aligned = align_entity(text, str(et), cs, ce)
                if aligned is None:
                    continue
                start, end = aligned
                predictions.append({
                    "text": text[start:end],
                    "snomed_ct_id": cid,
                    "start": start,
                    "end": end,
                })

        uniq = {}
        for pr in predictions:
            uniq[(pr["start"], pr["end"], pr["snomed_ct_id"])] = pr

        results.append({
            "note_id": note_id,
            "prediction": {"entities": list(uniq.values())},
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_ents = sum(len(r["prediction"]["entities"]) for r in results)
    print(f"Wrote {total_ents} entities from {len(results)} notes to {args.output}")


if __name__ == "__main__":
    main()
