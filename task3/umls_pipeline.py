# task3/umls_pipeline.py
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from src.data_utils import load_notes


_WS = re.compile(r"\s+")


def clean(s: str) -> str:
    return _WS.sub(" ", str(s)).strip().lower()


# ---- assertion ----
NEG = ["no ", "not ", "denies", "denied", "without", "negative for",
       "no evidence of", "no signs of", "free of", "ruled out",
       "rule out", "r/o ", "absence of", "never had"]
HIST = ["history of", "h/o ", "past medical history", "pmh",
        "prior", "previous", "formerly", "previously"]
HYPO = ["possible", "probable", "suspected", "concern for",
        "concerning for", "may have", "might have",
        "recommend", "consider", "consistent with"]


def classify_assertion(text, start, end):
    pre = text[max(0, start - 80):start].lower().split(".")[-1]
    return {
        "negated": any(c in pre for c in NEG),
        "historical": any(c in pre for c in HIST),
        "hypothetical": any(c in pre for c in HYPO),
    }


# ---- NER (chunked) ----
@torch.no_grad()
def predict_spans(text, tok, model, device, label_names,
                  max_length=512, stride=128, batch_size=4):
    chunk_size = max_length - 2
    step = max(1, chunk_size - stride)
    full = tok(text, truncation=False, return_offsets_mapping=True,
               add_special_tokens=False)
    ids = full["input_ids"]
    offsets = full["offset_mapping"]
    n = len(ids)
    if n == 0:
        return []
    chunks = []
    for s in range(0, n, step):
        e = min(s + chunk_size, n)
        if e - s < 4:
            continue
        chunks.append((s, e))
    if chunks and chunks[-1][1] < n:
        chunks.append((max(0, n - chunk_size), n))
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    pad_id = tok.pad_token_id or 0
    pred_map = {}
    for i in range(0, len(chunks), batch_size):
        bc = chunks[i:i + batch_size]
        ids_list, attn_list = [], []
        for s, e in bc:
            xi = [cls_id] + ids[s:e] + [sep_id]
            ids_list.append(xi)
            attn_list.append([1] * len(xi))
        ml = max(len(x) for x in ids_list)
        pid = [x + [pad_id] * (ml - len(x)) for x in ids_list]
        pat = [x + [0] * (ml - len(x)) for x in attn_list]
        logits = model(
            input_ids=torch.tensor(pid, device=device),
            attention_mask=torch.tensor(pat, device=device),
        ).logits
        preds = logits.argmax(-1).cpu().numpy()
        for bi, (s, e) in enumerate(bc):
            for j in range(1, 1 + (e - s)):
                pred_map[s + j - 1] = int(preds[bi, j])
    seq = [pred_map.get(i, 0) for i in range(n)]
    spans, cur = [], None
    for i, lab in enumerate(seq):
        nm = label_names[lab]
        if nm == "B-MEDICAL":
            if cur is not None:
                spans.append(tuple(cur))
            cur = list(offsets[i])
        elif nm == "I-MEDICAL":
            if cur is None:
                cur = list(offsets[i])
            else:
                cur[1] = offsets[i][1]
        else:
            if cur is not None:
                spans.append(tuple(cur))
                cur = None
    if cur is not None:
        spans.append(tuple(cur))
    return [(s, e) for s, e in spans if e - s >= 2]


# ---- retrieval over full UMLS ----
@torch.no_grad()
def retrieve_cuis(entity_texts, sapbert, tok, emb_fp16, meta,
                  device, retrieve_k=20, batch_size=64):
    cuis = meta["CUI"].tolist()
    strs = meta["STR"].tolist()
    sabs = meta["SAB"].tolist()
    codes = meta["CODE"].tolist()
    results = []
    for i in range(0, len(entity_texts), batch_size):
        batch = [clean(t) for t in entity_texts[i:i + batch_size]]
        t = tok(batch, padding=True, truncation=True, max_length=64,
                return_tensors="pt").to(device)
        h = sapbert(**t).last_hidden_state
        m = t["attention_mask"].unsqueeze(-1).half()
        q = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1).half()
        scores = q @ emb_fp16.T
        v, j = scores.topk(retrieve_k * 5, dim=1)
        v = v.float().cpu().tolist()
        j = j.cpu().tolist()
        for b in range(len(batch)):
            best = {}
            for sc, ix in zip(v[b], j[b]):
                cui = cuis[ix]
                if cui not in best or sc > best[cui]["score"]:
                    best[cui] = {
                        "cui": cui,
                        "concept_text": strs[ix],
                        "snomed_code": codes[ix] if sabs[ix] == "SNOMEDCT_US" else None,
                        "source": sabs[ix],
                        "score": float(sc),
                    }
            ranked = sorted(best.values(), key=lambda x: x["score"], reverse=True)
            results.append(ranked[:retrieve_k])
    return results


# ---- linker rerank ----
@torch.no_grad()
def rerank(entity_texts, cand_lists, linker, tok, device, batch_size=64):
    chosen = []
    for i in range(0, len(entity_texts), batch_size):
        e_b = entity_texts[i:i + batch_size]
        c_b = cand_lists[i:i + batch_size]
        flat_e, flat_c, offs = [], [], []
        for e, cs in zip(e_b, c_b):
            offs.append((len(flat_e), len(flat_e) + len(cs)))
            for c in cs:
                flat_e.append(e)
                flat_c.append(c["concept_text"])
        if not flat_e:
            chosen.extend([None] * len(e_b))
            continue
        enc = tok(flat_e, flat_c, padding=True, truncation=True,
                  max_length=64, return_tensors="pt").to(device)
        probs = torch.softmax(linker(**enc).logits, -1)[:, 1].cpu().numpy()
        for (s, e), cs in zip(offs, c_b):
            if e <= s:
                chosen.append(None)
                continue
            chosen.append(cs[int(np.argmax(probs[s:e]))])
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="Data")
    p.add_argument("--ner-model", default="outputs/clinical-bert-ner")
    p.add_argument("--sapbert", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--linker", default="task3/outputs/sapbert-linker-umls")
    p.add_argument("--index-dir", default="task3/outputs/umls_index")
    p.add_argument("--output", default="task3/outputs/umls_predictions.jsonl")
    p.add_argument("--retrieve-k", type=int, default=20)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading index from {args.index_dir}")
    emb = torch.load(Path(args.index_dir) / "umls_emb.pt",
                     map_location="cpu").half().to(device)
    meta = pd.read_parquet(Path(args.index_dir) / "umls_meta.parquet")
    print(f"Index: {tuple(emb.shape)} | rows {len(meta):,}")

    ner_tok = AutoTokenizer.from_pretrained(args.ner_model)
    ner_model = AutoModelForTokenClassification.from_pretrained(
        args.ner_model).to(device).eval()
    label_names = list(ner_model.config.id2label.values())

    sap_tok = AutoTokenizer.from_pretrained(args.sapbert)
    sapbert = AutoModel.from_pretrained(args.sapbert).to(device).eval().half()

    link_tok = AutoTokenizer.from_pretrained(args.linker)
    linker = AutoModelForSequenceClassification.from_pretrained(
        args.linker).to(device).eval().half()

    notes = load_notes(args.data_dir)
    total = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in tqdm(notes.itertuples(), total=len(notes), desc="Pipeline"):
            note_id = str(row.note_id)
            text = row.text
            spans = predict_spans(text, ner_tok, ner_model, device, label_names,
                                  max_length=args.max_length, stride=args.stride)
            if not spans:
                f.write(json.dumps({"note_id": note_id, "entities": []}) + "\n")
                continue
            ent_texts = [text[s:e] for s, e in spans]
            asserts = [classify_assertion(text, s, e) for s, e in spans]
            cands = retrieve_cuis(ent_texts, sapbert, sap_tok, emb, meta, device,
                                  retrieve_k=args.retrieve_k)
            chosen = rerank(ent_texts, cands, linker, link_tok, device)
            ents = []
            for (s, e), txt, a, c in zip(spans, ent_texts, asserts, chosen):
                ents.append({
                    "start": int(s), "end": int(e), "text": txt,
                    "cui": c["cui"] if c else None,
                    "snomed_id": c.get("snomed_code") if c else None,
                    "concept_text": c["concept_text"] if c else None,
                    "concept_source": c["source"] if c else None,
                    **a,
                })
            f.write(json.dumps({"note_id": note_id, "entities": ents}) + "\n")
            total += len(ents)
    print(f"Wrote {total} entities to {args.output}")


if __name__ == "__main__":
    main()
