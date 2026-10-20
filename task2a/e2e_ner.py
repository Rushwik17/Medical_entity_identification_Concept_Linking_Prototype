from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
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
from src.data_utils import load_notes, valid_annotations

_WS = re.compile(r"\s+")

def clean(s: str) -> str:
    return _WS.sub(" ", str(s)).strip().lower()

@torch.no_grad()
def predict_spans_for_note(
    text: str,
    tokenizer,
    model,
    device,
    label_names,
    max_length=512,
    stride=128,
    batch_size=4,
):
    chunk_size = max_length - 2  # reserve CLS/SEP
    step = max(1, chunk_size - stride)

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
        return []

    starts = list(range(0, n, step))
    chunks = []
    for s in starts:
        e = min(s + chunk_size, n)
        if e - s < 4:
            continue
        chunks.append((s, e))
    if chunks and chunks[-1][1] < n:
        chunks.append((max(0, n - chunk_size), n))

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id or 0

    all_preds = []  # list of (offset_index_in_note, label_id)

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        ids_list = []
        attn_list = []
        for s, e in batch_chunks:
            ids = [cls_id] + input_ids[s:e] + [sep_id]
            attn = [1] * len(ids)
            ids_list.append(ids)
            attn_list.append(attn)

        maxlen = max(len(x) for x in ids_list)
        padded_ids = [x + [pad_id] * (maxlen - len(x)) for x in ids_list]
        padded_attn = [x + [0] * (maxlen - len(x)) for x in attn_list]

        input_tensor = torch.tensor(padded_ids, device=device)
        attn_tensor = torch.tensor(padded_attn, device=device)

        logits = model(input_ids=input_tensor, attention_mask=attn_tensor).logits
        preds = logits.argmax(dim=-1).cpu().numpy()  # (B, L)

        for bi, (s, e) in enumerate(batch_chunks):
            for j in range(1, 1 + (e - s)):
                global_idx = s + (j - 1)
                all_preds.append((global_idx, int(preds[bi, j])))

    pred_map = {}
    for gi, lab in all_preds:
        pred_map[gi] = lab

    labels_seq = [pred_map.get(i, 0) for i in range(n)]

    spans = []
    cur_start = None
    cur_end = None
    for i, lab in enumerate(labels_seq):
        name = label_names[lab]
        if name == "B-MEDICAL":
            if cur_start is not None:
                spans.append((cur_start, cur_end))
            cur_start, cur_end = offsets[i]
        elif name == "I-MEDICAL":
            if cur_start is None:
                cur_start, cur_end = offsets[i]
            else:
                cur_end = offsets[i][1]
        else:
            if cur_start is not None:
                spans.append((cur_start, cur_end))
                cur_start = cur_end = None
    if cur_start is not None:
        spans.append((cur_start, cur_end))

    spans = [(s, e) for (s, e) in spans if e - s >= 2]
    return spans

def build_or_load_index(args, device):
    idx_dir = Path(args.index_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)
    emb_path = idx_dir / "umls_emb.pt"
    meta_path = idx_dir / "umls_meta.parquet"

    if emb_path.exists() and meta_path.exists():
        print(f"Loading cached index from {idx_dir}")
        emb = torch.load(emb_path, map_location="cpu")
        meta = pd.read_parquet(meta_path)
        return emb, meta

    print("Building UMLS index (first time, ~5 min)...")
    parts = []
    for chunk in pd.read_csv(args.umls, usecols=["SAB", "CODE", "STR"], chunksize=200_000):
        chunk = chunk[
            (chunk.SAB == "SNOMEDCT_US") & chunk.CODE.notna() & chunk.STR.notna()
        ]
        chunk = chunk[chunk.STR.str.strip().ne("")]
        parts.append(chunk[["CODE", "STR"]])
    meta = pd.concat(parts, ignore_index=True)
    meta = meta.drop_duplicates(subset=["CODE", "STR"]).reset_index(drop=True)
    meta["CODE"] = meta["CODE"].astype(str)
    meta["STR_CLEAN"] = meta["STR"].map(clean)
    meta = meta[meta.STR_CLEAN.str.len() > 0].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.sapbert)
    model = AutoModel.from_pretrained(args.sapbert).to(device).eval()

    embs = []
    texts = meta["STR_CLEAN"].tolist()
    bs = 256
    for i in tqdm(range(0, len(texts), bs), desc="Encoding UMLS"):
        t = tok(
            texts[i : i + bs],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            h = model(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        embs.append(F.normalize(e, dim=1).cpu())
    emb = torch.cat(embs)

    torch.save(emb, emb_path)
    meta.to_parquet(meta_path, index=False)
    print(f"Saved index to {idx_dir}")
    return emb, meta

@torch.no_grad()
def retrieve_candidates(
    entity_texts, sapbert, sap_tok, emb, meta, device,
    retrieve_k=20, batch_size=64,
):
    results = []
    for i in range(0, len(entity_texts), batch_size):
        batch = [clean(t) for t in entity_texts[i : i + batch_size]]
        t = sap_tok(
            batch, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(device)
        h = sapbert(**t).last_hidden_state
        m = t["attention_mask"].unsqueeze(-1).float()
        q = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1).cpu()

        scores = q @ emb.T  # (B, N)
        v, j = scores.topk(retrieve_k * 5, dim=1)

        codes = meta["CODE"].tolist()
        strs = meta["STR"].tolist()
        for b in range(len(batch)):
            best = {}
            for sc, ix in zip(v[b].tolist(), j[b].tolist()):
                cid = codes[ix]
                if cid not in best or sc > best[cid][0]:
                    best[cid] = (sc, strs[ix])
            ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)[:retrieve_k]
            results.append([
                {"concept_id": str(cid), "concept_text": txt, "score": float(sc)}
                for cid, (sc, txt) in ranked
            ])
    return results

@torch.no_grad()
def rerank_with_linker(
    entity_texts, candidate_lists, linker, link_tok, device, batch_size=64
):
    chosen = []
    for i in range(0, len(entity_texts), batch_size):
        e_batch = entity_texts[i : i + batch_size]
        c_batch = candidate_lists[i : i + batch_size]

        flat_entities = []
        flat_concepts = []
        offsets = []  # per entity, index into flat list
        for e, cands in zip(e_batch, c_batch):
            offsets.append((len(flat_entities), len(flat_entities) + len(cands)))
            for c in cands:
                flat_entities.append(e)
                flat_concepts.append(c["concept_text"])

        if not flat_entities:
            chosen.extend([None] * len(e_batch))
            continue

        enc = link_tok(
            flat_entities, flat_concepts,
            padding=True, truncation=True, max_length=64, return_tensors="pt",
        ).to(device)
        logits = linker(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        for (s, e), cands in zip(offsets, c_batch):
            if e <= s:
                chosen.append(None)
                continue
            sub = probs[s:e]
            best = int(np.argmax(sub))
            chosen.append(cands[best]["concept_id"])
    return chosen

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="../Data")
    p.add_argument("--ner-model", default="../outputs/clinical-bert-ner")
    p.add_argument("--sapbert", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--linker", default="../outputs/sapbert-linker")
    p.add_argument("--umls", default="../Data/UMLS.csv")
    p.add_argument("--index-dir", default="../outputs/sapbert_index")
    p.add_argument("--out-pred", default="../outputs/predictions.jsonl")
    p.add_argument("--out-gold", default="../outputs/test_gold.jsonl")
    p.add_argument("--eval-note-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--retrieve-k", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    notes = load_notes(args.data_dir)
    annots = valid_annotations(args.data_dir)

    import random
    note_ids = sorted(set(str(x) for x in notes["note_id"]))
    rng = random.Random(args.seed)
    rng.shuffle(note_ids)
    n_test = max(1, int(args.eval_note_frac * len(note_ids)))
    test_notes = set(note_ids[:n_test])
    print(f"Evaluating on {len(test_notes)} held-out notes")

    test_note_rows = notes[notes["note_id"].astype(str).isin(test_notes)]

    gold_rows = []
    for note_id in test_notes:
        sub = annots[annots["note_id"].astype(str) == note_id]
        ents = []
        for r in sub.itertuples():
            ents.append({
                "start": int(r.start),
                "end": int(r.end),
                "text": str(r.entity_text),
                "snomed_id": str(r.concept_id),
            })
        gold_rows.append({"note_id": note_id, "entities": ents})
    Path(args.out_gold).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_gold, "w") as f:
        for r in gold_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote gold to {args.out_gold} ({sum(len(r['entities']) for r in gold_rows)} entities)")

    ner_tok = AutoTokenizer.from_pretrained(args.ner_model)
    ner_model = AutoModelForTokenClassification.from_pretrained(args.ner_model).to(device).eval()
    labels = list(ner_model.config.id2label.values())
    print("NER labels:", labels)

    sap_tok = AutoTokenizer.from_pretrained(args.sapbert)
    sapbert = AutoModel.from_pretrained(args.sapbert).to(device).eval()
    emb, meta = build_or_load_index(args, device)
    print(f"UMLS index size: {len(meta)}")

    link_tok = AutoTokenizer.from_pretrained(args.linker)
    linker = AutoModelForSequenceClassification.from_pretrained(args.linker).to(device).eval()

    per_note_spans = {}
    for row in tqdm(test_note_rows.itertuples(), total=len(test_note_rows), desc="NER"):
        text = row.text
        note_id = str(row.note_id)
        spans = predict_spans_for_note(
            text, ner_tok, ner_model, device, labels,
            max_length=args.max_length, stride=args.stride,
        )
        per_note_spans[note_id] = [
            (s, e, text[s:e]) for s, e in spans
        ]

    flat_entities = []
    flat_index = []  # (note_id, start, end, text)
    for note_id, spans in per_note_spans.items():
        for (s, e, t) in spans:
            flat_entities.append(t)
            flat_index.append((note_id, s, e, t))

    print(f"Total predicted spans: {len(flat_entities)}")
    cand_lists = retrieve_candidates(
        flat_entities, sapbert, sap_tok, emb, meta, device,
        retrieve_k=args.retrieve_k,
    )

    chosen = rerank_with_linker(
        flat_entities, cand_lists, linker, link_tok, device,
    )

    by_note = {}
    for (note_id, s, e, t), cid in zip(flat_index, chosen):
        by_note.setdefault(note_id, []).append({
            "start": int(s),
            "end": int(e),
            "text": str(t),
            "snomed_id": str(cid) if cid is not None else "",
        })

    with open(args.out_pred, "w") as f:
        for note_id in sorted(test_notes):
            ents = by_note.get(note_id, [])
            f.write(json.dumps({"note_id": note_id, "entities": ents}) + "\n")
    print(f"Wrote predictions to {args.out_pred}")


if __name__ == "__main__":
    main()