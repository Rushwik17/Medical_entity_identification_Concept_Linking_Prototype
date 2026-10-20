# task3/retrieve_umls_cands.py
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.data_utils import valid_annotations, load_notes
from src.distributed_utils import distributed_state, init_process_group


_WS = re.compile(r"\s+")


def clean(s: str) -> str:
    return _WS.sub(" ", str(s)).strip().lower()


def encode(texts, model, tok, device, bs, max_length=64):
    out = []
    for i in range(0, len(texts), bs):
        t = tok(texts[i:i + bs], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).half()
            e = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)
        out.append(e.cpu().half())
    return torch.cat(out)


def build_gold_queries(data_dir: str, with_context: bool, window: int):
    annots = (
        valid_annotations(data_dir)
        .drop_duplicates(["note_id", "start", "end", "concept_id"])
        .reset_index(drop=True)
    )
    if not with_context:
        return [
            {
                "note_id": str(r.note_id),
                "start": int(r.start),
                "end": int(r.end),
                "entity_text": clean(r.entity_text),
                "snomed_id": str(r.concept_id),
                "query": clean(r.entity_text),
            }
            for r in annots.itertuples()
        ]

    notes = load_notes(data_dir)
    text_by_note = {str(nid): txt for nid, txt in zip(notes["note_id"], notes["text"])}
    out = []
    for r in annots.itertuples():
        note_text = text_by_note.get(str(r.note_id), "")
        s = max(0, int(r.start) - window)
        e = min(len(note_text), int(r.end) + window)
        ctx = clean(note_text[s:e])
        ent = clean(r.entity_text)
        out.append({
            "note_id": str(r.note_id),
            "start": int(r.start),
            "end": int(r.end),
            "entity_text": ent,
            "snomed_id": str(r.concept_id),
            "query": f"{ent} [SEP] {ctx}",
        })
    return out


def main(a):
    rank, world, local, is_dist = distributed_state()
    init_process_group(is_dist)
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local)

    # ---- Load cached UMLS index (fp16) ----
    idx_dir = Path(a.index_dir)
    emb_full = torch.load(idx_dir / "umls_emb.pt", map_location="cpu")  # fp16 CPU
    meta_full = pd.read_parquet(idx_dir / "umls_meta.parquet")
    if rank == 0:
        print(f"Index: {tuple(emb_full.shape)} | rows {len(meta_full):,}")

    # ---- Shard index across ranks ----
    s = len(meta_full) * rank // world
    e = len(meta_full) * (rank + 1) // world
    meta = meta_full.iloc[s:e].reset_index(drop=True)
    emb = emb_full[s:e].to(device)

    cuis = meta["CUI"].tolist()
    strs = meta["STR"].tolist()
    sabs = meta["SAB"].tolist()
    codes = meta["CODE"].tolist()

    # ---- Build queries ----
    queries = build_gold_queries(a.data_dir, a.with_context, a.context_window)
    if a.limit > 0:
        queries = queries[: a.limit]
    if rank == 0:
        print(f"Queries: {len(queries)}")

    # ---- Load SapBERT ----
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device).eval().half()

    if rank == 0:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(a.output, "w", encoding="utf-8") if rank == 0 else None

    topK_local = max(a.top_k * 2, 40)
    M_local = min(topK_local * 8, len(meta))

    n_batches = (len(queries) + a.batch_size - 1) // a.batch_size
    iterator = range(0, len(queries), a.batch_size)
    if rank == 0:
        iterator = tqdm(iterator, total=n_batches, desc="Retrieving", unit="batch")

    for i in iterator:
        batch = queries[i:i + a.batch_size]
        Bq = len(batch)

        q = encode([x["query"] for x in batch], model, tok, device,
                   a.batch_size, max_length=a.max_query_length).to(device)

        score = q @ emb.T
        v, j = score.topk(M_local, dim=1)
        v_cpu = v.float().cpu().tolist()
        j_cpu = j.cpu().tolist()

        local_scores = torch.zeros(Bq, topK_local, dtype=torch.float32)
        local_rows = torch.full((Bq, topK_local), -1, dtype=torch.long)

        for b in range(Bq):
            best = {}
            for sc, ix in zip(v_cpu[b], j_cpu[b]):
                cui = cuis[ix]
                if cui not in best or sc > best[cui][0]:
                    best[cui] = (sc, ix)
            ranked = sorted(best.items(), key=lambda x: x[1][0], reverse=True)[:topK_local]
            for r_, (cui, (sc, ix)) in enumerate(ranked):
                local_scores[b, r_] = sc
                local_rows[b, r_] = ix

        if is_dist:
            local_scores = local_scores.to(device)
            local_rows = local_rows.to(device)
            g_scores = [torch.zeros_like(local_scores) for _ in range(world)]
            g_rows = [torch.zeros_like(local_rows) for _ in range(world)]
            dist.all_gather(g_scores, local_scores)
            dist.all_gather(g_rows, local_rows)
            S = torch.stack(g_scores, dim=0).cpu()
            C = torch.stack(g_rows, dim=0).cpu()
        else:
            S = local_scores.unsqueeze(0).cpu()
            C = local_rows.unsqueeze(0).cpu()

        if rank == 0:
            S_l = S.tolist()
            C_l = C.tolist()
            for b in range(Bq):
                merged = {}
                for w_ in range(len(S_l)):
                    offset = len(meta_full) * w_ // world
                    for r_ in range(len(S_l[w_][b])):
                        ix = int(C_l[w_][b][r_])
                        if ix < 0:
                            continue
                        sc = float(S_l[w_][b][r_])
                        gi = offset + ix
                        row = meta_full.iloc[gi]
                        cui = row.CUI
                        if cui not in merged or sc > merged[cui]["score"]:
                            merged[cui] = {
                                "cui": str(cui),
                                "concept_text": str(row.STR),
                                "snomed_code": str(row.CODE) if row.SAB == "SNOMEDCT_US" else None,
                                "source": str(row.SAB),
                                "score": sc,
                            }
                ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
                top = ranked[:a.top_k]

                out_f.write(json.dumps({
                    "note_id": batch[b]["note_id"],
                    "start": batch[b]["start"],
                    "end": batch[b]["end"],
                    "entity_text": batch[b]["entity_text"],
                    "snomed_id": batch[b]["snomed_id"],
                    "candidates": top,
                }) + "\n")

    if rank == 0:
        out_f.close()
        print(f"Wrote candidates to {a.output}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="Data")
    p.add_argument("--index-dir", default="task3/outputs/umls_index")
    p.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--output", default="task3/outputs/umls_candidates.jsonl")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--with-context", action="store_true")
    p.add_argument("--context-window", type=int, default=200)
    p.add_argument("--max-query-length", type=int, default=64)
    p.add_argument("--limit", type=int, default=-1)
    main(p.parse_args())
