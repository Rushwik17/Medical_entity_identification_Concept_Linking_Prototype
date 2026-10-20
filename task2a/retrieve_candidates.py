from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
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
        t = tok(
            texts[i : i + bs],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            h = model(**t).last_hidden_state
            mask = t["attention_mask"].unsqueeze(-1).float()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(F.normalize(emb, dim=1).cpu())
    return torch.cat(out)

def load_umls(path: str) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, usecols=["SAB", "CODE", "STR"], chunksize=200_000):
        chunk = chunk[
            (chunk.SAB == "SNOMEDCT_US") & chunk.CODE.notna() & chunk.STR.notna()
        ]
        chunk = chunk[chunk.STR.str.strip().ne("")]
        parts.append(chunk[["CODE", "STR"]])
    umls = pd.concat(parts, ignore_index=True)
    umls = umls.drop_duplicates(subset=["CODE", "STR"]).reset_index(drop=True)
    umls["CODE"] = umls["CODE"].astype(str)
    umls["STR_CLEAN"] = umls["STR"].map(clean)
    umls = umls[umls.STR_CLEAN.str.len() > 0].reset_index(drop=True)
    return umls

def build_gold_queries(data_dir: str, with_context: bool, window: int):
    annots = (
        valid_annotations(data_dir)
        .drop_duplicates(["note_id", "start", "end", "concept_id"])
        .reset_index(drop=True)
    )

    if not with_context:
        queries = []
        for r in annots.itertuples():
            ent = clean(r.entity_text)
            queries.append(
                {
                    "note_id": r.note_id,
                    "start": int(r.start),
                    "end": int(r.end),
                    "entity_text": ent,
                    "concept_id": str(r.concept_id),
                    "query": ent,
                }
            )
        return queries

    notes = load_notes(data_dir)
    text_by_note = {str(nid): txt for nid, txt in zip(notes["note_id"], notes["text"])}

    queries = []
    for r in annots.itertuples():
        note_text = text_by_note.get(str(r.note_id), "")
        s = max(0, int(r.start) - window)
        e = min(len(note_text), int(r.end) + window)
        ctx = clean(note_text[s:e])
        ent = clean(r.entity_text)
        queries.append(
            {
                "note_id": r.note_id,
                "start": int(r.start),
                "end": int(r.end),
                "entity_text": ent,
                "concept_id": str(r.concept_id),
                "query": f"{ent} [SEP] {ctx}",
            }
        )
    return queries

def main(a):
    rank, world, local, is_dist = distributed_state()
    init_process_group(is_dist)
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local)

    umls = load_umls(a.umls)
    code_to_text = dict(zip(umls["CODE"].tolist(), umls["STR"].tolist()))
    if rank == 0:
        print(f"UMLS rows after filter: {len(umls)} | unique CODEs: {umls.CODE.nunique()}")

    queries = build_gold_queries(a.data_dir, a.with_context, a.context_window)
    if a.limit > 0:
        queries = queries[: a.limit]
    if rank == 0:
        print(f"Queries: {len(queries)}")

    s = len(umls) * rank // world
    e = len(umls) * (rank + 1) // world
    vocab = umls.iloc[s:e].reset_index(drop=True)
    vocab_codes = vocab["CODE"].tolist()  # python list, per-rank slice

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device).eval()

    if rank == 0:
        print(f"Encoding UMLS shard ({len(vocab)} rows)...")
    cv = encode(vocab.STR_CLEAN.tolist(), model, tok, device, a.batch_size).to(device)
    if rank == 0:
        print(f"[rank {rank}] Local vocab size: {len(vocab)}")

    if rank == 0:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(a.output, "w", encoding="utf-8") if rank == 0 else None

    topK_local = max(a.top_k * 2, 40)         # concepts kept per-rank per query
    M_local = min(topK_local * 8, len(vocab)) # rows pulled per-rank per query

    gold_topk_hits = {1: 0, 5: 0, 20: 0, 100: 0}
    gold_total = 0

    n_batches = (len(queries) + a.batch_size - 1) // a.batch_size
    iterator = range(0, len(queries), a.batch_size)
    if rank == 0:
        iterator = tqdm(iterator, total=n_batches, desc="Retrieving", unit="batch")

    for i in iterator:
        batch = queries[i : i + a.batch_size]
        Bq = len(batch)

        q = encode(
            [x["query"] for x in batch],
            model, tok, device, a.batch_size,
            max_length=a.max_query_length,
        ).to(device)

        score = q @ cv.T                       # (Bq, N_local)
        v, j = score.topk(M_local, dim=1)      # (Bq, M_local)

        v_cpu = v.cpu().tolist()
        j_cpu = j.cpu().tolist()

        local_scores = torch.zeros(Bq, topK_local, dtype=torch.float32)
        local_cids = torch.full((Bq, topK_local), -1, dtype=torch.long)

        for b in range(Bq):
            best = {}
            for sc, ix in zip(v_cpu[b], j_cpu[b]):
                cid = vocab_codes[ix]
                if cid not in best or sc > best[cid]:
                    best[cid] = sc
            ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:topK_local]
            for r, (cid, sc) in enumerate(ranked):
                local_scores[b, r] = sc
                local_cids[b, r] = int(cid)

        if is_dist:
            local_scores = local_scores.to(device)
            local_cids = local_cids.to(device)
            gathered_scores = [torch.zeros_like(local_scores) for _ in range(world)]
            gathered_cids = [torch.zeros_like(local_cids) for _ in range(world)]
            dist.all_gather(gathered_scores, local_scores)
            dist.all_gather(gathered_cids, local_cids)
            S = torch.stack(gathered_scores, dim=0).cpu()   # (W, Bq, K)
            C = torch.stack(gathered_cids, dim=0).cpu()     # (W, Bq, K)
        else:
            S = local_scores.unsqueeze(0).cpu()
            C = local_cids.unsqueeze(0).cpu()

        if rank == 0:
            S_l = S.tolist()
            C_l = C.tolist()
            for b in range(Bq):
                merged = {}
                for w in range(len(S_l)):
                    for r in range(len(S_l[w][b])):
                        cid = int(C_l[w][b][r])
                        if cid < 0:
                            continue
                        sc = float(S_l[w][b][r])
                        if cid not in merged or sc > merged[cid]:
                            merged[cid] = sc
                ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[: a.top_k]
                top = [
                    {
                        "concept_id": str(cid),
                        "concept_text": code_to_text.get(str(cid), ""),
                        "score": sc,
                    }
                    for cid, sc in ranked
                ]
                row = batch[b]
                gold = str(row["concept_id"])
                gold_total += 1
                for K in gold_topk_hits:
                    if any(c["concept_id"] == gold for c in top[:K]):
                        gold_topk_hits[K] += 1
                out_f.write(
                    json.dumps(
                        {
                            "note_id": row["note_id"],
                            "start": row["start"],
                            "end": row["end"],
                            "entity_text": row["entity_text"],
                            "concept_id": row["concept_id"],
                            "candidates": top,
                        }
                    )
                    + "\n"
                )
    if rank == 0:
        out_f.close()
        print("\n=== Concept-level recall on gold queries ===")
        for K in sorted(gold_topk_hits):
            r = gold_topk_hits[K] / max(gold_total, 1)
            print(f"  recall@{K:<4}: {r:.4f}")
    if is_dist:
        dist.destroy_process_group()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="../Data")
    p.add_argument("--umls", default="../Data/UMLS.csv")
    p.add_argument(
        "--model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="SapBERT checkpoint. If sentencepiece+protobuf are installed, "
             "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR is stronger.",
    )
    p.add_argument("--output", default="../outputs/sapbert_candidates_v2.jsonl")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--retrieve-k", type=int, default=100, help="(unused in v2, kept for CLI compat)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--with-context", action="store_true")
    p.add_argument("--context-window", type=int, default=200)
    p.add_argument("--max-query-length", type=int, default=64,
                   help="Raise to 128+ when --with-context.")
    p.add_argument("--limit", type=int, default=-1,
                   help="If >0, only use the first N queries (smoke test).")
    main(p.parse_args())