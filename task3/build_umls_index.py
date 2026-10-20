# task3/build_umls_index.py
from __future__ import annotations

import argparse
import gc
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

from src.distributed_utils import distributed_state, init_process_group


_WS = re.compile(r"\s+")


def clean(s: str) -> str:
    return _WS.sub(" ", str(s)).strip().lower()


# --- Tighter SAB set: only sources that carry concepts useful for clinical NLP ---
KEEP_SAB = {"SNOMEDCT_US", "MSH", "RXNORM", "LNC", "HPO", "OMIM", "DRUGBANK"}

# --- Preference order when multiple SABs hold the same (CUI, cleaned string) ---
SAB_PRIORITY = {
    "SNOMEDCT_US": 0,
    "MSH": 1,
    "RXNORM": 2,
    "LNC": 3,
    "HPO": 4,
    "OMIM": 5,
    "DRUGBANK": 6,
}


def load_filtered_umls(path: str) -> pd.DataFrame:
    """
    Load UMLS with tight filters:
      - English only
      - not suppressed
      - curated SABs
      - dedup exact (CUI, SAB, CODE, STR)
      - dedup to one row per (CUI, cleaned STR), preferring SNOMED > MSH > ...
    """
    parts = []
    for ch in pd.read_csv(
        path,
        usecols=["CUI", "LAT", "SUPPRESS", "SAB", "CODE", "STR"],
        chunksize=500_000,
        dtype=str,
    ):
        ch = ch[
            (ch.LAT == "ENG")
            & (ch.SUPPRESS == "N")
            & ch.SAB.isin(KEEP_SAB)
            & ch.CUI.notna() & ch.CODE.notna() & ch.STR.notna()
        ]
        ch = ch[ch.STR.str.strip().ne("")]
        parts.append(ch[["CUI", "SAB", "CODE", "STR"]])

    umls = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    umls = umls.drop_duplicates(subset=["CUI", "SAB", "CODE", "STR"]).reset_index(drop=True)
    umls["CUI"] = umls["CUI"].astype(str)
    umls["CODE"] = umls["CODE"].astype(str)
    umls["STR_CLEAN"] = umls["STR"].map(clean)
    umls = umls[umls.STR_CLEAN.str.len() > 0].reset_index(drop=True)

    # ---- Dedup to one row per (CUI, cleaned string) ----
    umls["_prio"] = umls["SAB"].map(SAB_PRIORITY).fillna(99).astype(int)
    umls = umls.sort_values(["CUI", "STR_CLEAN", "_prio"])
    umls = umls.drop_duplicates(subset=["CUI", "STR_CLEAN"], keep="first")
    umls = umls.drop(columns=["_prio"]).reset_index(drop=True)

    return umls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--umls", default="Data/UMLS.csv")
    p.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    p.add_argument("--out-dir", default="task3/outputs/umls_index")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--force-rebuild-meta", action="store_true")
    args = p.parse_args()

    rank, world, local, is_dist = distributed_state()
    init_process_group(is_dist)
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "umls_meta.parquet"

    # ---- Rank 0 builds metadata once ----
    need_meta = (not meta_path.exists()) or args.force_rebuild_meta
    if rank == 0 and need_meta:
        print("[rank 0] Building UMLS metadata (filtered)...")
        umls = load_filtered_umls(args.umls)
        print(f"[rank 0] Rows: {len(umls):,} | unique CUIs: {umls.CUI.nunique():,}")
        umls.to_parquet(meta_path, index=False)
        del umls
        gc.collect()
        print(f"[rank 0] Saved {meta_path}")

    if is_dist:
        dist.barrier()

    print(f"[rank {rank}] Loading metadata from {meta_path}")
    umls = pd.read_parquet(meta_path)
    print(f"[rank {rank}] Rows: {len(umls):,}")

    # ---- Shard rows across ranks ----
    s = len(umls) * rank // world
    e = len(umls) * (rank + 1) // world
    vocab = umls.iloc[s:e].reset_index(drop=True)
    del umls
    gc.collect()
    print(f"[rank {rank}] Local shard: {len(vocab):,} rows")

    # ---- Load model ----
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval().half()

    # ---- Encode shard ----
    texts = vocab["STR_CLEAN"].tolist()
    embs = []
    for i in tqdm(
        range(0, len(texts), args.batch_size),
        desc=f"rank{rank} encoding",
        disable=(rank != 0),
    ):
        t = tok(
            texts[i : i + args.batch_size],
            padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            h = model(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1).half()
            emb = F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)
        embs.append(emb.cpu().half())
        del t, h, m, emb
    local_emb = torch.cat(embs)
    del embs
    gc.collect()

    torch.save(local_emb, out_dir / f"emb_rank{rank}.pt")
    vocab.to_parquet(out_dir / f"meta_rank{rank}.parquet", index=False)
    print(f"[rank {rank}] Saved {len(local_emb):,} rows")

    if is_dist:
        dist.barrier()

    # ---- Rank 0 concatenates ----
    if rank == 0:
        print("[rank 0] Concatenating shards...")
        all_emb, all_meta = [], []
        for r in range(world):
            all_emb.append(torch.load(out_dir / f"emb_rank{r}.pt", map_location="cpu"))
            all_meta.append(pd.read_parquet(out_dir / f"meta_rank{r}.parquet"))
        full_emb = torch.cat(all_emb, dim=0)
        full_meta = pd.concat(all_meta, ignore_index=True)
        torch.save(full_emb, out_dir / "umls_emb.pt")
        full_meta.to_parquet(meta_path, index=False)
        print(f"[rank 0] Final index: emb {tuple(full_emb.shape)} | rows {len(full_meta):,}")

        # cleanup per-rank shards
        for r in range(world):
            (out_dir / f"emb_rank{r}.pt").unlink(missing_ok=True)
            (out_dir / f"meta_rank{r}.parquet").unlink(missing_ok=True)

    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
