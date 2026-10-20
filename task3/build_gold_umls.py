# task3/build_gold_umls.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data_utils import valid_annotations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="Data")
    p.add_argument("--umls", default="Data/UMLS.csv")
    p.add_argument("--out", default="task3/outputs/gold_umls.jsonl")
    args = p.parse_args()

    # SNOMED CT CODE -> UMLS CUI
    print("Building SNOMED -> CUI map...")
    s2c = {}
    for ch in pd.read_csv(args.umls, usecols=["CUI", "SAB", "CODE"], chunksize=200_000):
        ch = ch[(ch.SAB == "SNOMEDCT_US") & ch.CUI.notna() & ch.CODE.notna()]
        for r in ch.itertuples():
            s2c.setdefault(str(r.CODE), str(r.CUI))
    print(f"  mapped {len(s2c):,} codes")

    annots = valid_annotations(args.data_dir)
    by_note = {}
    miss = 0
    total = 0
    for r in annots.itertuples():
        nid = str(r.note_id)
        sn = str(r.concept_id)
        cui = s2c.get(sn)
        if cui is None:
            miss += 1
        by_note.setdefault(nid, []).append({
            "start": int(r.start),
            "end": int(r.end),
            "text": str(r.entity_text),
            "snomed_id": sn,
            "cui": cui,
        })
        total += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for nid, ents in by_note.items():
            f.write(json.dumps({"note_id": nid, "entities": ents}) + "\n")

    print(f"Wrote gold for {len(by_note)} notes; total entities {total}")
    print(f"SNOMED codes without CUI: {miss}")


if __name__ == "__main__":
    main()
