# task3/eval_umls.py
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def prf(g, p):
    tp = len(g & p)
    fp = len(p - g)
    fn = len(g - p)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--gold", default="task3/outputs/gold_umls.jsonl")
    a.add_argument("--pred", default="task3/outputs/umls_predictions.jsonl")
    a.add_argument(
        "--test-notes",
        default="",
        help="Optional JSON file: list of note_ids to restrict eval to. "
             "If empty, uses the full gold file.",
    )
    a = a.parse_args()

    G_all = []
    for r in read(a.gold):
        nid = str(r["note_id"])
        for e in r.get("entities", []):
            G_all.append((
                nid,
                int(e["start"]),
                int(e["end"]),
                str(e.get("snomed_id") or ""),
                str(e.get("cui") or ""),
            ))

    P_all = []
    for r in read(a.pred):
        nid = str(r["note_id"])
        for e in r.get("entities", []):
            P_all.append((
                nid,
                int(e["start"]),
                int(e["end"]),
                str(e.get("snomed_id") or ""),
                str(e.get("cui") or ""),
            ))

    # Optional test-set restriction
    if a.test_notes and Path(a.test_notes).exists():
        test_ids = set(json.loads(Path(a.test_notes).read_text()))
        G_all = [x for x in G_all if x[0] in test_ids]
        P_all = [x for x in P_all if x[0] in test_ids]
        print(f"Restricted to {len(test_ids)} test notes")

    print(f"Gold entities      : {len(G_all)}")
    print(f"Predicted entities : {len(P_all)}")
    print()

    # ---- Entity Recognition (span only) ----
    gs = {(n, s, e) for (n, s, e, _, _) in G_all}
    ps = {(n, s, e) for (n, s, e, _, _) in P_all}
    ep, er, ef = prf(gs, ps)
    print("Entity Recognition")
    print(f"Precision : {ep:.4f}")
    print(f"Recall    : {er:.4f}")
    print(f"F1        : {ef:.4f}")
    print()

    # ---- Concept Linking (CUI accuracy on matched spans) ----
    gm = {(n, s, e): (sn, cui) for (n, s, e, sn, cui) in G_all}
    pm = {(n, s, e): (sn, cui) for (n, s, e, sn, cui) in P_all}
    common = gs & ps
    correct_cui = 0
    correct_snomed = 0
    for k in common:
        gsn, gcui = gm[k]
        psn, pcui = pm[k]
        if gcui and pcui and gcui == pcui:
            correct_cui += 1
        if gsn and psn and gsn == psn:
            correct_snomed += 1

    print("Concept Linking")
    print(f"Matched spans      : {len(common)}")
    print(f"Correct CUI        : {correct_cui}/{len(common)} "
          f"= {correct_cui / len(common) if common else 0:.4f}")
    print(f"Correct SNOMED ID  : {correct_snomed}/{len(common)} "
          f"= {correct_snomed / len(common) if common else 0:.4f}")
    print()

    # ---- End-to-End ----
    # Correct = span AND CUI both match.
    ge_cui = {(n, s, e, cui) for (n, s, e, _, cui) in G_all if cui}
    pe_cui = {(n, s, e, cui) for (n, s, e, _, cui) in P_all if cui}
    xp, xr, xf = prf(ge_cui, pe_cui)
    print("End-to-End (span + CUI)")
    print(f"Precision : {xp:.4f}")
    print(f"Recall    : {xr:.4f}")
    print(f"F1        : {xf:.4f}")


if __name__ == "__main__":
    main()
