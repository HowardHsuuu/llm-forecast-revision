"""Summarize the fixed one-step localization checks without case outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


STEPS_SHA = "2fafd8c5bd0569bffca470eb988e63f34b43461b79b560cc07876e8fe59c3c91"
SUSTAINED_SHA = "362482f1fd04d48095ae146321194d02e9b0ef18f546b54c2ad503c0971a7b9d"
LOCI = ("first_number", "first_verdict")
CONDITIONS = {"zero", "plus", "minus", "orth_plus"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float], seed: int = 20261116) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"n": 0, "mean": None, "ci95": None}
    draws = np.random.default_rng(seed).integers(0, len(x), size=(10000, len(x)))
    return {"n": len(x), "mean": float(x.mean()),
            "ci95": np.quantile(x[draws].mean(axis=1), (0.025, 0.975)).tolist()}


def main(sustained: Path, run: Path) -> None:
    records = run / "records.jsonl"
    marker = json.loads((run / "completion.json").read_text())
    rows = [json.loads(s) for s in records.read_text().splitlines() if s]
    sustained_file = sustained / "records.jsonl"
    sustained_marker = json.loads((sustained / "completion.json").read_text())
    prior = {r["case_id"]: r for r in
             (json.loads(s) for s in sustained_file.read_text().splitlines() if s)}
    if (len(rows) != 160 or len({r["case_id"] for r in rows}) != 160 or
            marker["total_records"] != 160 or sha(records) != STEPS_SHA or
            marker["records_sha256"] != STEPS_SHA or
            sustained_marker["records_sha256"] != SUSTAINED_SHA or
            sha(sustained_file) != SUSTAINED_SHA or
            set(prior) != {r["case_id"] for r in rows} or
            any(set(r["loci"]) != set(LOCI) or
                any(set(r["loci"][locus]) != CONDITIONS or
                    any(not z["parse_ok"] for z in r["loci"][locus].values())
                    for locus in LOCI) or
                any(z["p_accept"] != r["natural_opposite_p"]
                    for z in r["loci"]["first_verdict"].values())
                for r in rows)):
        raise ValueError("fixed decision-step data are incomplete or changed")
    flip = [r for r in rows if r["first_verdict"] == "accept" and
            r["natural_opposite_verdict"] == "reject"]
    stable = [r for r in rows if r["first_verdict"] == "accept" and
              r["natural_opposite_verdict"] == "accept"]
    initial_reject = [r for r in rows if r["first_verdict"] == "reject"]
    if (len(flip), len(stable), len(initial_reject)) != (82, 64, 14):
        raise ValueError("frozen natural verdict groups changed")
    sustained_restored = {r["case_id"] for r in flip if
                          prior[r["case_id"]]["contexts"]["opposite"]
                          ["conditions"]["high_plus"]["verdict"] == "accept"}
    if len(sustained_restored) != 38:
        raise ValueError("sustained confirmation result changed")
    result = {"protocol": "decision-step-localization-summary-v1",
              "records_sha256": STEPS_SHA, "sustained_records_sha256": SUSTAINED_SHA,
              "counts": {"all": 160, "natural_flips": 82,
                         "natural_stable_accepts": 64, "initial_rejects": 14},
              "loci": {}}
    restored_ids = {}
    for locus in LOCI:
        plus = [r["loci"][locus]["plus"] for r in flip]
        orth = [r["loci"][locus]["orth_plus"] for r in flip]
        restored = {r["case_id"] for r, p in zip(flip, plus)
                    if p["verdict"] == "accept"}
        restored_ids[locus] = restored
        entry = {"plus_restored": len(restored),
                 "orth_restored": sum(z["verdict"] == "accept" for z in orth),
                 "paired_restoration_difference": interval([
                     float(p["verdict"] == "accept") - float(o["verdict"] == "accept")
                     for p, o in zip(plus, orth)]),
                 "sustained_overlap": len(restored & sustained_restored),
                 "stable_accepts_preserved": sum(r["loci"][locus]["plus"]["verdict"] == "accept"
                                                 for r in stable),
                 "initial_rejects_preserved": sum(r["loci"][locus]["plus"]["verdict"] == "reject"
                                                  for r in initial_reject),
                 "complete_percentage_changed": sum(
                     r["loci"][locus]["plus"]["p_accept"] != r["natural_opposite_p"]
                     for r in flip)}
        if locus == "first_verdict":
            entry["paired_accept_minus_reject_margin_shift"] = interval([
                p["accept_minus_reject_first_logit"] -
                o["accept_minus_reject_first_logit"]
                for p, o in zip(plus, orth)])
        result["loci"][locus] = entry
    result["restoration_overlap_between_loci"] = len(
        restored_ids["first_number"] & restored_ids["first_verdict"])
    (run / "public-summary.json").write_text(json.dumps(result, indent=2,
                                                         sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sustained", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.sustained, args.run)
