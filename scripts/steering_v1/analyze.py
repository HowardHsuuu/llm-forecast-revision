"""Analyze the frozen verdict-steering confirmation and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


OPPOSITE = {"zero", "high_plus", "high_orth_plus", "low_plus", "low_orth_plus", "high_minus"}
NEUTRAL = {"zero", "high_plus"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(values: list[float], seed: int = 20261116) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(10000, len(array)))
    return {"n": len(array), "mean": float(array.mean()),
            "ci95": np.quantile(array[draws].mean(axis=1), (0.025, 0.975)).tolist()}


def measures(rows: list[dict], context: str, key: str) -> dict:
    outputs = [r["contexts"][context]["conditions"][key] for r in rows]
    zero = [r["contexts"][context]["conditions"]["zero"] for r in rows]
    return {"n": len(rows), "parsed": sum(z["parse_ok"] for z in outputs),
            "prefix_ok": sum(z["prefix_ok"] for z in outputs),
            "retained": sum(z["parse_ok"] and z["verdict"] == r["first_verdict"]
                            for r, z in zip(rows, outputs)),
            "accept": sum(z["parse_ok"] and z["verdict"] == "accept" for z in outputs),
            "answer_changed": sum(z["response"] != b["response"]
                                  for z, b in zip(outputs, zero)),
            "inconsistent": sum(z["parse_ok"] and
                                ((z["p_accept"] >= 50) != (z["verdict"] == "accept"))
                                for z in outputs),
            "p_shift": estimate([z["p_accept"] - b["p_accept"]
                                 for z, b in zip(outputs, zero) if z["parse_ok"]]),
            "edited_steps": estimate([z["edited_generation_steps"] for z in outputs]),
            "relative_dose": estimate([z["mean_relative_dose"] for z in outputs
                                       if z["mean_relative_dose"] is not None])}


def summarize(rows: list[dict]) -> dict:
    opposite = {key: measures(rows, "opposite", key) for key in sorted(OPPOSITE)}
    neutral = {key: measures(rows, "neutral", key) for key in sorted(NEUTRAL)}
    effects = {}
    for dose in ("high", "low"):
        plus = [r["contexts"]["opposite"]["conditions"][f"{dose}_plus"] for r in rows]
        orth = [r["contexts"]["opposite"]["conditions"][f"{dose}_orth_plus"] for r in rows]
        effects[dose] = {"plus_orth_retention_difference": estimate([
            float(p["parse_ok"] and p["verdict"] == r["first_verdict"]) -
            float(o["parse_ok"] and o["verdict"] == r["first_verdict"])
            for r, p, o in zip(rows, plus, orth)]),
            "plus_orth_p_difference": estimate([
                p["p_accept"] - o["p_accept"] for p, o in zip(plus, orth)
                if p["parse_ok"] and o["parse_ok"]])}
    return {"n": len(rows), "opposite": opposite, "neutral": neutral, "effects": effects}


def main(baseline: Path, run: Path) -> None:
    records = run / "records.jsonl"
    marker = json.loads((run / "completion.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    baseline_marker = json.loads((baseline / "completion.json").read_text())
    baseline_rows = [json.loads(s) for s in (baseline / "records.jsonl").read_text().splitlines() if s]
    rows = [json.loads(s) for s in records.read_text().splitlines() if s]
    if (len(rows) != 160 or sha(records) != marker["records_sha256"] or
            len(rows) != marker["total_records"] or
            len({r["case_id"] for r in rows}) != len(rows) or
            manifest["baseline_records_sha256"] != baseline_marker["records_sha256"] or
            baseline_marker["records_sha256"] != sha(baseline / "records.jsonl") or
            set(manifest["opposite_conditions"]) != OPPOSITE or
            set(manifest["neutral_conditions"]) != NEUTRAL or
            any(set(r["contexts"]) != {"neutral", "opposite"} or
                set(r["contexts"]["opposite"]["conditions"]) != OPPOSITE or
                set(r["contexts"]["neutral"]["conditions"]) != NEUTRAL or
                any(not z["parse_ok"] or not z["prefix_ok"]
                    for context in r["contexts"].values()
                    for z in context["conditions"].values())
                for r in rows)):
        raise ValueError("confirmation run incomplete or changed")
    if len({r["case_id"] for r in baseline_rows if r["round"] == 1}) != 160:
        raise ValueError("baseline first-round cases incomplete")
    groups = {
        "all": rows,
        "initial_accept_flip": [r for r in rows if r["first_verdict"] == "accept" and
                                r["contexts"]["opposite"]["conditions"]["zero"]["verdict"] == "reject"],
        "initial_accept_stable": [r for r in rows if r["first_verdict"] == "accept" and
                                  r["contexts"]["opposite"]["conditions"]["zero"]["verdict"] == "accept"],
        "initial_reject": [r for r in rows if r["first_verdict"] == "reject"],
        "grad_flip": [r for r in rows if r["domain"] == "grad_admissions" and
                      r["first_verdict"] == "accept" and
                      r["contexts"]["opposite"]["conditions"]["zero"]["verdict"] == "reject"],
        "paper_flip": [r for r in rows if r["domain"] == "paper_acceptance" and
                       r["first_verdict"] == "accept" and
                       r["contexts"]["opposite"]["conditions"]["zero"]["verdict"] == "reject"],
    }
    complete_ids = {r["case_id"] for r in rows}
    first = [r for r in baseline_rows if r["round"] == 1]
    excluded = {r["case_id"] for r in first} - complete_ids
    exclusion_reasons = {}
    by_id = {}
    for r in baseline_rows:
        by_id.setdefault(r["case_id"], {})["first" if r["round"] == 1 else r["condition"]] = r
    for case_id in sorted(excluded):
        pair = by_id[case_id]
        exclusion_reasons[case_id] = [key for key in ("first", "neutral", "opposite")
                                      if key not in pair or not pair[key]["parse_ok"]]
    result = {"protocol": "generation-steering-confirmation-analysis-v1",
              "records_sha256": marker["records_sha256"],
              "baseline_records_sha256": baseline_marker["records_sha256"],
              "baseline_n": 160, "analyzed_n": len(rows),
              "excluded_case_ids_and_reasons": exclusion_reasons,
              "groups": {name: summarize(group) for name, group in groups.items()}}
    (run / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"analyzed_n": len(rows), "excluded": len(excluded),
                      "flip_n": len(groups["initial_accept_flip"]),
                      "high_plus_flip_retained": result["groups"]["initial_accept_flip"]["opposite"]["high_plus"]["retained"],
                      "high_orth_flip_retained": result["groups"]["initial_accept_flip"]["opposite"]["high_orth_plus"]["retained"],
                      "high_plus_orth_effect": result["groups"]["initial_accept_flip"]["effects"]["high"]["plus_orth_retention_difference"]},
                     sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    main(args.baseline, args.run)
