"""Summarize complete-number replacements before the verdict token."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from forecast_mechanisms.core import read_jsonl


CONDITIONS = {"opposite_natural_p", "opposite_first_p", "neutral_first_p"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows: list[dict]) -> dict:
    natural = [r["conditions"]["opposite_natural_p"] for r in rows]
    forced = [r["conditions"]["opposite_first_p"] for r in rows]
    neutral = [r["conditions"]["neutral_first_p"] for r in rows]
    return {
        "n": len(rows),
        "natural_verdict_tokens": dict(Counter(x["candidate"] for x in natural)),
        "forced_initial_percentage_tokens": dict(Counter(x["candidate"] for x in forced)),
        "neutral_initial_percentage_tokens": dict(Counter(x["candidate"] for x in neutral)),
        "forced_initial_percentage_accept": sum(x["candidate"] == "accept" for x in forced),
        "neutral_accept_opposite_reject_with_same_written_percentage": sum(
            n["candidate"] == "accept" and f["candidate"] == "reject"
            for n, f in zip(neutral, forced)),
        "mean_accept_minus_reject_logit_change": (
            sum(f["accept_minus_reject_logit"] - n["accept_minus_reject_logit"]
                for f, n in zip(forced, natural)) / len(rows) if rows else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    completion = json.loads((args.run / "completion.json").read_text())
    record_file = args.run / "records.jsonl"
    records = read_jsonl(record_file)
    if (manifest["protocol"] != "forced-number-verdict-v1" or
            set(manifest["conditions"]) != CONDITIONS or
            len(records) != len(manifest["case_ids"]) or
            len({r["case_id"] for r in records}) != len(records) or
            set(manifest["case_ids"]) != {r["case_id"] for r in records} or
            completion["total_records"] != len(records) or
            completion["records_sha256"] != digest(record_file)):
        raise ValueError("forced-number run incomplete or changed")
    for row in records:
        c = row["conditions"]
        if set(c) != CONDITIONS:
            raise ValueError("condition missing")
        source = c["opposite_natural_p"]["prefix_token_ids"]
        target = c["opposite_first_p"]["prefix_token_ids"]
        neutral = c["neutral_first_p"]["prefix_token_ids"]
        if (len(source) != 11 or len(target) != 11 or
                source[:5] != target[:5] or source[7:] != target[7:] or
                neutral != target or
                c["opposite_natural_p"]["token"] !=
                row["natural_opposite_verdict_token"] or
                c["opposite_natural_p"]["forced_p"] != row["opposite_p"] or
                c["opposite_first_p"]["forced_p"] != row["first_p"]):
            raise ValueError("nonnumeric prefix or sham differs")
        if row["first_p"] == row["opposite_p"] and (
                c["opposite_first_p"]["token"] != c["opposite_natural_p"]["token"] or
                abs(c["opposite_first_p"]["accept_minus_reject_logit"] -
                    c["opposite_natural_p"]["accept_minus_reject_logit"]) > 1e-5):
            raise ValueError("equal-number sham differs")
    if len(records) != manifest["full_cohort_cases"]:
        raise ValueError("analysis requires the complete cohort, not --only-case")
    flips = [r for r in records if r["first_verdict"] == "accept" and
             r["opposite_verdict"] == "reject"]
    primary = [r for r in flips if r["first_p"] != r["opposite_p"]]
    grad = [r for r in primary if r["domain"] == "grad_admissions"]
    paper = [r for r in primary if r["domain"] == "paper_acceptance"]
    if len(primary) != manifest["full_cohort_primary"]:
        raise ValueError("primary denominator changed")
    result = {
        "protocol": "forced-number-verdict-v1-summary",
        "records_sha256": digest(record_file),
        "groups": {
            "all": summary(records), "natural_flips": summary(flips),
            "primary_distinct_percentage_flips": summary(primary),
            "admissions_primary": summary(grad),
            "paper_primary": summary(paper),
            "admissions_primary_75": summary([r for r in grad if r["first_p"] == 75]),
            "paper_primary_75": summary([r for r in paper if r["first_p"] == 75]),
            "equal_percentage_flip_shams": summary(
                [r for r in flips if r["first_p"] == r["opposite_p"]]),
        },
    }
    (args.run / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: {"n": v["n"],
                          "forced_initial_percentage_accept": v[
                              "forced_initial_percentage_accept"]}
                      for k, v in result["groups"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
