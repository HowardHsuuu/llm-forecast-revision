"""Summarize the frozen third-cohort wording and prefix-cache experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


CONDITIONS = ("opposite", "paraphrase_a", "paraphrase_b", "unrelated_expectation")
DOMAINS = ("grad_admissions", "paper_acceptance")
EDIT_PAIRS = {
    "head20": ("head20_paired", "head20_orthogonal"),
    "all_groups": ("all_groups_paired", "all_groups_orthogonal"),
}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(directory: Path) -> list[dict]:
    marker = json.loads((directory / "completion.json").read_text())
    records = directory / "records.jsonl"
    rows = read(records)
    if marker["records_sha256"] != digest(records) or marker["total_records"] != len(rows):
        raise ValueError(f"incomplete or changed run: {directory}")
    return rows


def interval(values: list[float], seed: int) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "ci95": None,
                "negative": 0, "positive": 0, "zero": 0}
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(10000, dtype=np.float64)
    for index in range(len(estimates)):
        estimates[index] = arr[rng.integers(0, len(arr), len(arr))].mean()
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci95": [float(x) for x in np.quantile(estimates, [0.025, 0.975])],
        "negative": int(np.count_nonzero(arr < 0)),
        "positive": int(np.count_nonzero(arr > 0)),
        "zero": int(np.count_nonzero(arr == 0)),
    }


def route_summary(rows: list[dict], condition: str, domain: str | None, seed: int,
                  allowed_ids: set[str] | None = None) -> dict:
    subset = [row for row in rows if (domain is None or row["domain"] == domain)
              and (allowed_ids is None or row["case_id"] in allowed_ids)]
    statuses = Counter(row["conditions"][condition]["status"]
                       if row["status"] == "valid" else row["status"]
                       for row in subset)
    eligible = [row for row in subset if row["status"] == "valid"
                and row["conditions"][condition]["status"] == "valid"]
    output = {
        "n_cases": len(subset), "statuses": dict(statuses), "eligible": len(eligible),
        "eligible_initial_verdicts": dict(Counter(row["initial_verdict"] for row in eligible)),
        "naturally_changed_percentage": sum(
            row["conditions"][condition]["natural_percentage"] != row["neutral_percentage"]
            for row in eligible),
        "naturally_changed_verdict": sum(
            row["conditions"][condition]["natural_verdict"] != row["neutral_verdict"]
            for row in eligible),
        "natural_margin": interval([row["conditions"][condition]["natural_processed_margin"]
                                    for row in eligible], seed + 1),
        "edits": {},
    }
    for offset, (scope, (paired, control)) in enumerate(EDIT_PAIRS.items()):
        metrics = {}
        for field in ("processed_margin_shift", "raw_margin_shift"):
            paired_values = [row["conditions"][condition]["interventions"][paired][field]
                             for row in eligible]
            control_values = [row["conditions"][condition]["interventions"][control][field]
                              for row in eligible]
            metrics[field] = {
                "paired": interval(paired_values, seed + offset * 10 + 2),
                "control": interval(control_values, seed + offset * 10 + 3),
                "paired_minus_control": interval(
                    [a - b for a, b in zip(paired_values, control_values)],
                    seed + offset * 10 + 4),
            }
        outcomes = {}
        for name in (paired, control):
            cases = [row["conditions"][condition]["interventions"][name]
                     for row in eligible]
            outcomes[name] = {
                "first_token_restored": sum(r["neutral_first_token_match"] for r in cases),
                "full_parse_ok": sum(r["parse_ok"] for r in cases),
                "full_percentage_restored": sum(
                    r["parse_ok"] and r["full_percentage"] == row["neutral_percentage"]
                    for r, row in zip(cases, eligible)),
                "verdict_matches_neutral": sum(
                    r["parse_ok"] and r["full_verdict"] == row["neutral_verdict"]
                    for r, row in zip(cases, eligible)),
                "verdict_restored_among_changed": sum(
                    r["parse_ok"] and r["full_verdict"] == row["neutral_verdict"]
                    and row["conditions"][condition]["natural_verdict"] != row["neutral_verdict"]
                    for r, row in zip(cases, eligible)),
            }
        outcomes["paired_minus_control_ci95"] = {}
        for field in ("first_token_restored", "full_percentage_restored",
                      "verdict_restored_among_changed"):
            def restored(row: dict, intervention_name: str) -> bool:
                result = row["conditions"][condition]["interventions"][intervention_name]
                if field == "first_token_restored":
                    return result["neutral_first_token_match"]
                if field == "full_percentage_restored":
                    return result["parse_ok"] and result["full_percentage"] == row["neutral_percentage"]
                return (result["parse_ok"] and result["full_verdict"] == row["neutral_verdict"]
                        and row["conditions"][condition]["natural_verdict"] != row["neutral_verdict"])
            metric_rows = eligible if field != "verdict_restored_among_changed" else [
                row for row in eligible
                if row["conditions"][condition]["natural_verdict"] != row["neutral_verdict"]]
            outcomes["paired_minus_control_ci95"][field] = interval(
                [int(restored(row, paired)) - int(restored(row, control)) for row in metric_rows],
                seed + offset * 10 + 8,
            )
        metrics["outcomes"] = outcomes
        output["edits"][scope] = metrics
    return output


def main(root: Path) -> None:
    baseline = complete(root / "baseline")
    variants = complete(root / "variants")
    route = complete(root / "route")
    if len(route) != 160:
        raise ValueError("unexpected total run counts")
    first = {r["case_id"]: r for r in baseline if r["round"] == 1}
    second = {(r["case_id"], r["condition"]): r for r in baseline if r["round"] == 2}
    second.update({(r["case_id"], r["condition"]): r for r in variants})
    valid_first = sum(row["parse_ok"] for row in first.values())
    if (len(first) != 160 or len(baseline) != 160 + 2 * valid_first
            or len(variants) != 3 * valid_first or len(second) != 5 * valid_first):
        raise ValueError("conversation matrix incomplete")
    common_ids = {row["case_id"] for row in route
                  if row["status"] == "valid" and all(
                      row["conditions"][condition]["status"] == "valid"
                      for condition in CONDITIONS)}
    summary: dict = {"protocol": "cross-wording-prefix-route-v2", "n": 160,
                     "common_eligible": len(common_ids), "behavior": {},
                     "route": {}, "route_common": {}}
    for ci, condition in enumerate(CONDITIONS):
        summary["behavior"][condition] = {}
        summary["route"][condition] = {}
        summary["route_common"][condition] = {}
        for di, domain in enumerate((None, *DOMAINS)):
            label = domain or "all"
            cases = [row for row in route if domain is None or row["domain"] == domain]
            observed = [row for row in cases
                        if (row["case_id"], condition) in second
                        and (row["case_id"], "neutral") in second]
            parsed = [row for row in observed
                      if second[row["case_id"], condition]["parse_ok"]
                      and second[row["case_id"], "neutral"]["parse_ok"]]
            changed_percentage = sum(
                second[row["case_id"], condition]["p_accept"]
                != second[row["case_id"], "neutral"]["p_accept"] for row in parsed)
            changed_verdict = sum(
                second[row["case_id"], condition]["verdict"]
                != second[row["case_id"], "neutral"]["verdict"] for row in parsed)
            summary["behavior"][condition][label] = {
                "n_selected": len(cases), "n_observed": len(observed),
                "n_parsed_pairs": len(parsed),
                "percentage_changed": changed_percentage,
                "verdict_changed": changed_verdict,
                "parse_ok": sum(second[row["case_id"], condition]["parse_ok"] for row in observed),
            }
            summary["route"][condition][label] = route_summary(
                route, condition, domain, 20261115 + ci * 100 + di * 1000)
            summary["route_common"][condition][label] = route_summary(
                route, condition, domain, 20261116 + ci * 100 + di * 1000,
                common_ids)
    output = root / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for condition in CONDITIONS:
        r = summary["route"][condition]["all"]
        head = r["edits"]["head20"]
        all_groups = r["edits"]["all_groups"]
        print(condition, "n", r["eligible"], "statuses", r["statuses"],
              "head-net", head["processed_margin_shift"]["paired_minus_control"]["mean"],
              "head-P", head["outcomes"]["head20_paired"]["full_percentage_restored"],
              "all-net", all_groups["processed_margin_shift"]["paired_minus_control"]["mean"],
              "all-P", all_groups["outcomes"]["all_groups_paired"]["full_percentage_restored"])
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
