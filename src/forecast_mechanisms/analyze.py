"""Behavior, response revision, and intervention summaries with denominators."""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .core import read_jsonl, verify_complete_run


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _bootstrap_mean_ci(values: list[float], seed: int = 42, n_boot: int = 1000) -> list[float] | None:
    if len(values) < 10:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return [float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot) - 1])]


def behavior_summary(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    verify_complete_run(run)
    records = read_jsonl(run / "records.jsonl")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        name = f"round{row['round']}:{row['arm']}:{row.get('condition') or 'none'}"
        groups[name].append(row)
    by_group = {}
    for name, rows in sorted(groups.items()):
        parsed = [r for r in rows if r["parse_ok"]]
        format_applicable = rows[0]["arm"] != "natural"
        p = [r["p_accept"] for r in parsed if r["p_accept"] is not None]
        verdicts = [r["verdict"] for r in parsed if r["verdict"] is not None]
        both = [r for r in parsed if r["p_accept"] is not None and r["verdict"] is not None]
        by_group[name] = {
            "total": len(rows), "format_applicable": format_applicable,
            "parsed": len(parsed) if format_applicable else None,
            "parse_rate": len(parsed) / len(rows) if format_applicable else None,
            "n_probability": len(p), "mean_probability": _mean(p),
            "n_verdict": len(verdicts),
            "accept_verdict_rate": verdicts.count("accept") / len(verdicts) if verdicts else None,
            "p_verdict_disagreements": sum((r["p_accept"] >= 50) != (r["verdict"] == "accept") for r in both),
            "n_p_and_verdict": len(both),
        }
    indexed = {(r["case_id"], r["round"], r["arm"], r.get("condition")): r for r in records}
    paired = {}
    for arm in ("p_first", "v_only"):
        for condition in ("neutral", "opposite", "aligned"):
            p_change, switches = [], []
            for case_id, round_no, source_arm, _ in indexed:
                if round_no != 1 or source_arm != arm:
                    continue
                first = indexed[(case_id, 1, arm, None)]
                second = indexed.get((case_id, 2, arm, condition))
                if not second or not first["parse_ok"] or not second["parse_ok"]:
                    continue
                if first["p_accept"] is not None:
                    p_change.append(second["p_accept"] - first["p_accept"])
                switches.append(first["verdict"] != second["verdict"])
            paired[f"{arm}:{condition}"] = {
                "n_paired_verdicts": len(switches), "verdict_switch_rate": _mean(switches),
                "n_paired_probabilities": len(p_change), "mean_p_change": _mean(p_change),
                "mean_p_change_ci95": _bootstrap_mean_ci(p_change),
            }
    feedback_effect = {}
    for arm in ("p_first", "v_only"):
        for condition in ("opposite", "aligned"):
            differences = []
            for case_id, round_no, source_arm, _ in indexed:
                if round_no != 1 or source_arm != arm:
                    continue
                neutral = indexed.get((case_id, 2, arm, "neutral"))
                feedback = indexed.get((case_id, 2, arm, condition))
                if neutral and feedback and neutral["parse_ok"] and feedback["parse_ok"]:
                    differences.append(feedback["p_accept"] - neutral["p_accept"])
            feedback_effect[f"{arm}:{condition}"] = {
                "n_pairs": len(differences),
                "mean_p_difference_vs_neutral": _mean(differences),
                "mean_p_difference_ci95": _bootstrap_mean_ci(differences),
            }
    # Outcome score is withheld unless each included row has independently
    # verified predecision provenance; legacy rows cannot establish calibration.
    verified = [r for r in records if r["round"] == 1 and r["arm"] == "p_first" and r["parse_ok"] and r["audit_status"] == "verified_predecision" and r["outcome"] in ("accept", "reject")]
    outcome = {"n_verified": len(verified), "brier": None, "base_rate_brier": None}
    if verified:
        ys = [float(r["outcome"] == "accept") for r in verified]
        ps = [r["p_accept"] / 100 for r in verified]
        base_rate = statistics.mean(ys)
        outcome.update({
            "brier": _mean([(p - y) ** 2 for p, y in zip(ps, ys)]),
            "base_rate_brier": _mean([(base_rate - y) ** 2 for y in ys]),
            "accept_base_rate": base_rate,
            "brier_ci95": _bootstrap_mean_ci([(p - y) ** 2 for p, y in zip(ps, ys)]),
        })
    return {"n_records": len(records), "groups": by_group, "paired_revisions": paired, "feedback_vs_neutral": feedback_effect, "outcome_calibration": outcome}


def intervention_summary(intervention_dir: str | Path) -> dict[str, Any]:
    records = read_jsonl(Path(intervention_dir) / "interventions.jsonl")
    baselines = {r["source_key"]: r for r in records if r["kind"] == "baseline" and r["replay_matches_collection"]}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["kind"] == "baseline":
            continue
        name = f"{row['kind']}:{row['dose']:+g}:{row['round']}:{row['arm']}:{row.get('condition') or 'none'}"
        groups[name].append(row)
    summary = {}
    for name, rows in sorted(groups.items()):
        generated = [r for r in rows if r.get("status") == "generated"]
        parsed = [r for r in generated if r["parse_ok"]]
        p_delta, verdict_flip = [], []
        for row in parsed:
            baseline = baselines.get(row["source_key"])
            if not baseline or not baseline["parse_ok"]:
                continue
            if row["p_accept"] is not None and baseline["p_accept"] is not None:
                p_delta.append(row["p_accept"] - baseline["p_accept"])
            if row["verdict"] is not None and baseline["verdict"] is not None:
                verdict_flip.append(row["verdict"] != baseline["verdict"])
        summary[name] = {
            "scheduled": len(rows), "generated": len(generated), "parsed": len(parsed),
            "n_paired_p": len(p_delta), "mean_p_delta": _mean(p_delta),
            "mean_p_delta_ci95": _bootstrap_mean_ci(p_delta),
            "n_paired_verdict": len(verdict_flip), "verdict_flip_rate": _mean(verdict_flip),
        }
    return {"n_baseline_replay_ok": len(baselines), "n_baseline_replay_failed": sum(r["kind"] == "baseline" and not r["replay_matches_collection"] for r in records), "groups": summary}


def write_analysis(run_dir: str | Path, output_dir: str | Path, intervention_dir: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"behavior": behavior_summary(run_dir)}
    if intervention_dir is not None:
        result["intervention"] = intervention_summary(intervention_dir)
    (out / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
