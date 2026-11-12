"""Aggregate the frozen route phases and check the reported causal contrasts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .core import read_jsonl
from .reproduce_route import REFERENCE_PHASES_SHA256

PHASE_SIZES = {"pilot": 12, "replication": 12, "extension": 39}
MANIFEST_KEYS = (
    "protocol", "run_records_sha256", "trace_records_sha256", "phase_file_sha256",
    "route_code_sha256", "attention_code_sha256", "model", "revision",
    "decoder_repetition_penalty",
)


def _estimate(values: list[float], seed: int = 42) -> dict:
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(10_000, len(a)))
    return {
        "mean": round(float(a.mean()), 3),
        "ci95": [round(float(x), 3) for x in np.quantile(a[indices].mean(axis=1), [0.025, 0.975])],
    }


def _count(rows: list[dict], predicate) -> int:
    return sum(bool(predicate(row)) for row in rows)


def summarize_route(route_root: str | Path, output_file: str | Path) -> dict:
    root, output = Path(route_root), Path(output_file)
    rows = []
    by_phase = {}
    fingerprint = None
    for phase, expected in PHASE_SIZES.items():
        path = root / phase / "records.jsonl"
        manifest = json.loads((root / phase / "manifest.json").read_text())
        if manifest["phase"] != phase or manifest["only_case"] is not None:
            raise ValueError(f"incomplete or mismatched route phase: {phase}")
        current_fingerprint = tuple(manifest[key] for key in MANIFEST_KEYS)
        if manifest["phase_file_sha256"] != REFERENCE_PHASES_SHA256:
            raise ValueError(f"phase {phase} uses a different case selection")
        if fingerprint is not None and current_fingerprint != fingerprint:
            raise ValueError(f"phase {phase} uses different inputs or intervention code")
        fingerprint = current_fingerprint
        phase_rows = read_jsonl(path)
        if len(phase_rows) != expected:
            raise ValueError(f"phase {phase} has {len(phase_rows)} instead of {expected} rows")
        if any(row["phase"] != phase for row in phase_rows):
            raise ValueError(f"route rows have the wrong phase: {phase}")
        valid = [row for row in phase_rows if row["status"] == "valid"]
        by_phase[phase] = {"rows": expected, "valid": len(valid),
                           "status_counts": dict(Counter(row["status"] for row in phase_rows)),
                           "state_target_tokens": _count(valid, lambda r: r["state_forward"]["target_match"]),
                           "head20_target_tokens": _count(valid, lambda r: r["head20"]["target_match"]),
                           "reverse_neutral_tokens": _count(valid, lambda r: r["prefix_positions"]["):"]["reverse"]["source_match"])}
        rows.extend(phase_rows)
    if len({row["case_id"] for row in rows}) != 63:
        raise ValueError("phase selections overlap or omit cases")
    valid = [row for row in rows if row["status"] == "valid"]
    if not valid:
        raise ValueError("no informative cases")
    if any(any(row["qkv_max_errors"].values()) or row["later_attention"]["27"]["max_raw_logit_error"] != 0
           for row in valid):
        raise ValueError("projection or final-logit validation failed")

    result = {"cases": len(rows), "informative": len(valid), "by_phase": by_phase}
    result["state"] = {
        "forward_target_percentages": _count(valid, lambda r: r["state_forward"]["p_accept"] == r["pushback_p"]),
        "reverse_neutral_percentages": _count(valid, lambda r: r["state_reverse"]["p_accept"] == r["neutral_p"]),
        "forward_orthogonal_target_percentages": _count(valid, lambda r: r["state_orthogonal"]["forward"]["p_accept"] == r["pushback_p"]),
        "reverse_orthogonal_neutral_percentages": _count(valid, lambda r: r["state_orthogonal"]["reverse"]["p_accept"] == r["neutral_p"]),
    }
    result["heads"] = {
        "state_target_tokens": _count(valid, lambda r: r["state_forward"]["target_match"]),
        "head20_target_tokens": _count(valid, lambda r: r["head20"]["target_match"]),
        "all_heads_target_tokens": _count(valid, lambda r: r["all_heads"]["target_match"]),
        "other_single_head_incremental_tokens": {
            str(head): _count(valid, lambda r, head=head: r["single_heads"][head]["target_match"] and not r["state_forward"]["target_match"])
            for head in range(28) if head != 20
        },
        "head20_omission_losses": _count(valid, lambda r: r["all_heads"]["target_match"] and not r["without_head20"]["target_match"]),
    }
    result["cache_segments"] = {
        name: {"target_tokens": _count(valid, lambda r, name=name: r["cache_segments"][name]["target_match"]),
               "margin_gain": _estimate([r["cache_segments"][name]["processed_margin"] - r["state_forward"]["processed_margin"] for r in valid])}
        for name in ("all_past", "user_turn", "generated_prefix")
    }
    result["prefix_positions"] = {
        name: {
            "forward_target_tokens": _count(valid, lambda r, name=name: r["prefix_positions"][name]["forward"]["target_match"]),
            "forward_margin_gain": _estimate([r["prefix_positions"][name]["forward"]["processed_margin"] - r["state_forward"]["processed_margin"] for r in valid]),
            "reverse_neutral_tokens": _count(valid, lambda r, name=name: r["prefix_positions"][name]["reverse"]["source_match"]),
            "reverse_margin_change": _estimate([r["prefix_positions"][name]["reverse"]["processed_margin"] - r["natural_target"]["processed_margin"] for r in valid]),
        }
        for name in ("P", "(", "accept", "):")
    }
    result["specificity"] = {
        "donor_cosine": _estimate([r["donor_cosine"] for r in valid]),
        "orthogonal_cosine": _estimate([r["orthogonal_cosine"] for r in valid]),
        "forward": {}, "reverse": {},
    }
    for direction, baseline in (("forward", "state_forward"), ("reverse", "natural_target")):
        for name in ("same_case", "donor", "orthogonal"):
            result["specificity"][direction][name] = {
                "margin_change": _estimate([
                    r["specificity"][name][direction]["processed_margin"] - r[baseline]["processed_margin"]
                    for r in valid]),
                "matching_tokens": _count(valid, lambda r, name=name, direction=direction:
                                          r["specificity"][name][direction]["target_match" if direction == "forward" else "source_match"]),
            }
        result["specificity"][direction]["donor_minus_same_margin"] = _estimate([
            r["specificity"]["donor"][direction]["processed_margin"]
            - r["specificity"]["same_case"][direction]["processed_margin"]
            for r in valid])
    result["later_attention"] = {
        str(block): _count(valid, lambda r, block=block: r["later_attention"][str(block)]["target_match"])
        for block in range(21, 28)
    }
    result["domains"] = {
        domain: {
            "n": len(group),
            "reverse_same_case_margin_change": _estimate([
                r["specificity"]["same_case"]["reverse"]["processed_margin"]
                - r["natural_target"]["processed_margin"] for r in group]),
            "reverse_neutral_tokens": _count(group, lambda r: r["specificity"]["same_case"]["reverse"]["source_match"]),
        }
        for domain in sorted({r["domain"] for r in valid})
        for group in [[r for r in valid if r["domain"] == domain]]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
