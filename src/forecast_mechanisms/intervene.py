"""Held-out, position-specific tests of a percentage-probe direction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import append_jsonl, parse_response, read_jsonl, verify_complete_run
from .model import EXPECTED_HIDDEN, NLA_LAYER, ForecastRunner


def _orthogonal_controls(direction: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    controls = []
    for _ in range(count):
        vector = rng.normal(size=direction.shape)
        vector -= np.dot(vector, direction) * direction
        for prior in controls:
            vector -= np.dot(vector, prior) * prior
        norm = np.linalg.norm(vector)
        if norm <= 1e-10:
            raise ValueError("failed to generate orthogonal control")
        controls.append((vector / norm).astype(np.float32))
    return controls


def intervene(
    run_dir: str | Path, probe_dir: str | Path, output_dir: str | Path,
    runner: ForecastRunner, *, doses: tuple[float, ...] = (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0),
    n_controls: int = 3, seed: int = 42, max_fraction_of_state_norm: float = 0.15,
) -> dict[str, Any]:
    run, probe, out = Path(run_dir), Path(probe_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    completion = verify_complete_run(run)
    probe_results = json.loads((probe / "probe_results.json").read_text(encoding="utf-8"))
    if probe_results.get("records_sha256") != completion["records_sha256"]:
        raise ValueError("probe fit does not match the completed collection")
    run_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if runner.model_name != run_manifest["model"] or runner.revision != run_manifest["revision"] or runner.max_new_tokens != run_manifest["max_new_tokens"]:
        raise ValueError("intervention runner differs from collection model/revision/generation settings")
    direction_path = probe / "first_block20_direction.npz"
    with np.load(direction_path, allow_pickle=False) as saved:
        direction = saved["direction"].astype(np.float32)
        projection_sd = float(saved["projection_sd"])
        state_index = int(saved["state_index"])
    if direction.shape != (EXPECTED_HIDDEN,) or state_index != NLA_LAYER + 1 or not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-4):
        raise ValueError("probe direction does not match Qwen block-20 output")
    if n_controls < 1 or any(d == 0 for d in doses) or not doses:
        raise ValueError("need nonzero doses and at least one orthogonal control")
    probe_sha = hashlib.sha256(direction_path.read_bytes()).hexdigest()
    settings = {
        "run_manifest": run_manifest, "direction_sha256": probe_sha,
        "doses": list(doses), "n_controls": n_controls, "seed": seed,
        "max_fraction_of_state_norm": max_fraction_of_state_norm,
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("intervention settings changed; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records_path = out / "interventions.jsonl"
    existing_rows = read_jsonl(records_path) if records_path.exists() else []
    existing = {r["intervention_key"] for r in existing_rows}
    saved_baselines = {r["intervention_key"]: r for r in existing_rows if r["kind"] == "baseline"}
    source = [r for r in read_jsonl(run / "records.jsonl") if r["split"] == "test" and r["parse_ok"] and (
        (r["round"] == 1 and r["arm"] in ("p_first", "v_first", "p_only", "v_only")) or
        (r["round"] == 2 and r["arm"] in ("p_first", "v_only") and r["condition"] in ("neutral", "opposite"))
    )]
    controls = _orthogonal_controls(direction, n_controls, seed)
    new = replay_mismatch = out_of_range = 0
    for row in source:
        baseline_key = row["key"] + "|baseline"
        if baseline_key in saved_baselines:
            baseline_ok = saved_baselines[baseline_key]["replay_matches_collection"]
        else:
            replay = runner.generate(row["messages"])
            baseline_ok = replay == row["response"]
            baseline_row = {
                "intervention_key": baseline_key, "source_key": row["key"], "case_id": row["case_id"],
                "domain": row["domain"],
                "round": row["round"], "arm": row["arm"], "condition": row.get("condition"),
                "kind": "baseline", "dose": 0.0, "response": replay,
                "replay_matches_collection": baseline_ok,
                **parse_response(replay, "p_first" if row["round"] == 2 else row["arm"]),
            }
            append_jsonl(records_path, baseline_row)
            saved_baselines[baseline_key] = baseline_row
            existing.add(baseline_key)
            new += 1
        if not baseline_ok:
            replay_mismatch += 1
            continue
        state = np.load(run / row["activation_file"], allow_pickle=False)[state_index].astype(np.float32)
        state_norm = float(np.linalg.norm(state))
        if state_norm <= 0 or not np.isfinite(state_norm):
            raise ValueError(f"invalid source state norm for {row['key']}")
        vectors = [("probe", direction)] + [(f"orthogonal_{i}", v) for i, v in enumerate(controls)]
        for kind, unit in vectors:
            for dose in doses if kind == "probe" else (min(doses), max(doses)):
                key = f"{row['key']}|{kind}|{dose:+g}"
                if key in existing:
                    continue
                delta = unit * (dose * projection_sd)
                relative_norm = float(np.linalg.norm(delta) / state_norm)
                if relative_norm > max_fraction_of_state_norm:
                    out_of_range += 1
                    append_jsonl(records_path, {
                        "intervention_key": key, "source_key": row["key"], "case_id": row["case_id"],
                        "domain": row["domain"],
                        "round": row["round"], "arm": row["arm"], "condition": row.get("condition"),
                        "kind": kind, "dose": dose, "relative_norm": relative_norm,
                        "status": "skipped_large_intervention", "response": None,
                        "parse_ok": False, "p_accept": None, "verdict": None,
                    })
                    existing.add(key)
                    new += 1
                    continue
                response = runner.generate(row["messages"], intervention=(NLA_LAYER, delta))
                append_jsonl(records_path, {
                    "intervention_key": key, "source_key": row["key"], "case_id": row["case_id"],
                    "domain": row["domain"],
                    "round": row["round"], "arm": row["arm"], "condition": row.get("condition"),
                    "kind": kind, "dose": dose, "relative_norm": relative_norm,
                    "status": "generated", "response": response,
                    **parse_response(response, "p_first" if row["round"] == 2 else row["arm"]),
                })
                existing.add(key)
                new += 1
    return {"eligible_source_rows": len(source), "new_records": new, "replay_mismatch_rows": replay_mismatch, "skipped_large_interventions": out_of_range}
