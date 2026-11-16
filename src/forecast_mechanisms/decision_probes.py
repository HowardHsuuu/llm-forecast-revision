"""Held-out readouts at the first number and verdict generation steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error

from .core import read_jsonl, verify_complete_run
from .decision_trace import _selected
from .model import EXPECTED_BLOCKS, EXPECTED_HIDDEN
from .probes import _fit, _metrics, _save_direction


SPECS = (
    ("first_pre_number", 1, "p_first", None, "pre_number"),
    ("first_pre_verdict", 1, "v_first", None, "pre_verdict"),
    ("first_p_first_pre_verdict", 1, "p_first", None, "pre_verdict"),
    ("neutral_pre_number", 2, "p_first", "neutral", "pre_number"),
    ("neutral_pre_verdict", 2, "p_first", "neutral", "pre_verdict"),
    ("opposite_pre_number", 2, "p_first", "opposite", "pre_number"),
    ("opposite_pre_verdict", 2, "p_first", "opposite", "pre_verdict"),
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(run: Path, trace: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completion = verify_complete_run(run)
    source_manifest = json.loads((run / "manifest.json").read_text())
    trace_manifest = json.loads((trace / "manifest.json").read_text())
    if (trace_manifest["source_records_sha256"] != completion["records_sha256"] or
            trace_manifest["source_manifest"] != source_manifest):
        raise ValueError("decision traces do not match the completed collection")
    raw_source = read_jsonl(run / "records.jsonl")
    source = {r["key"]: r for r in raw_source}
    expected = {key for key, row in source.items() if _selected(row)}
    traces = read_jsonl(trace / "traces.jsonl")
    keys = [r["key"] for r in traces]
    if (len(source) != len(raw_source) or len(keys) != len(set(keys)) or
            set(keys) != expected):
        raise ValueError("decision traces are incomplete, duplicated or unexpected")
    first = {r["case_id"]: r for r in source.values()
             if r["round"] == 1 and r["arm"] == "p_first" and r["parse_ok"]}
    data = []
    for row in traces:
        original = source[row["key"]]
        for field in ("case_id", "domain", "split", "round", "arm", "condition", "p_accept", "verdict"):
            if row.get(field) != original.get(field):
                raise ValueError(f"trace metadata differs from its collection: {field}")
        path = (trace / row["trace_file"]).resolve()
        if not path.is_relative_to(trace.resolve()):
            raise ValueError("trace array must be inside the trace directory")
        with np.load(path, allow_pickle=False) as arrays:
            item = dict(row)
            for position in ("pre_number", "pre_verdict"):
                states = arrays[position]
                if (states.shape != (EXPECTED_BLOCKS + 1, EXPECTED_HIDDEN) or
                        not np.isfinite(states).all()):
                    raise ValueError("invalid decision-state dimensions or values")
                item[position] = states
        if row["round"] == 2:
            prior = first[row["case_id"]]
            item["prior_p"], item["prior_verdict"] = prior["p_accept"], prior["verdict"]
        data.append(item)
    return data, {"model": source_manifest["model"], "revision": source_manifest["revision"],
                  "prompt_voice": source_manifest.get("prompt_voice", "third_person"),
                  "source_records_sha256": completion["records_sha256"],
                  "trace_records_sha256": _digest(trace / "traces.jsonl")}


def _save_verdict_direction(path: Path, model: Any, train: list[dict], state_index: int) -> dict | None:
    if model is None:
        return None
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    direction = classifier.coef_[0] / scaler.scale_
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("invalid verdict-probe direction")
    unit = direction / length
    states = np.stack([r["_state"][state_index] for r in train]).astype(np.float64)
    spread = float(np.std(states @ unit, ddof=1))
    if not np.isfinite(spread) or spread <= 0:
        raise ValueError("invalid verdict-direction projection spread")
    np.savez(path, direction=unit.astype(np.float32), projection_sd=spread,
             state_index=state_index, train_mean=scaler.mean_.astype(np.float32))
    return {"file": path.name, "projection_sd": spread, "direction_norm": length}


def fit_decision_probes(run_dir: str | Path, trace_dir: str | Path,
                        output_dir: str | Path, *, block: int = 20,
                        all_layers: bool = False) -> dict[str, Any]:
    """Fit each condition on collection train cases and score its held-out cases.

    Block indices are zero based. ``block=20`` selects hidden-state index 21,
    the raw output of transformer block 20. The saved decision traces supply
    states before the first numeric or verdict token rather than prompt states.
    """
    if not 0 <= block < EXPECTED_BLOCKS:
        raise ValueError("block is outside the model")
    data, identity = _load(Path(run_dir), Path(trace_dir))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_index = block + 1
    if block == EXPECTED_BLOCKS - 1:
        raise ValueError("the final hidden-state index is normalized; select an earlier raw block output")
    manifest = {"protocol": "decision-token-probes-v1", **identity,
                "block": block, "all_layers": all_layers,
                "percentage_model": {"standardize": True, "ridge_alpha": 100.0},
                "verdict_model": {"standardize": True, "logistic_c": 0.01,
                                  "class_weight": "balanced", "max_iter": 2000}}
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("probe inputs or settings changed; use a new output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    fits = {}
    for name, round_no, arm, condition, position in SPECS:
        rows = [{**r, "_state": r[position]} for r in data
                if (r["round"], r["arm"], r.get("condition")) == (round_no, arm, condition)]
        train = [r for r in rows if r["split"] == "train"]
        test = [r for r in rows if r["split"] == "test"]
        if len(train) < 10 or len(test) < 3:
            fits[name] = {"status": "insufficient_parsed_cases", "n_train": len(train), "n_test": len(test)}
            continue
        percentage, verdict, mean, majority, n, counts = _fit(train, state_index)
        p_file = out / f"{name}_percentage_direction.npz"
        v_file = out / f"{name}_verdict_direction.npz"
        p_direction = _save_direction(p_file, percentage, train, state_index)
        v_direction = _save_verdict_direction(v_file, verdict, train, state_index)
        metrics = _metrics(test, state_index, percentage, verdict, mean, majority)
        prior = None
        if test and "prior_p" in test[0]:
            prior = {"p_mae": float(mean_absolute_error([r["p_accept"] for r in test],
                                                         [r["prior_p"] for r in test])),
                     "verdict_accuracy": float(np.mean([r["verdict"] == r["prior_verdict"] for r in test]))}
        curves = []
        for index in range(EXPECTED_BLOCKS + 1) if all_layers else (state_index,):
            p_model, v_model, train_mean, train_majority, *_ = _fit(train, index)
            curves.append({"state_index": index,
                           **_metrics(test, index, p_model, v_model, train_mean, train_majority)})
        fits[name] = {"status": "fit", "position": position, "n_train": n, "n_test": len(test),
                      "train_accept_counts": dict(counts), "at_selected_block": metrics,
                      "prior_answer_baseline": prior, "layer_curve": curves,
                      "percentage_direction": {"file": p_file.name, **p_direction},
                      "verdict_direction": v_direction,
                      "percentage_visible_in_prefix": position == "pre_verdict" and arm == "p_first"}
    result = {**manifest, "fits": fits}
    (out / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
