"""Grouped, held-out layerwise readouts of the model's own later reports."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import read_jsonl, verify_complete_run
from .model import NLA_LAYER


def _load(run_dir: Path) -> list[dict[str, Any]]:
    records = read_jsonl(run_dir / "records.jsonl")
    for row in records:
        row["_state"] = np.load(run_dir / row["activation_file"], allow_pickle=False)
        if row["_state"].ndim != 2 or not np.isfinite(row["_state"]).all():
            raise ValueError(f"bad activation file for {row['key']}")
    return records


def _rows(records: list[dict[str, Any]], *, split: str, round_no: int, arm: str, condition: str | None = None) -> list[dict[str, Any]]:
    return [
        r for r in records if r["split"] == split and r["round"] == round_no
        and r["arm"] == arm and r.get("condition") == condition and r["parse_ok"]
    ]


def _fit(train: list[dict[str, Any]], state_index: int):
    both = [r for r in train if r["p_accept"] is not None and r["verdict"] is not None]
    if len(both) < 10:
        raise ValueError("need at least 10 parsed training cases with both p and verdict")
    x = np.stack([r["_state"][state_index] for r in both]).astype(np.float64)
    p = np.asarray([r["p_accept"] for r in both], dtype=np.float64)
    y = np.asarray([r["verdict"] == "accept" for r in both], dtype=int)
    p_model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(x, p)
    v_model = None
    if len(np.unique(y)) == 2:
        v_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000)).fit(x, y)
    return p_model, v_model, float(p.mean()), int(y.mean() >= 0.5), len(both), Counter(y.tolist())


def _metrics(rows: list[dict[str, Any]], state_index: int, p_model: Any, v_model: Any, mean_p: float, majority: int) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows), "n_p": 0, "n_v": 0}
    if not rows:
        return result
    x_all = np.stack([r["_state"][state_index] for r in rows]).astype(np.float64)
    p_idx = [i for i, r in enumerate(rows) if r["p_accept"] is not None]
    v_idx = [i for i, r in enumerate(rows) if r["verdict"] is not None]
    result["n_p"] = len(p_idx)
    result["n_v"] = len(v_idx)
    if p_idx:
        p = np.asarray([rows[i]["p_accept"] for i in p_idx], dtype=float)
        predicted = np.clip(p_model.predict(x_all[p_idx]), 0, 100)
        result.update({
            "p_mae": float(mean_absolute_error(p, predicted)),
            "p_baseline_mae": float(mean_absolute_error(p, np.full_like(p, mean_p))),
            "p_r2": float(r2_score(p, predicted)) if len(p) > 1 else None,
            "p_mean": float(p.mean()),
        })
    if v_idx and v_model is not None:
        y = np.asarray([rows[i]["verdict"] == "accept" for i in v_idx], dtype=int)
        score = v_model.predict_proba(x_all[v_idx])[:, 1]
        prediction = (score >= 0.5).astype(int)
        result.update({
            "v_accuracy": float((prediction == y).mean()),
            "v_baseline_accuracy": float((majority == y).mean()),
            "v_accept_rate": float(y.mean()),
            "v_balanced_accuracy": float(np.mean([
                (prediction[y == c] == c).mean() for c in (0, 1)
            ])) if len(np.unique(y)) == 2 else None,
            "v_auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        })
    return result


def _save_direction(path: Path, p_model: Any, train: list[dict[str, Any]], state_index: int) -> dict[str, float]:
    scaler, ridge = p_model.steps[0][1], p_model.steps[1][1]
    direction = ridge.coef_ / scaler.scale_
    length = float(np.linalg.norm(direction))
    if length <= 0 or not np.isfinite(length):
        raise ValueError("zero or invalid probe direction")
    unit = direction / length
    x = np.stack([r["_state"][state_index] for r in train if r["p_accept"] is not None]).astype(np.float64)
    projected_sd = float(np.std(x @ unit, ddof=1))
    if projected_sd <= 0 or not np.isfinite(projected_sd):
        raise ValueError("zero projected activation variation")
    np.savez(path, direction=unit.astype(np.float32), projection_sd=projected_sd,
             state_index=state_index, train_mean=scaler.mean_.astype(np.float32))
    return {"projection_sd": projected_sd, "direction_norm": length}


def _paired_feedback_shift(
    records: list[dict[str, Any]], p_model: Any, state_index: int,
    source_arm: str, condition: str,
) -> dict[str, Any]:
    rows = {
        (r["case_id"], r.get("condition")): r
        for r in records if r["split"] == "test" and r["round"] == 2
        and r["arm"] == source_arm and r.get("condition") in ("neutral", condition)
        and r["parse_ok"] and r["p_accept"] is not None
    }
    observed, predicted = [], []
    for case_id in sorted({case_id for case_id, _ in rows}):
        neutral = rows.get((case_id, "neutral"))
        feedback = rows.get((case_id, condition))
        if neutral is None or feedback is None:
            continue
        states = np.stack((neutral["_state"][state_index], feedback["_state"][state_index])).astype(np.float64)
        scores = p_model.predict(states)
        observed.append(float(feedback["p_accept"] - neutral["p_accept"]))
        predicted.append(float(scores[1] - scores[0]))
    correlation = None
    if len(observed) > 2 and np.std(observed) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
    nonzero = [(actual, estimate) for actual, estimate in zip(observed, predicted)
               if actual != 0 and estimate != 0]
    return {
        "n_pairs": len(observed),
        "mean_reported_p_shift": float(np.mean(observed)) if observed else None,
        "mean_probe_p_shift": float(np.mean(predicted)) if predicted else None,
        "shift_correlation": correlation,
        "n_nonzero_shift_pairs": len(nonzero),
        "sign_agreement": float(np.mean([np.sign(actual) == np.sign(estimate)
                                         for actual, estimate in nonzero])) if nonzero else None,
    }


def fit_probes(run_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    completion = verify_complete_run(run)
    records = _load(run)
    if not records:
        raise ValueError("no records to probe")
    n_states, hidden = records[0]["_state"].shape
    if any(r["_state"].shape != (n_states, hidden) for r in records):
        raise ValueError("inconsistent activation dimensions")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    results: dict[str, Any] = {"model": manifest["model"], "revision": manifest["revision"], "records_sha256": completion["records_sha256"], "state_count": n_states, "hidden_size": hidden, "fits": {}}
    fit_specs = (
        ("first", 1, "p_first", None),
        ("second_neutral", 2, "p_first", "neutral"),
    )
    for fit_name, round_no, arm, condition in fit_specs:
        train = _rows(records, split="train", round_no=round_no, arm=arm, condition=condition)
        test = _rows(records, split="test", round_no=round_no, arm=arm, condition=condition)
        if len(train) < 10 or len(test) < 3:
            results["fits"][fit_name] = {"status": "insufficient_parsed_cases", "n_train": len(train), "n_test": len(test)}
            continue
        curves = []
        for state_index in range(n_states):
            p_model, v_model, mean_p, majority, n_train, class_counts = _fit(train, state_index)
            curves.append({"state_index": state_index, **_metrics(test, state_index, p_model, v_model, mean_p, majority)})
            if state_index == NLA_LAYER + 1:
                direction_info = _save_direction(out / f"{fit_name}_block20_direction.npz", p_model, train, state_index)
                transfers = {}
                if fit_name == "first":
                    transfer_specs = (
                        ("first_v_first", 1, "v_first", None),
                        ("first_p_only", 1, "p_only", None),
                        ("first_v_only", 1, "v_only", None),
                        ("second_p_first_neutral", 2, "p_first", "neutral"),
                        ("second_v_only_neutral", 2, "v_only", "neutral"),
                    )
                else:
                    transfer_specs = tuple(
                        (f"second_{source}_{feedback}", 2, source, feedback)
                        for source in ("p_first", "v_only") for feedback in ("neutral", "opposite", "aligned")
                    )
                for name, t_round, t_arm, t_condition in transfer_specs:
                    target = _rows(records, split="test", round_no=t_round, arm=t_arm, condition=t_condition)
                    transfers[name] = _metrics(target, state_index, p_model, v_model, mean_p, majority)
                domains = sorted({r["domain"] for r in train + test})
                heldout_by_domain = {
                    domain: _metrics([r for r in test if r["domain"] == domain],
                                     state_index, p_model, v_model, mean_p, majority)
                    for domain in domains
                }
                cross_domain = {}
                for source_domain in domains:
                    source_train = [r for r in train if r["domain"] == source_domain]
                    if len(source_train) < 10:
                        continue
                    domain_p, domain_v, domain_mean, domain_majority, domain_n, _ = _fit(source_train, state_index)
                    for target_domain in domains:
                        if target_domain == source_domain:
                            continue
                        target_rows = [r for r in test if r["domain"] == target_domain]
                        cross_domain[f"{source_domain}:{target_domain}"] = {
                            "n_train": domain_n,
                            **_metrics(target_rows, state_index, domain_p, domain_v,
                                       domain_mean, domain_majority),
                        }
                feedback_shifts = {}
                if fit_name == "second_neutral":
                    for source_arm in ("p_first", "v_only"):
                        for feedback in ("opposite", "aligned"):
                            feedback_shifts[f"{source_arm}:{feedback}"] = _paired_feedback_shift(
                                records, p_model, state_index, source_arm, feedback
                            )
                results["fits"][fit_name] = {
                    "status": "fit", "n_train": n_train, "n_test": len(test),
                    "train_accept_counts": dict(class_counts), "direction": direction_info,
                    "transfers_at_block20": transfers,
                    "heldout_by_domain_at_block20": heldout_by_domain,
                    "cross_domain_at_block20": cross_domain,
                    "feedback_shifts_at_block20": feedback_shifts,
                }
        results["fits"][fit_name]["layer_curve"] = curves
    (out / "probe_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results
