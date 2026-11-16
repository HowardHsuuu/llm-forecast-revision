"""Recompute held-out results and refit a stronger first-answer-only baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold

from forecast_mechanisms.core import read_jsonl
from common import SEED, TRAIN_SHA, TEST_SHA, bootstrap, load_pair, metrics, save, sha


def ordered(path, cases):
    rows = read_jsonl(path)
    index = {r["case_id"]: r for r in rows}
    if len(index) != len(rows) or set(index) != {r["case_id"] for r in cases}:
        raise ValueError("prediction case set differs from its frozen cohort")
    return [index[r["case_id"]] for r in cases]


def forest(tr, te):
    def features(rows):
        return np.asarray([[r["first_p"], int(r["first_verdict"] == "accept"),
                            int(r["domain"] == "paper_acceptance")] for r in rows], dtype=float)
    x, xt = features(tr), features(te)
    y = np.asarray([r["opposite_p"] for r in tr])
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(x))
    scored = []
    for depth in (3, 5, None):
        for leaf in (3, 6, 10):
            predicted = np.empty(len(tr))
            for fit_ids, val_ids in folds:
                model = RandomForestRegressor(n_estimators=200, max_depth=depth,
                    min_samples_leaf=leaf, random_state=SEED, n_jobs=1).fit(x[fit_ids], y[fit_ids])
                predicted[val_ids] = model.predict(x[val_ids])
            scored.append((float(mean_absolute_error(y, predicted)), depth, leaf))
    scored.sort(key=lambda row: (row[0], str(row[1]), row[2]))
    cv_mae, depth, leaf = scored[0]
    model = RandomForestRegressor(n_estimators=200, max_depth=depth,
        min_samples_leaf=leaf, random_state=SEED, n_jobs=1).fit(x, y)
    return model.predict(xt), {"cv_mae": cv_mae, "max_depth": depth, "min_samples_leaf": leaf}


def summarize(train, test, percentage, verdict, first_control, output):
    tr, te = load_pair(train, test, activations=False)
    numeric = ordered(percentage, te)
    y = np.asarray([r["opposite_p"] for r in te])
    for r, original in zip(numeric, te):
        if r["opposite_p"] != original["opposite_p"]:
            raise ValueError("percentage prediction target differs from its collection")
    probe = np.asarray([r["probe_p"] for r in numeric])
    baseline = np.asarray([r["baseline_p"] for r in numeric])
    rf, rf_selection = forest(tr, te)
    numeric_summary = {"n_train": len(tr), "n_test": len(te), "excluded_test": 2,
        "probe_mae": float(mean_absolute_error(y, probe)),
        "field_baseline_mae": float(mean_absolute_error(y, baseline)),
        "copy_first_mae": float(mean_absolute_error(y, [r["first_p"] for r in te])),
        "field_baseline_mae_gain_ci95": bootstrap(np.abs(y - baseline) - np.abs(y - probe)),
        "posthoc_random_forest": {"selection": rf_selection,
            "mae": float(mean_absolute_error(y, rf)),
            "probe_gain_ci95": bootstrap(np.abs(y - rf) - np.abs(y - probe))}}
    accepted = [r for r in te if r["first_verdict"] == "accept"]
    vr = ordered(verdict, accepted)
    target = np.asarray([r["opposite_verdict"] == "accept" for r in accepted], dtype=int)
    if any(bool(r["stable"]) != bool(target[i]) for i, r in enumerate(vr)):
        raise ValueError("verdict prediction target differs from its collection")
    vp = np.asarray([r["state_prob"] for r in vr])
    vb = np.asarray([r["baseline_prob"] for r in vr])
    first = ordered(first_control, accepted) if first_control else vr
    fp = np.asarray([r["first_turn_prob"] for r in first])
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(5000):
        sample = rng.integers(0, len(target), size=len(target))
        if len(np.unique(target[sample])) == 2:
            differences.append(roc_auc_score(target[sample], vp[sample]) -
                               roc_auc_score(target[sample], fp[sample]))
    summary = {"protocol": "pre-answer-readouts-summary-v1", "train_records_sha256": TRAIN_SHA,
        "test_records_sha256": TEST_SHA, "analysis_script_sha256": sha(__file__),
        "common_sha256": sha(Path(__file__).with_name("common.py")),
        "percentage_predictions_sha256": sha(percentage), "verdict_predictions_sha256": sha(verdict),
        "first_turn_predictions_sha256": sha(first_control) if first_control else sha(verdict),
        "percentage": numeric_summary, "verdict": {"n_train": 147, "n_test": 137,
            "field_baseline": metrics(target, vb), "second_turn_pre_answer": metrics(target, vp),
            "posthoc_first_turn_control": metrics(target, fp),
            "second_minus_first_auroc": float(roc_auc_score(target, vp) - roc_auc_score(target, fp)),
            "second_minus_first_auroc_ci95": np.quantile(differences, (.025, .975)).tolist()}}
    save(output, summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("train", "test", "percentage-predictions", "verdict-predictions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--first-turn-control", type=Path)
    args = parser.parse_args()
    summarize(args.train, args.test, args.percentage_predictions, args.verdict_predictions,
              args.first_turn_control, args.output)
