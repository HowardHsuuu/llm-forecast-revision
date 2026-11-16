"""Train-only layer and regularization selection for pre-answer numeric readouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from forecast_mechanisms.core import write_jsonl
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION
from common import (TRAIN_SHA, TEST_SHA, INDICES, SEED, bootstrap, fields, fit_ridge,
                    freeze, load_pair, ridge_predict, save, sha, states)


ALPHAS = (1., 10., 100., 1000., 10000.)


def fit(train, test, out):
    tr, te = load_pair(train, test)
    freeze(out, {"protocol": "pre-answer-percentage-readout-v1", "train_sha256": TRAIN_SHA,
                 "test_sha256": TEST_SHA, "script_sha256": sha(__file__),
                 "common_sha256": sha(Path(__file__).with_name("common.py")),
                 "state_indices": INDICES, "alphas": ALPHAS, "seed": SEED})
    btr, bte = fields(tr), fields(te)
    ytr = np.asarray([r["opposite_p"] for r in tr])
    yte = np.asarray([r["opposite_p"] for r in te])
    cv = np.full((len(INDICES), len(ALPHAS), len(tr)), np.nan)
    cv_base = np.full(len(tr), np.nan)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_ids, val_ids in folds.split(btr):
        base = fit_ridge(btr[fit_ids], ytr[fit_ids], 10.)
        fit_base = ridge_predict(base, btr[fit_ids])
        val_base = ridge_predict(base, btr[val_ids])
        cv_base[val_ids] = val_base
        for j, index in enumerate(INDICES):
            x = states(tr, "opposite", index)
            for k, alpha in enumerate(ALPHAS):
                model = fit_ridge(x[fit_ids], ytr[fit_ids] - fit_base, alpha)
                cv[j, k, val_ids] = val_base + ridge_predict(model, x[val_ids])
    if not np.isfinite(cv).all():
        raise ValueError("incomplete percentage cross-validation")
    losses = np.mean(np.abs(cv - ytr[None, None, :]), axis=-1)
    chosen = np.unravel_index(int(np.argmin(losses)), losses.shape)
    index, alpha = INDICES[chosen[0]], ALPHAS[chosen[1]]
    base = fit_ridge(btr, ytr, 10.)
    baseline = ridge_predict(base, bte)
    model = fit_ridge(states(tr, "opposite", index), ytr - ridge_predict(base, btr), alpha)
    predicted = np.clip(baseline + ridge_predict(model, states(te, "opposite", index)), 0, 100)
    gain = np.abs(yte - baseline) - np.abs(yte - predicted)
    write_jsonl(out / "predictions.jsonl", [{"case_id": r["case_id"], "domain": r["domain"],
        "first_p": r["first_p"], "first_verdict": r["first_verdict"],
        "opposite_p": r["opposite_p"], "baseline_p": float(baseline[i]),
        "probe_p": float(predicted[i])} for i, r in enumerate(te)])
    summary = {"n_train": len(tr), "n_test": len(te), "excluded_test": 2,
               "selected_state_index": index, "selected_block": index - 1, "selected_alpha": alpha,
               "cv_selected_mae": float(losses[chosen]),
               "cv_baseline_mae": float(mean_absolute_error(ytr, cv_base)),
               "test_probe_mae": float(mean_absolute_error(yte, predicted)),
               "test_field_baseline_mae": float(mean_absolute_error(yte, baseline)),
               "test_copy_first_mae": float(mean_absolute_error(yte, [r["first_p"] for r in te])),
               "mae_gain": float(gain.mean()), "mae_gain_ci95": bootstrap(gain),
               "predictions_sha256": sha(out / "predictions.jsonl")}
    save(out / "cv.json", {"state_indices": INDICES, "alphas": ALPHAS, "mae": losses.tolist()})
    save(out / "summary.json", summary)
    np.savez(out / "fitted_readout.npz", selected_state_index=index, selected_alpha=alpha,
        baseline_scaler_mean=base[0].mean_, baseline_scaler_scale=base[0].scale_,
        baseline_coef=base[1].coef_, baseline_intercept=base[1].intercept_,
        state_scaler_mean=model[0].mean_, state_scaler_scale=model[0].scale_,
        state_coef=model[1].coef_, state_intercept=model[1].intercept_)
    save(out / "fitted_readout_manifest.json", {"protocol": "pre-answer-percentage-fit-v1",
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
        "train_records_sha256": TRAIN_SHA, "selected_state_index": index, "selected_alpha": alpha,
        "file_sha256": sha(out / "fitted_readout.npz"),
        "source_probe_summary_sha256": sha(out / "summary.json")})
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("train", "test", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    fit(args.train, args.test, args.output)
