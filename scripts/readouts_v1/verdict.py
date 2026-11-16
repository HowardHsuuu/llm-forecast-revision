"""Read the forthcoming verdict before the followup answer and before its prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from forecast_mechanisms.core import write_jsonl
from common import (TRAIN_SHA, TEST_SHA, INDICES, SEED, fit_logistic, freeze,
                    load_pair, logistic_predict, metrics, save, sha, states, verdict_fields)


CS = (.001, .01, .1, 1.)


def selected_fit(tr, te, condition, btr, bte, ytr, yte):
    cv = np.full((len(INDICES), len(CS), len(tr)), np.nan)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_ids, val_ids in folds.split(btr, ytr):
        for j, index in enumerate(INDICES):
            x = states(tr, condition, index)
            for k, c in enumerate(CS):
                model = fit_logistic(btr[fit_ids], x[fit_ids], ytr[fit_ids], c)
                cv[j, k, val_ids] = logistic_predict(model, btr[val_ids], x[val_ids])
    if not np.isfinite(cv).all():
        raise ValueError("incomplete verdict cross-validation")
    safe = np.clip(cv, 1e-7, 1 - 1e-7)
    losses = np.mean(-(ytr[None, None, :] * np.log(safe) +
                       (1 - ytr[None, None, :]) * np.log(1 - safe)), axis=-1)
    chosen = np.unravel_index(int(np.argmin(losses)), losses.shape)
    index, c = INDICES[chosen[0]], CS[chosen[1]]
    model = fit_logistic(btr, states(tr, condition, index), ytr, c)
    predicted = logistic_predict(model, bte, states(te, condition, index))
    return predicted, {"selected_state_index": index, "selected_block": index - 1,
                       "selected_c": c, "cv_logloss": float(losses[chosen]),
                       "test": metrics(yte, predicted)}, losses


def fit(train, test, out):
    tr, te = load_pair(train, test)
    tr = [r for r in tr if r["first_verdict"] == "accept"]
    te = [r for r in te if r["first_verdict"] == "accept"]
    ytr = np.asarray([r["opposite_verdict"] == "accept" for r in tr], dtype=int)
    yte = np.asarray([r["opposite_verdict"] == "accept" for r in te], dtype=int)
    if (len(tr), len(te), int(ytr.sum()), int(yte.sum())) != (147, 137, 70, 63):
        raise ValueError("verdict stability cohort changed")
    freeze(out, {"protocol": "pre-answer-verdict-readout-v1", "train_sha256": TRAIN_SHA,
                 "test_sha256": TEST_SHA, "script_sha256": sha(__file__),
                 "common_sha256": sha(Path(__file__).with_name("common.py")),
                 "state_indices": INDICES, "cs": CS, "seed": SEED})
    btr, bte = verdict_fields(tr), verdict_fields(te)
    field_model = fit_logistic(btr, None, ytr, 1.)
    field_pred = logistic_predict(field_model, bte, None)
    second, second_summary, second_cv = selected_fit(tr, te, "opposite", btr, bte, ytr, yte)
    first, first_summary, first_cv = selected_fit(tr, te, "first", btr, bte, ytr, yte)
    write_jsonl(out / "predictions.jsonl", [{"case_id": r["case_id"], "domain": r["domain"],
        "first_p": r["first_p"], "opposite_verdict": r["opposite_verdict"], "stable": bool(yte[i]),
        "baseline_prob": float(field_pred[i]), "state_prob": float(second[i]),
        "first_turn_prob": float(first[i])} for i, r in enumerate(te)])
    save(out / "summary.json", {"n_train": len(tr), "n_test": len(te), "stable_test": 63,
         "field_baseline": metrics(yte, field_pred), "second_turn_pre_answer": second_summary,
         "first_turn_control": first_summary,
         "first_turn_control_status": "Post-hoc control using the same independent test cases.",
         "predictions_sha256": sha(out / "predictions.jsonl")})
    for condition, values in (("second", second_cv), ("first", first_cv)):
        save(out / (condition + "-cv.json"), {"state_indices": INDICES, "cs": CS,
                                             "log_loss": values.tolist()})
    print({"second_turn_auroc": second_summary["test"]["auroc"],
           "first_turn_auroc": first_summary["test"]["auroc"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("train", "test", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    fit(args.train, args.test, args.output)
