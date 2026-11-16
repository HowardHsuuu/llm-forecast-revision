"""Frozen cohorts, feature construction, and estimators for pre-answer readouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from forecast_mechanisms.core import read_jsonl, verify_complete_run
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION


TRAIN_SHA = "76c1954d3a44c345ef3556f79adf840a93bd0446cc34160ee61ccd87daa64a05"
TEST_SHA = "194543ab49fabd2b8dc68a80ae3c7a7f56dfd401bf13afd3fe91f734279809cf"
INDICES = tuple(range(1, 28))
SEED = 20261115


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze(out, value):
    value = json.loads(json.dumps(value))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("readout inputs or settings changed; use a new output directory")
    save(path, value)


def cohort(path, digest, *, activations=True):
    completion = verify_complete_run(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if (completion["records_sha256"] != digest or
            (manifest["model"], manifest["revision"]) != (DEFAULT_MODEL, DEFAULT_REVISION) or
            manifest.get("prompt_voice") != "first_person"):
        raise ValueError("readout cohort differs from the frozen first-person collection")
    cases = {}
    for row in read_jsonl(path / "records.jsonl"):
        condition = "first" if row["round"] == 1 else row["condition"]
        if row["arm"] != "p_first" or condition not in ("first", "neutral", "opposite", "aligned"):
            raise ValueError("unexpected readout source condition")
        bucket = cases.setdefault(row["case_id"], {})
        if condition in bucket:
            raise ValueError("duplicate readout source condition")
        bucket[condition] = row
    if len(cases) != 160:
        raise ValueError("expected 160 source cases")
    result = []
    for case_id, pair in sorted(cases.items()):
        if not all(k in pair for k in ("first", "neutral", "opposite")):
            continue
        first, neutral, opposite = (pair[k] for k in ("first", "neutral", "opposite"))
        if any(not r["parse_ok"] or r["p_accept"] is None or r["verdict"] is None
               for r in (first, neutral, opposite)):
            continue
        item = {"case_id": case_id, "domain": first["domain"],
                "first_p": float(first["p_accept"]), "first_verdict": first["verdict"],
                "neutral_p": float(neutral["p_accept"]), "neutral_verdict": neutral["verdict"],
                "opposite_p": float(opposite["p_accept"]), "opposite_verdict": opposite["verdict"]}
        if activations:
            for condition in ("first", "neutral", "opposite"):
                source = (path / pair[condition]["activation_file"]).resolve()
                if not source.is_relative_to(path.resolve()):
                    raise ValueError("activation is outside the collection directory")
                matrix = np.load(source, allow_pickle=False)
                if matrix.shape != (29, 3584) or not np.isfinite(matrix).all():
                    raise ValueError("invalid pre-answer activation matrix")
                item[condition + "_state"] = matrix.astype(np.float32, copy=False)
        result.append(item)
    return result


def load_pair(train, test, *, activations=True):
    tr = cohort(train, TRAIN_SHA, activations=activations)
    te = cohort(test, TEST_SHA, activations=activations)
    if (len(tr), len(te)) != (160, 158):
        raise ValueError("frozen parsed cohort counts changed")
    if {r["case_id"] for r in tr} & {r["case_id"] for r in te}:
        raise ValueError("training and test cases overlap")
    return tr, te


def fields(rows):
    simple = np.asarray([[r["first_p"], int(r["first_verdict"] == "accept"),
                          int(r["domain"] == "paper_acceptance")] for r in rows], dtype=np.float64)
    return PolynomialFeatures(degree=2, include_bias=False).fit_transform(simple)


def verdict_fields(rows):
    return np.asarray([[r["first_p"], float(r["domain"] == "paper_acceptance"),
                        r["first_p"] ** 2, r["first_p"] * float(r["domain"] == "paper_acceptance")]
                       for r in rows], dtype=np.float64)


def states(rows, condition, index):
    return np.stack([r[condition + "_state"][index] for r in rows]).astype(np.float64)


def fit_ridge(x, y, alpha):
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha, solver="cholesky").fit(scaler.transform(x), y)
    return scaler, model


def ridge_predict(fit, x):
    return fit[1].predict(fit[0].transform(x))


def fit_logistic(b, x, y, c):
    bs = StandardScaler().fit(b)
    xs = StandardScaler().fit(x) if x is not None else None
    design = bs.transform(b) if xs is None else np.column_stack((bs.transform(b), xs.transform(x)))
    model = LogisticRegression(C=c, solver="liblinear", max_iter=3000,
                               random_state=SEED).fit(design, y)
    return bs, xs, model


def logistic_predict(fit, b, x):
    bs, xs, model = fit
    design = bs.transform(b) if xs is None else np.column_stack((bs.transform(b), xs.transform(x)))
    return model.predict_proba(design)[:, 1]


def metrics(y, p):
    return {"n": len(y), "positive": int(y.sum()), "accuracy": float(np.mean((p >= .5) == y)),
            "auroc": float(roc_auc_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p))}


def bootstrap(values):
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(values), size=(10000, len(values)))
    return np.quantile(np.mean(values[draws], axis=1), (.025, .975)).tolist()
