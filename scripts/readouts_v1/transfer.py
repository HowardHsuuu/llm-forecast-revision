"""Capture pre-answer wording states and evaluate a frozen percentage readout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error

from forecast_mechanisms.core import append_jsonl, read_jsonl, verify_complete_run, write_jsonl
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION
from common import TRAIN_SHA, TEST_SHA, bootstrap, cohort, fields, freeze, save, sha


SOURCE_SHA = "a3433571051e4fe340d1d637fd510bf817a101e5b6fde7afa35dcd586eba6eda"
CONDITIONS = ("paraphrase_a", "paraphrase_b", "unrelated_expectation")


def load_model(path):
    marker = json.loads(path.with_name("fitted_readout_manifest.json").read_text())
    if marker["file_sha256"] != sha(path):
        raise ValueError("fitted percentage readout changed")
    summary_path = path.with_name("summary.json")
    if sha(summary_path) != marker["source_probe_summary_sha256"]:
        raise ValueError("fitted readout's source summary changed")
    summary = json.loads(summary_path.read_text())
    identity = marker if "model" in marker else summary
    if ((identity["model"], identity["revision"]) != (DEFAULT_MODEL, DEFAULT_REVISION) or
            marker.get("train_records_sha256", summary.get("train_records_sha256")) != TRAIN_SHA):
        raise ValueError("fitted readout has a different target checkpoint or training cohort")
    with np.load(path, allow_pickle=False) as saved:
        model = {name: saved[name] for name in saved.files}
    index = int(model["selected_state_index"])
    if (index != marker["selected_state_index"] or not 1 <= index <= 27 or
            float(model["selected_alpha"]) != marker["selected_alpha"]):
        raise ValueError("fitted readout selection metadata differs")
    for group, size in (("baseline", 9), ("state", 3584)):
        for suffix in ("scaler_mean", "scaler_scale", "coef"):
            value = model[group + "_" + suffix]
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError("fitted percentage readout has invalid dimensions or values")
        if np.any(model[group + "_scaler_scale"] <= 0):
            raise ValueError("invalid fitted scaler")
        if not np.isfinite(model[group + "_intercept"]).all():
            raise ValueError("invalid fitted intercept")
    return model


def source_rows(source):
    marker = verify_complete_run(source)
    if marker["records_sha256"] != SOURCE_SHA:
        raise ValueError("wording source differs from the frozen test conversations")
    rows = read_jsonl(source / "records.jsonl")
    keys = {(r["case_id"], r["condition"]) for r in rows}
    if (len(rows) != 474 or len(keys) != 474 or
            any(sum(r["condition"] == c for r in rows) != 158 for c in CONDITIONS) or
            any(not r["parse_ok"] for r in rows)):
        raise ValueError("wording source matrix is incomplete")
    return rows


def validate_capture(row, source, directory):
    for name in ("case_id", "condition", "domain", "prompt_sha256"):
        if row[name] != source[name]:
            raise ValueError("captured state metadata differs from its wording prompt")
    path = (directory / row["activation_file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("captured activation is outside the state directory")
    value = np.load(path, allow_pickle=False)
    if value.shape != (3584,) or not np.isfinite(value).all():
        raise ValueError("expected one finite selected-layer state of dimension 3584")
    return value


def capture(source, model_path, out, max_rows):
    if max_rows is not None and max_rows < 1:
        raise ValueError("max-rows must be positive")
    rows = source_rows(source)
    model = load_model(model_path)
    index = int(model["selected_state_index"])
    freeze(out, {"protocol": "pre-answer-readout-transfer-capture-v1", "source_sha256": SOURCE_SHA,
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION, "state_index": index,
        "fitted_readout_sha256": sha(model_path), "script_sha256": sha(__file__)})
    path = out / "records.jsonl"
    previous = read_jsonl(path) if path.exists() else []
    expected = {(r["case_id"], r["condition"]): r for r in rows}
    done = {(r["case_id"], r["condition"]) for r in previous}
    if len(done) != len(previous) or not done <= set(expected):
        raise ValueError("saved state collection has duplicate or unexpected keys")
    for row in previous:
        validate_capture(row, expected[row["case_id"], row["condition"]], out)
    if done != set(expected):
        from forecast_mechanisms.model import ForecastRunner
        runner = ForecastRunner(max_new_tokens=96)
        captured = 0
        for row in rows:
            key = row["case_id"], row["condition"]
            if key in done:
                continue
            if tuple(m["role"] for m in row["messages"]) != ("system", "user", "assistant", "user"):
                raise ValueError("unexpected wording conversation")
            matrix = runner.preanswer_states(row["messages"])
            if matrix.shape != (29, 3584) or not np.isfinite(matrix).all():
                raise ValueError("invalid pre-answer state matrix")
            name = hashlib.sha256(f"{key[0]}|{key[1]}".encode()).hexdigest() + ".npy"
            target = out / "activations" / name
            target.parent.mkdir(exist_ok=True)
            np.save(target, matrix[index].astype(np.float32), allow_pickle=False)
            append_jsonl(path, {"case_id": key[0], "condition": key[1], "domain": row["domain"],
                               "prompt_sha256": row["prompt_sha256"],
                               "activation_file": "activations/" + name})
            done.add(key)
            captured += 1
            if max_rows is not None and captured >= max_rows:
                break
    if done == set(expected):
        save(out / "completion.json", {"total_records": len(done), "records_sha256": sha(path)})
    print(json.dumps({"captured_states": len(done), "expected": len(expected)}))


def predict(model, group, x):
    scaled = (x - model[group + "_scaler_mean"]) / model[group + "_scaler_scale"]
    return scaled @ model[group + "_coef"] + model[group + "_intercept"]


def evaluate(source, test, state_dir, model_path, out):
    source_index = {(r["case_id"], r["condition"]): r for r in source_rows(source)}
    test_rows = cohort(test, TEST_SHA, activations=False)
    test_index = {r["case_id"]: r for r in test_rows}
    model = load_model(model_path)
    verify_complete_run(state_dir)
    manifest = json.loads((state_dir / "manifest.json").read_text())
    if ((manifest["model"], manifest["revision"], manifest["state_index"], manifest["source_sha256"]) !=
            (DEFAULT_MODEL, DEFAULT_REVISION, int(model["selected_state_index"]), SOURCE_SHA)):
        raise ValueError("captured states differ from the fitted readout position or target model")
    if manifest.get("fitted_readout_sha256", sha(model_path)) != sha(model_path):
        raise ValueError("capture used a different fitted readout")
    rows = read_jsonl(state_dir / "records.jsonl")
    keys = [(r["case_id"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(source_index):
        raise ValueError("captured wording state matrix differs from its source")
    freeze(out, {"protocol": "pre-answer-readout-transfer-evaluation-v1", "source_sha256": SOURCE_SHA,
        "test_records_sha256": TEST_SHA, "state_records_sha256": sha(state_dir / "records.jsonl"),
        "fitted_readout_sha256": sha(model_path), "script_sha256": sha(__file__)})
    predictions, results = [], {}
    for condition in CONDITIONS:
        group = sorted((r for r in rows if r["condition"] == condition), key=lambda r: r["case_id"])
        if {r["case_id"] for r in group} != set(test_index) or len(group) != 158:
            raise ValueError("wording group differs from the 158 parsed test cases")
        x = np.stack([validate_capture(r, source_index[r["case_id"], condition], state_dir)
                      for r in group]).astype(np.float64)
        baseline = predict(model, "baseline", fields([test_index[r["case_id"]] for r in group]))
        predicted = np.clip(baseline + predict(model, "state", x), 0, 100)
        target = np.asarray([source_index[r["case_id"], condition]["p_accept"] for r in group])
        results[condition] = {}
        for domain in ("all", "grad_admissions", "paper_acceptance"):
            mask = np.asarray([domain == "all" or r["domain"] == domain for r in group])
            gain = np.abs(target[mask] - baseline[mask]) - np.abs(target[mask] - predicted[mask])
            results[condition][domain] = {"n": int(mask.sum()),
                "baseline_mae": float(mean_absolute_error(target[mask], baseline[mask])),
                "probe_mae": float(mean_absolute_error(target[mask], predicted[mask])),
                "mae_gain": float(gain.mean()), "mae_gain_ci95": bootstrap(gain)}
        predictions.extend({"case_id": r["case_id"], "condition": condition, "domain": r["domain"],
            "p_accept": int(target[i]), "baseline_p": float(baseline[i]),
            "probe_p": float(predicted[i])} for i, r in enumerate(group))
    write_jsonl(out / "predictions.jsonl", predictions)
    summary = {"protocol": "pre-answer-readout-transfer-summary-v1", "n_total": len(predictions),
        "source_records_sha256": SOURCE_SHA, "test_records_sha256": TEST_SHA,
        "script_sha256": sha(__file__),
        "selected_block": int(model["selected_state_index"]) - 1,
        "selected_alpha": float(model["selected_alpha"]), "conditions": results,
        "state_records_sha256": sha(state_dir / "records.jsonl"),
        "fitted_readout_sha256": sha(model_path), "predictions_sha256": sha(out / "predictions.jsonl")}
    save(out / "summary.json", summary)
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("capture", "evaluate"):
        sub = commands.add_parser(action)
        for name in ("source", "model", "output"):
            sub.add_argument("--" + name, type=Path, required=True)
        if action == "capture":
            sub.add_argument("--max-rows", type=int)
        else:
            sub.add_argument("--test", type=Path, required=True)
            sub.add_argument("--states", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "capture":
        capture(args.source, args.model, args.output, args.max_rows)
    else:
        evaluate(args.source, args.test, args.states, args.model, args.output)
