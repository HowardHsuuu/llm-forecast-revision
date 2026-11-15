"""Reproduce the fixed first-number and first-verdict intervention checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forecast_mechanisms.core import append_jsonl, read_jsonl
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION, ForecastRunner
from forecast_mechanisms.steering import orthogonal_control
from forecast_mechanisms.steering_steps import (
    natural_verdict_state, number_once, verdict_once,
)
from run import DIRECTION_SHA, load_cases, sha


SUSTAINED_SHA = "362482f1fd04d48095ae146321194d02e9b0ef18f546b54c2ad503c0971a7b9d"
CONDITIONS = ("zero", "plus", "minus", "orth_plus")


def main(baseline: Path, direction: Path, sustained: Path, output: Path,
         max_cases: int | None) -> None:
    if (DEFAULT_MODEL, DEFAULT_REVISION) != (
            "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"):
        raise ValueError("pinned model changed")
    if sha(direction) != DIRECTION_SHA:
        raise ValueError("frozen training vector changed")
    sustained_file = sustained / "records.jsonl"
    marker = json.loads((sustained / "completion.json").read_text())
    if marker["records_sha256"] != SUSTAINED_SHA or sha(sustained_file) != SUSTAINED_SHA:
        raise ValueError("sustained confirmation source changed")
    cases = load_cases(baseline)
    sustained_ids = {r["case_id"] for r in read_jsonl(sustained_file)}
    if sustained_ids != set(cases):
        raise ValueError("confirmation cohort changed")
    arrays = np.load(direction, allow_pickle=False)
    block = int(arrays["state_index"]) - 1
    unit = np.asarray(arrays["paired_neutral"], dtype=np.float32)
    dose = float(arrays["high_dose"])
    if (block != 21 or unit.shape != (3584,) or not np.isfinite(unit).all() or
            abs(np.linalg.norm(unit) - 1) > 1e-4 or not 28 < dose < 30):
        raise ValueError("frozen vector, block, or dose changed")
    manifest = {"protocol": "fresh-decision-steps-v1",
                "script_sha256": sha(Path(__file__)),
                "baseline_records_sha256": sha(baseline / "records.jsonl"),
                "sustained_records_sha256": SUSTAINED_SHA,
                "directions_sha256": DIRECTION_SHA,
                "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
                "block": block, "dose_norm": dose,
                "number_step": 5,
                "conditions": list(CONDITIONS),
                "loci": ["first_number", "first_verdict"],
                "expected_cases": len(cases)}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("decision-step manifest changed")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    records = output / "records.jsonl"
    previous = read_jsonl(records) if records.exists() else []
    done = {r["case_id"] for r in previous}
    if len(done) != len(previous) or not done <= set(cases):
        raise ValueError("duplicate or unexpected decision-step case")
    runner = ForecastRunner(max_new_tokens=96)
    accept = runner.tokenizer.encode(" ACCEPT", add_special_tokens=False)
    reject = runner.tokenizer.encode(" REJECT", add_special_tokens=False)
    if len(accept) != 1 or not reject or accept[0] == reject[0]:
        raise ValueError("verdict tokenization changed")
    count = 0
    for case_id in sorted(cases):
        if case_id in done:
            continue
        pair = cases[case_id]
        row = pair["opposite"]
        step, state, prefix_digest = natural_verdict_state(
            runner, row["messages"], row["response"], block)
        orth = orthogonal_control(unit, case_id, "generation:paired_neutral")
        deltas = {"zero": np.zeros(3584, dtype=np.float32),
                  "plus": unit * dose, "minus": -unit * dose,
                  "orth_plus": orth * dose}
        numeric = {key: number_once(runner, row["messages"], block, deltas[key])
                   for key in CONDITIONS}
        verdict = {key: verdict_once(runner, row["messages"], block, step,
                                     state, deltas[key], prefix_digest,
                                     accept[0], reject[0]) for key in CONDITIONS}
        for outputs in (numeric, verdict):
            zero = outputs["zero"]
            if (zero["response"] != row["response"] or not zero["parse_ok"] or
                    (zero["p_accept"], zero["verdict"]) !=
                    (row["p_accept"], row["verdict"])):
                raise ValueError(f"zero replay mismatch: {case_id}")
            if any(not z["parse_ok"] for z in outputs.values()):
                raise ValueError(f"unparsed edited answer: {case_id}")
        if any(z["source_state_error"] > 1e-3 for z in verdict.values()):
            raise ValueError("verdict-step state mismatch")
        if any(z["p_accept"] != row["p_accept"] for z in verdict.values()):
            raise ValueError("verdict edit changed an already-written percentage")
        result = {"case_id": case_id, "domain": row["domain"],
                  "first_p": pair["first"]["p_accept"],
                  "first_verdict": pair["first"]["verdict"],
                  "natural_opposite_p": row["p_accept"],
                  "natural_opposite_verdict": row["verdict"],
                  "verdict_step": step,
                  "verdict_source_state_norm": float(np.linalg.norm(state)),
                  "loci": {"first_number": numeric, "first_verdict": verdict}}
        append_jsonl(records, result)
        done.add(case_id)
        count += 1
        if max_cases is not None and count >= max_cases:
            break
    if done == set(cases):
        (output / "completion.json").write_text(json.dumps({
            "total_records": len(done), "records_sha256": sha(records),
        }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed_this_run": count, "total_records": len(done),
                      "expected": len(cases)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--direction", type=Path, required=True)
    parser.add_argument("--sustained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    main(args.baseline, args.direction, args.sustained, args.output,
         args.max_cases)
