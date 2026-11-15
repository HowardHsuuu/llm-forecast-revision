"""Reproduce the frozen generation-time verdict-steering confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from forecast_mechanisms.core import append_jsonl, read_jsonl
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION, ForecastRunner
from forecast_mechanisms.steering import generate_with_vector, orthogonal_control


CASE_SHA = "15f8f9bb81d4c94ea9a9f10b94ef42119f53e33b3fe8de286171665a7b144aeb"
BASELINE_SHA = "0d25bdfbdf0652cdff66de68df074539dda7a24e332a6b8c67201208c3a61f92"
DIRECTION_SHA = "424d6b6c8f22e9ff1c7d2ae083111232f0ebdd15b4f864f14d72d31fd4fd6ab4"
OPPOSITE = ("zero", "high_plus", "high_orth_plus", "low_plus",
            "low_orth_plus", "high_minus")
NEUTRAL = ("zero", "high_plus")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(baseline: Path) -> dict[str, dict]:
    manifest = json.loads((baseline / "manifest.json").read_text())
    completion = json.loads((baseline / "completion.json").read_text())
    records = baseline / "records.jsonl"
    rows = read_jsonl(records)
    if (manifest["cases_sha256"] != CASE_SHA or
            manifest.get("prompt_voice") != "first_person" or
            (manifest["model"], manifest["revision"]) !=
            (DEFAULT_MODEL, DEFAULT_REVISION) or
            set(manifest["followups"]) != {"neutral", "opposite"} or
            sha(records) != BASELINE_SHA or
            completion["records_sha256"] != BASELINE_SHA or
            completion["total_records"] != len(rows)):
        raise ValueError("frozen first-person baseline changed or incomplete")
    cases: dict[str, dict] = {}
    for row in rows:
        if row["arm"] != "p_first":
            raise ValueError("unexpected output arm")
        key = "first" if row["round"] == 1 else row["condition"]
        bucket = cases.setdefault(row["case_id"], {})
        if key in bucket:
            raise ValueError("duplicate response condition")
        bucket[key] = row
    if len(cases) != 160 or any(set(pair) != {"first", "neutral", "opposite"}
                                or any(not r["parse_ok"] for r in pair.values())
                                for pair in cases.values()):
        raise ValueError("frozen baseline case flow changed")
    return cases


def main(baseline: Path, direction: Path, output: Path,
         max_cases: int | None) -> None:
    if (DEFAULT_MODEL, DEFAULT_REVISION) != (
            "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"):
        raise ValueError("model revision changed")
    if sha(direction) != DIRECTION_SHA:
        raise ValueError("frozen training direction changed")
    arrays = np.load(direction, allow_pickle=False)
    block = int(arrays["state_index"]) - 1
    unit = np.asarray(arrays["paired_neutral"], dtype=np.float32)
    doses = {name: float(arrays[f"{name}_dose"]) for name in ("low", "high")}
    if (block != 21 or unit.shape != (3584,) or not np.isfinite(unit).all() or
            abs(np.linalg.norm(unit) - 1) > 1e-4 or
            not 14 < doses["low"] < 15 or not 28 < doses["high"] < 30):
        raise ValueError("frozen block, vector, or dose changed")
    cases = load_cases(baseline)
    manifest = {"protocol": "generation-steering-confirmation-v1",
                "script_sha256": sha(Path(__file__)),
                "baseline_records_sha256": BASELINE_SHA,
                "directions_sha256": DIRECTION_SHA,
                "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
                "block": block, "dose_norms": doses,
                "opposite_conditions": list(OPPOSITE),
                "neutral_conditions": list(NEUTRAL),
                "intervention_steps": "all generated token positions after prompt prefill"}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("run manifest changed")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    records = output / "records.jsonl"
    previous = read_jsonl(records) if records.exists() else []
    done = {r["case_id"] for r in previous}
    if len(done) != len(previous) or not done <= set(cases):
        raise ValueError("duplicate or unexpected run case")
    runner = ForecastRunner(max_new_tokens=96)
    count = 0
    for case_id in sorted(cases):
        if case_id in done:
            continue
        pair = cases[case_id]
        orth = orthogonal_control(unit, case_id, "generation:paired_neutral")
        deltas = {"zero": np.zeros(3584, dtype=np.float32),
                  "low_plus": unit * doses["low"],
                  "low_orth_plus": orth * doses["low"],
                  "high_plus": unit * doses["high"],
                  "high_orth_plus": orth * doses["high"],
                  "high_minus": -unit * doses["high"]}
        result = {"case_id": case_id, "domain": pair["first"]["domain"],
                  "first_p": pair["first"]["p_accept"],
                  "first_verdict": pair["first"]["verdict"], "contexts": {}}
        for context, conditions in (("opposite", OPPOSITE), ("neutral", NEUTRAL)):
            row = pair[context]
            outputs = {name: generate_with_vector(runner, row["messages"],
                                                  block, deltas[name])
                       for name in conditions}
            zero = outputs["zero"]
            if (zero["response"] != row["response"] or not zero["parse_ok"] or
                    (zero["p_accept"], zero["verdict"]) !=
                    (row["p_accept"], row["verdict"])):
                raise ValueError(f"zero-dose replay mismatch: {case_id} {context}")
            if any(not v["parse_ok"] or not v["prefix_ok"] for v in outputs.values()):
                raise ValueError(f"unparsed or changed answer prefix: {case_id} {context}")
            result["contexts"][context] = {"conditions": outputs}
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    main(args.baseline, args.direction, args.output, args.max_cases)
