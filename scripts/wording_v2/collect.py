"""Collect frozen third-cohort conversations on the pinned Qwen checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forecast_mechanisms.collect import collect
from forecast_mechanisms.core import (
    FORMATS, append_jsonl, parse_response, prompt_digest, read_jsonl,
    verify_complete_run,
)
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION, ForecastRunner


WORK = Path.cwd()
CASES = WORK / "cross-wording-v2-cases.jsonl"
CONFIG = WORK / "feedback-wording-validation-v1.json"
OUT = WORK / "cross-wording-v2"
CASE_SHA = "8b2dc2783229e172c730e1a4dd8a137de26b4968af4348533ea0813eb80230eb"
CONFIG_SHA = "8200b16dd9525289fd42e2499e524517d9bb8378b6ab2ec6c932e17b71edb6d6"
SUFFIX = "\nPlease reply with exactly these lines and no explanation:\n" + FORMATS["p_first"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs() -> dict:
    if digest(CASES) != CASE_SHA or digest(CONFIG) != CONFIG_SHA:
        raise ValueError("frozen case or wording file changed")
    config = json.loads(CONFIG.read_text())
    if (config["model"], config["revision"]) != (DEFAULT_MODEL, DEFAULT_REVISION):
        raise ValueError("model identity changed")
    cases = read_jsonl(CASES)
    if len(cases) != 160 or any("outcome" in row for row in cases):
        raise ValueError("third-cohort case file is malformed")
    return config


def baseline() -> None:
    check_inputs()
    runner = ForecastRunner(max_new_tokens=96)
    result = collect(
        CASES, OUT / "baseline", runner, seed=42, arms=("p_first",),
        second_sources=("p_first",), followups=("neutral", "opposite"),
        prompt_voice="first_person",
    )
    complete = verify_complete_run(OUT / "baseline")
    rows = read_jsonl(OUT / "baseline" / "records.jsonl")
    first = [row for row in rows if row["round"] == 1]
    second = [row for row in rows if row["round"] == 2]
    if (len(first) != 160 or len(second) != 2 * sum(row["parse_ok"] for row in first)
            or any(not row["parse_ok"] for row in second)):
        raise ValueError("third-cohort baseline has unexplained missing or unparsed answers")
    print(json.dumps({"collection": result, "completion": complete}, sort_keys=True))


def variants(max_cases: int | None) -> None:
    config = check_inputs()
    baseline_complete = verify_complete_run(OUT / "baseline")
    original = read_jsonl(OUT / "baseline" / "records.jsonl")
    first = {row["case_id"]: row for row in original
             if row["round"] == 1 and row["arm"] == "p_first"}
    if len(first) != 160:
        raise ValueError("first-turn baseline incomplete")
    valid_first = {case_id for case_id, row in first.items() if row["parse_ok"]}
    destination = OUT / "variants"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "cross-wording-prefix-route-v2", "cases_sha256": CASE_SHA,
        "wording_config_sha256": CONFIG_SHA,
        "baseline_records_sha256": baseline_complete["records_sha256"],
        "script_sha256": digest(Path(__file__)),
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
    }
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("variant manifest differs; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    records = destination / "records.jsonl"
    previous = read_jsonl(records) if records.exists() else []
    done = {(row["case_id"], row["condition"]) for row in previous}
    if len(done) != len(previous):
        raise ValueError("duplicate variant output")
    runner = ForecastRunner(max_new_tokens=96)
    completed_cases = 0
    for case in read_jsonl(CASES):
        origin = first[case["case_id"]]
        if case["case_id"] not in valid_first:
            continue
        if all((case["case_id"], condition) in done for condition in config["conditions"]):
            continue
        for condition, templates in config["conditions"].items():
            key = (case["case_id"], condition)
            if key in done:
                continue
            wording = templates[case["domain"]]
            lead = wording[origin["verdict"]] if isinstance(wording, dict) else wording
            messages = origin["messages"] + [
                {"role": "assistant", "content": origin["response"]},
                {"role": "user", "content": lead + SUFFIX},
            ]
            response = runner.generate(messages)
            row = {
                "case_id": case["case_id"], "domain": case["domain"],
                "condition": condition, "initial_verdict": origin["verdict"],
                "messages": messages, "prompt_sha256": prompt_digest(messages),
                "response": response, **parse_response(response, "p_first"),
            }
            append_jsonl(records, row)
            done.add(key)
        completed_cases += 1
        if max_cases is not None and completed_cases >= max_cases:
            break
    if len(done) == len(valid_first) * len(config["conditions"]):
        rows = read_jsonl(records)
        (destination / "completion.json").write_text(json.dumps({
            "records_sha256": digest(records), "total_records": len(done),
            "first_round_parse_failures": 160 - len(valid_first),
            "variant_parse_failures": sum(not row["parse_ok"] for row in rows),
        }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed_cases": completed_cases, "total_records": len(done)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("baseline", "variants"))
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.phase == "baseline":
        baseline()
    else:
        variants(args.max_cases)
