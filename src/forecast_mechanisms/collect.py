"""Resumable collection of first and second round pre-answer states."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    ARMS, FOLLOWUPS, PROMPT_VOICES, append_jsonl, first_messages, parse_response,
    prompt_digest, read_jsonl, second_messages, split_for_case, validate_cases,
)


def _key(case_id: str, round_no: int, arm: str, condition: str | None) -> str:
    return f"{case_id}|{round_no}|{arm}|{condition or '-'}"


def _save_array(path: Path, states: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npy.tmp")
    with temp.open("wb") as handle:
        np.save(handle, states.astype(np.float32, copy=False), allow_pickle=False)
    temp.replace(path)


def _record(
    case: dict[str, Any], round_no: int, arm: str, condition: str | None,
    messages: list[dict[str, str]], response: str, activation_file: str,
    seed: int, model_name: str, revision: str,
) -> dict[str, Any]:
    parsed = parse_response(response, "p_first" if round_no == 2 else arm)
    return {
        "key": _key(case["case_id"], round_no, arm, condition),
        "case_id": case["case_id"], "domain": case["domain"],
        "source_ref": case["source_ref"], "audit_status": case["audit_status"],
        "outcome": case.get("outcome"), "split": split_for_case(case["case_id"], seed),
        "round": round_no, "arm": arm, "condition": condition,
        "messages": messages, "prompt_sha256": prompt_digest(messages),
        "response": response, "activation_file": activation_file,
        "model": model_name, "revision": revision,
        **parsed,
    }


def collect(
    cases_path: str | Path, out_dir: str | Path, runner: Any,
    *, seed: int = 42, arms: tuple[str, ...] = ARMS,
    second_sources: tuple[str, ...] = ("p_first", "v_only"),
    followups: tuple[str, ...] = FOLLOWUPS,
    prompt_voice: str = "third_person",
) -> dict[str, int]:
    cases = read_jsonl(cases_path)
    validate_cases(cases)
    if any(arm not in ARMS for arm in arms) or any(arm not in arms or arm not in ("p_first", "v_first", "v_only") for arm in second_sources):
        raise ValueError("invalid first or second arm selection")
    if any(condition not in FOLLOWUPS for condition in followups):
        raise ValueError("invalid followup selection")
    if prompt_voice not in PROMPT_VOICES:
        raise ValueError("invalid prompt voice")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(Path(cases_path).read_bytes()).hexdigest()
    manifest = {
        "cases_sha256": source_hash, "model": runner.model_name,
        "revision": runner.revision, "max_new_tokens": runner.max_new_tokens,
        "seed": seed, "arms": list(arms), "second_sources": list(second_sources),
        "followups": list(followups), "activation_convention": "HF Qwen hidden_states; index 21 = block 20 output",
    }
    if prompt_voice != "third_person":
        manifest["prompt_voice"] = prompt_voice
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing run manifest differs; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records_path = out / "records.jsonl"
    existing = {row["key"]: row for row in read_jsonl(records_path)} if records_path.exists() else {}
    completed = skipped = 0

    def run_one(case: dict[str, Any], round_no: int, arm: str, condition: str | None, messages: list[dict[str, str]]) -> dict[str, Any]:
        nonlocal completed
        key = _key(case["case_id"], round_no, arm, condition)
        if key in existing:
            record = existing[key]
            if not (out / record["activation_file"]).exists():
                raise ValueError(f"record {key} exists but activation file is missing")
            return record
        response, states = runner.predict(messages)
        rel_path = f"activations/{hashlib.sha256(key.encode()).hexdigest()}.npy"
        _save_array(out / rel_path, states)
        record = _record(case, round_no, arm, condition, messages, response, rel_path, seed, runner.model_name, runner.revision)
        append_jsonl(records_path, record)
        existing[key] = record
        completed += 1
        return record

    for case in cases:
        first: dict[str, dict[str, Any]] = {}
        for arm in arms:
            first[arm] = run_one(case, 1, arm, None, first_messages(case, arm, prompt_voice))
        for arm in second_sources:
            origin = first[arm]
            if origin.get("verdict") is None:
                skipped += len(followups)
                continue
            for condition in followups:
                run_one(case, 2, arm, condition, second_messages(origin, condition, prompt_voice))
    completion = {
        "total_records": len(existing), "skipped_followups": skipped,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
    }
    temp = out / "completion.json.tmp"
    temp.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(out / "completion.json")
    return {"new_records": completed, "total_records": len(existing), "skipped_followups": skipped}
