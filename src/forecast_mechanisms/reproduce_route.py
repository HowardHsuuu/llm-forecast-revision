"""Reproduce the equal-length feedback route from a completed first-person run."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .attention_route import (
    PREFIX_IDS, CachePatch, capture_step, evaluate_first_token,
    generate_with_state_replacement,
)
from .core import append_jsonl, parse_response, read_jsonl, verify_complete_run
from .model import DEFAULT_MODEL, DEFAULT_REVISION, ForecastRunner

CLAUSES_BY_GAP = {
    9: "I still don't know what to expect.",
    11: "I am still unsure what to expect from the decision.",
    12: "I still don't know what to expect from the decision.",
}
FOLLOWUP_MARKER = "\nPlease reply with exactly these lines and no explanation:\n"
REFERENCE_CASES_SHA256 = "bd61ddf20c0c54fa80f9be78709a2f7389cfd678bf5c543d9e4ef030c0d15ca0"
REFERENCE_PHASES_SHA256 = "691da5562d703f41e5c7291e9e5505dc19451e79520158db2731a90b10957393"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pairs(run_dir: Path, trace_dir: Path, phase_file: Path, phase: str) -> list[dict]:
    if _sha256(phase_file) != REFERENCE_PHASES_SHA256:
        raise ValueError("route case selection differs from the frozen protocol")
    completion = verify_complete_run(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if (manifest.get("model"), manifest.get("revision"), manifest.get("prompt_voice"),
            manifest.get("cases_sha256"), manifest.get("seed")) != (
        DEFAULT_MODEL, DEFAULT_REVISION, "first_person", REFERENCE_CASES_SHA256, 42
    ):
        raise ValueError("collection must use the pinned Qwen first-person protocol")
    trace_manifest = json.loads((trace_dir / "manifest.json").read_text())
    if trace_manifest.get("source_records_sha256") != completion["records_sha256"]:
        raise ValueError("decision traces do not match the collection")
    phases = json.loads(phase_file.read_text())
    if phases.get("protocol") != "qwen2.5-7b-first-person-equal-length-v1":
        raise ValueError("unknown route phase protocol")
    ids = phases["phases"][phase]
    if len(ids) != len(set(ids)) or len(ids) != {"pilot": 12, "replication": 12, "extension": 39}[phase]:
        raise ValueError("phase case IDs differ from the frozen selection")
    selected = set(ids)
    source = {
        (r["case_id"], r["condition"]): r
        for r in read_jsonl(run_dir / "records.jsonl")
        if r["case_id"] in selected and r["split"] == "test"
        and r["round"] == 2 and r["arm"] == "p_first"
        and r["condition"] in ("neutral", "opposite")
    }
    traces = {
        (r["case_id"], r["condition"]): r
        for r in read_jsonl(trace_dir / "traces.jsonl")
        if r["case_id"] in selected and r["split"] == "test"
        and r["round"] == 2 and r["arm"] == "p_first"
        and r["condition"] in ("neutral", "opposite")
    }
    if len(source) != len(ids) * 2 or len(traces) != len(ids) * 2:
        raise ValueError("neutral/opposite collection or trace rows are missing")
    pairs = []
    for case_id in ids:
        neutral, opposite = source[case_id, "neutral"], source[case_id, "opposite"]
        neutral_trace, opposite_trace = traces[case_id, "neutral"], traces[case_id, "opposite"]
        if neutral_trace["key"] != neutral["key"] or opposite_trace["key"] != opposite["key"]:
            raise ValueError(f"decision traces do not match source rows: {case_id}")
        if not neutral["parse_ok"] or not opposite["parse_ok"] or neutral["domain"] != opposite["domain"]:
            raise ValueError(f"unparsed or mismatched source pair: {case_id}")
        if neutral["messages"][:-1] != opposite["messages"][:-1]:
            raise ValueError(f"pair differs before the feedback turn: {case_id}")
        if (neutral_trace["number_token_index"] != opposite_trace["number_token_index"]
                or neutral_trace["number_token_index"] != 5):
            raise ValueError(f"number-token alignment differs: {case_id}")
        if tuple(opposite_trace["generated_token_ids"][:5]) != PREFIX_IDS:
            raise ValueError(f"unexpected answer prefix: {case_id}")
        pairs.append({"case_id": case_id, "domain": neutral["domain"],
                      "neutral": neutral, "opposite": opposite,
                      "neutral_trace": neutral_trace, "opposite_trace": opposite_trace,
                      "number_index": 5})
    return pairs


def _edited_neutral(runner: ForecastRunner, pair: dict) -> tuple[list[dict], int]:
    neutral = pair["neutral"]["messages"]
    opposite = pair["opposite"]["messages"]
    gap = int(runner._ids(opposite).shape[-1] - runner._ids(neutral).shape[-1])
    if gap not in CLAUSES_BY_GAP:
        raise ValueError(f"unexpected feedback prompt-length gap for {pair['case_id']}: {gap}")
    result = [dict(item) for item in neutral]
    if result[-1]["role"] != "user" or result[-1]["content"].count(FOLLOWUP_MARKER) != 1:
        raise ValueError("neutral followup wording differs from protocol")
    result[-1]["content"] = result[-1]["content"].replace(
        FOLLOWUP_MARKER, " " + CLAUSES_BY_GAP[gap] + FOLLOWUP_MARKER)
    if runner._ids(result).shape[-1] != runner._ids(opposite).shape[-1]:
        raise ValueError("neutral/pushback prompts do not have equal token lengths")
    return result, gap


def _full_generation(runner: ForecastRunner, messages: list[dict]) -> tuple[str, list[int]]:
    torch = runner.torch
    prompt_ids = runner._ids(messages)
    with torch.inference_mode():
        generated = runner.model.generate(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=runner.max_new_tokens, do_sample=False,
            pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                          else runner.tokenizer.eos_token_id),
        )
    tokens = generated[0, prompt_ids.shape[-1]:].tolist()
    return runner.tokenizer.decode(tokens, skip_special_tokens=True).strip(), tokens


def _orthogonal(delta: np.ndarray, key: str) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    draw = np.random.default_rng(seed).normal(size=delta.shape).astype(np.float32)
    draw -= float(np.dot(draw, delta) / np.dot(delta, delta)) * delta
    return draw * (float(np.linalg.norm(delta)) / float(np.linalg.norm(draw)))


def _state_orthogonal(delta: np.ndarray, key: str) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    unit = delta / norm
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    draw = np.random.default_rng(seed).normal(size=delta.shape).astype(np.float32)
    draw -= np.dot(draw, unit) * unit
    return draw / np.linalg.norm(draw) * norm


def _value_patch(source: dict, group: int, position: int, vector: np.ndarray) -> CachePatch:
    values = source["cache_v"].copy()
    values[0, group, position, :] += vector
    return CachePatch(source["cache_k"], values, (group,), ((position, position + 1),), "value")


def _check_run(capture: dict, prefix: list[int], source_state: np.ndarray | None = None) -> None:
    if capture["tokens"][:len(prefix)] != prefix:
        raise ValueError("intervention changed the generated prefix")
    if source_state is not None and capture.get("source_state_error") != 0:
        raise ValueError("source block-20 state differs at the hook")


def reproduce_route(
    run_dir: str | Path,
    trace_dir: str | Path,
    phase_file: str | Path,
    output_dir: str | Path,
    phase: str,
    *,
    only_case: str | None = None,
) -> dict:
    """Run the state, head, cache, position, and specificity interventions."""
    if phase not in ("pilot", "replication", "extension"):
        raise ValueError("unknown phase")
    run_dir, trace_dir, phase_file, output_dir = map(Path, (run_dir, trace_dir, phase_file, output_dir))
    pairs = _load_pairs(run_dir, trace_dir, phase_file, phase)
    if only_case is not None and only_case not in {p["case_id"] for p in pairs}:
        raise ValueError("case ID is outside the selected phase")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "protocol": "qwen2.5-7b-route-v1", "phase": phase, "only_case": only_case,
        "run_records_sha256": _sha256(run_dir / "records.jsonl"),
        "trace_records_sha256": _sha256(trace_dir / "traces.jsonl"),
        "phase_file_sha256": _sha256(phase_file),
        "route_code_sha256": _sha256(Path(__file__)),
        "attention_code_sha256": _sha256(Path(__file__).with_name("attention_route.py")),
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
        "decoder_repetition_penalty": 1.05,
    }
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != settings:
            raise ValueError("route output manifest differs from requested inputs")
    else:
        manifest.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    output = output_dir / "records.jsonl"
    previous = read_jsonl(output) if output.exists() else []
    done = {r["case_id"] for r in previous}
    if len(done) != len(previous):
        raise ValueError("duplicate route output case")
    expected_done = {only_case} if only_case else {p["case_id"] for p in pairs}
    if not done <= expected_done:
        raise ValueError("route output contains cases outside the requested selection")
    if done == expected_done:
        return {"phase": phase, "rows": len(done), "status": "already_complete"}
    runner = ForecastRunner(max_new_tokens=json.loads((run_dir / "manifest.json").read_text())["max_new_tokens"])
    if runner.model.generation_config.repetition_penalty != 1.05:
        raise ValueError("checkpoint decoder repetition penalty differs from protocol")

    # Establish natural responses, then collect donor vectors before any patch result.
    prepared = {}
    for pair in pairs:
        case_id = pair["case_id"]
        neutral_messages, gap = _edited_neutral(runner, pair)
        target_messages = pair["opposite"]["messages"]
        target_text, target_tokens = _full_generation(runner, target_messages)
        if target_text != pair["opposite"]["response"] or target_tokens != pair["opposite_trace"]["generated_token_ids"]:
            raise ValueError(f"pushback baseline replay differs: {case_id}")
        source_text, source_tokens = _full_generation(runner, neutral_messages)
        source_parsed = parse_response(source_text, "p_first")
        item = {"pair": pair, "neutral_messages": neutral_messages,
                "target_messages": target_messages, "gap": gap,
                "source_text": source_text, "source_tokens": source_tokens,
                "target_tokens": target_tokens, "source_parsed": source_parsed}
        if not source_parsed["parse_ok"]:
            item["status"] = "neutral_parse_failed"
        elif source_tokens[:5] != target_tokens[:5]:
            item["status"] = "prefix_mismatch"
        elif source_parsed["p_accept"] == pair["opposite"]["p_accept"]:
            item["status"] = "unchanged_percentage"
        elif source_tokens[5] == target_tokens[5]:
            item["status"] = "same_first_number_token"
        else:
            item["status"] = "valid"
            source = capture_step(runner, neutral_messages, 5, capture_cache=True)
            target = capture_step(runner, target_messages, 5, capture_cache=True, capture_later=True)
            if source["tokens"] != source_tokens[:6] or target["tokens"] != target_tokens[:6]:
                raise ValueError(f"first-number capture differs from full replay: {case_id}")
            item.update({"source": source, "target": target})
        prepared[case_id] = item

    donor_ids = {}
    for domain in sorted({p["domain"] for p in pairs}):
        ids = sorted(case_id for case_id, item in prepared.items()
                     if item["status"] == "valid" and item["pair"]["domain"] == domain)
        if len(ids) < 2:
            raise ValueError("same-domain donor unavailable")
        donor_ids.update({case_id: ids[(i + 1) % len(ids)] for i, case_id in enumerate(ids)})
    deltas = {}
    for case_id, item in prepared.items():
        if item["status"] == "valid":
            prompt_len = int(runner._ids(item["neutral_messages"]).shape[-1])
            position = prompt_len + 3
            deltas[case_id] = (item["target"]["cache_v"][0, 2, position, :]
                               - item["source"]["cache_v"][0, 2, position, :]).astype(np.float32)
            if np.linalg.norm(deltas[case_id]) == 0:
                raise ValueError("zero natural cached-value difference")

    for pair in pairs:
        case_id = pair["case_id"]
        if case_id in done or (only_case is not None and only_case != case_id):
            continue
        item = prepared[case_id]
        row = {"case_id": case_id, "domain": pair["domain"], "phase": phase,
               "status": item["status"]}
        if item["status"] != "valid":
            append_jsonl(output, row)
            done.add(case_id)
            continue
        source, target = item["source"], item["target"]
        neutral_messages, target_messages = item["neutral_messages"], item["target_messages"]
        prefix = source["tokens"][:5]
        source_token, target_token = source["tokens"][5], target["tokens"][5]
        state = capture_step(runner, neutral_messages, 5, replace_state=target["block20"],
                             expect_state=source["block20"], capture_cache=True)
        _check_run(state, prefix, source["block20"])
        qkv_errors = {name: float(np.max(np.abs(state["qkv"][name] - target["qkv"][name])))
                      for name in ("q", "k", "v")}
        if any(qkv_errors.values()):
            raise ValueError("current-token QKV differs after block-20 state replacement")
        forward_text, forward_tokens, forward_error = generate_with_state_replacement(
            runner, neutral_messages, 5, source["block20"], target["block20"])
        reverse_text, reverse_tokens, reverse_error = generate_with_state_replacement(
            runner, target_messages, 5, target["block20"], source["block20"])
        if forward_error or reverse_error or forward_tokens[:5] != prefix or reverse_tokens[:5] != prefix:
            raise ValueError("full-answer state replacement validation failed")
        state_orth = {}
        natural_delta = target["block20"] - source["block20"]
        for direction, messages, baseline in (
            ("forward", neutral_messages, source["block20"]),
            ("reverse", target_messages, target["block20"]),
        ):
            orth = _state_orthogonal(natural_delta, f"length-match:{case_id}:{direction}")
            text, tokens, error = generate_with_state_replacement(
                runner, messages, 5, baseline, baseline + orth)
            if error or tokens[:5] != prefix:
                raise ValueError("orthogonal state control validation failed")
            parsed = parse_response(text, "p_first")
            if not parsed["parse_ok"]:
                raise ValueError("orthogonal state control did not yield a structured answer")
            state_orth[direction] = {"p_accept": parsed["p_accept"],
                                     "first_token_target_match": tokens[5] == (target_token if direction == "forward" else source_token)}

        target_heads = target["heads"]
        head20 = capture_step(runner, neutral_messages, 5, replace_state=target["block20"],
                              expect_state=source["block20"], head_replacements={20: target_heads[20]})
        all_heads = capture_step(runner, neutral_messages, 5, replace_state=target["block20"],
                                 expect_state=source["block20"],
                                 head_replacements={h: target_heads[h] for h in range(28)})
        for result in (head20, all_heads):
            _check_run(result, prefix, source["block20"])
        if np.max(np.abs(all_heads["attention"][21] - target["attention"][21])) != 0:
            raise ValueError("all-head patch does not reconstruct target block-21 attention")
        single_heads = []
        for head in range(28):
            result = head20 if head == 20 else capture_step(
                runner, neutral_messages, 5, replace_state=target["block20"],
                expect_state=source["block20"], head_replacements={head: target_heads[head]})
            _check_run(result, prefix, source["block20"])
            single_heads.append({"head": head, **evaluate_first_token(result, source_token, target_token)})
        without_head20 = None
        if all_heads["tokens"][5] == target_token:
            without_head20 = capture_step(
                runner, neutral_messages, 5, replace_state=target["block20"],
                expect_state=source["block20"],
                head_replacements={h: target_heads[h] for h in range(28) if h != 20})
            _check_run(without_head20, prefix, source["block20"])

        source_ids = runner._ids(neutral_messages)[0].tolist()
        target_ids = runner._ids(target_messages)[0].tolist()
        first_diff = next((i for i, (a, b) in enumerate(zip(source_ids, target_ids)) if a != b), None)
        prompt_len, past_len = len(source_ids), source["cache_v"].shape[-2]
        if first_diff is None or not first_diff < prompt_len < past_len or past_len != prompt_len + 4:
            raise ValueError("unexpected prompt/cache position alignment")
        if np.max(np.abs(source["cache_v"][:, :, :first_diff] - target["cache_v"][:, :, :first_diff])) != 0:
            raise ValueError("cache differs before first prompt-token difference")
        cache_conditions = {
            "all_past": CachePatch(target["cache_k"], target["cache_v"], tuple(range(4)), ((0, past_len),)),
            "user_turn": CachePatch(target["cache_k"], target["cache_v"], tuple(range(4)), ((first_diff, prompt_len),)),
            "generated_prefix": CachePatch(target["cache_k"], target["cache_v"], tuple(range(4)), ((prompt_len, past_len),)),
        }
        cache_results = {}
        for name, patch in cache_conditions.items():
            result = capture_step(runner, neutral_messages, 5, replace_state=target["block20"],
                                  expect_state=source["block20"], cache_patch=patch)
            _check_run(result, prefix, source["block20"])
            if name == "all_past" and np.max(np.abs(result["attention"][21] - target["attention"][21])) != 0:
                raise ValueError("all-past cache patch does not reconstruct target attention")
            cache_results[name] = evaluate_first_token(result, source_token, target_token)

        positions = {}
        for offset, name in enumerate(("P", "(", "accept", "):")):
            position = prompt_len + offset
            forward = capture_step(
                runner, neutral_messages, 5, replace_state=target["block20"],
                expect_state=source["block20"],
                cache_patch=CachePatch(target["cache_k"], target["cache_v"], (2,), ((position, position + 1),), "value"),
                isolate_head=(20, state["heads"]))
            reverse = capture_step(
                runner, target_messages, 5, expect_state=target["block20"],
                cache_patch=CachePatch(source["cache_k"], source["cache_v"], (2,), ((position, position + 1),), "value"),
                isolate_head=(20, target["heads"]))
            _check_run(forward, prefix, source["block20"])
            _check_run(reverse, prefix, target["block20"])
            positions[name] = {"forward": evaluate_first_token(forward, source_token, target_token),
                               "reverse": evaluate_first_token(reverse, source_token, target_token)}
        delta = deltas[case_id]
        donor_id = donor_ids[case_id]
        donor = deltas[donor_id]
        donor = donor * (float(np.linalg.norm(delta)) / float(np.linalg.norm(donor)))
        orth = _orthogonal(delta, f"head20-v-orth:{case_id}")
        specificity = {}
        position = prompt_len + 3
        for name, vector in (("same_case", delta), ("donor", donor), ("orthogonal", orth)):
            forward = capture_step(
                runner, neutral_messages, 5, replace_state=target["block20"],
                expect_state=source["block20"],
                cache_patch=_value_patch(source, 2, position, vector),
                isolate_head=(20, state["heads"]))
            reverse = capture_step(
                runner, target_messages, 5, expect_state=target["block20"],
                cache_patch=_value_patch(target, 2, position, -vector),
                isolate_head=(20, target["heads"]))
            _check_run(forward, prefix, source["block20"])
            _check_run(reverse, prefix, target["block20"])
            specificity[name] = {"forward": evaluate_first_token(forward, source_token, target_token),
                                 "reverse": evaluate_first_token(reverse, source_token, target_token)}
        for direction in ("forward", "reverse"):
            if (abs(specificity["same_case"][direction]["processed_margin"]
                    - positions["):"][direction]["processed_margin"]) > 1e-6):
                raise ValueError("same-case vector does not reproduce direct position patch")

        later = {}
        for through in range(21, 28):
            attention = {block: target["attention"][block] for block in range(21, through + 1)}
            result = capture_step(runner, neutral_messages, 5, replace_state=target["block20"],
                                  expect_state=source["block20"], later_attention=attention)
            _check_run(result, prefix, source["block20"])
            later[str(through)] = {**evaluate_first_token(result, source_token, target_token),
                                   "max_raw_logit_error": float(np.max(np.abs(result["raw_logits"] - target["raw_logits"]))) }
        if later["27"]["max_raw_logit_error"] != 0:
            raise ValueError("later-attention endpoint does not reconstruct target raw logits")

        forward_parsed = parse_response(forward_text, "p_first")
        reverse_parsed = parse_response(reverse_text, "p_first")
        if not forward_parsed["parse_ok"] or not reverse_parsed["parse_ok"]:
            raise ValueError("state replacement did not yield a structured answer")
        row.update({"number_index": 5, "gap_filled": item["gap"],
                    "neutral_p": item["source_parsed"]["p_accept"],
                    "pushback_p": pair["opposite"]["p_accept"],
                    "source_token": source_token, "target_token": target_token,
                    "qkv_max_errors": qkv_errors,
                    "natural_source": evaluate_first_token(source, source_token, target_token),
                    "natural_target": evaluate_first_token(target, source_token, target_token),
                    "state_forward": {"p_accept": forward_parsed["p_accept"],
                                      **evaluate_first_token(state, source_token, target_token)},
                    "state_reverse": {"p_accept": reverse_parsed["p_accept"],
                                      "first_token_neutral_match": reverse_tokens[5] == source_token},
                    "state_orthogonal": state_orth,
                    "head20": evaluate_first_token(head20, source_token, target_token),
                    "all_heads": evaluate_first_token(all_heads, source_token, target_token),
                    "single_heads": single_heads,
                    "without_head20": (evaluate_first_token(without_head20, source_token, target_token)
                                       if without_head20 is not None else None),
                    "cache_segments": cache_results,
                    "prefix_positions": positions,
                    "donor_case_id": donor_id,
                    "donor_cosine": float(np.dot(delta, donor) / (np.linalg.norm(delta) * np.linalg.norm(donor))),
                    "orthogonal_cosine": float(np.dot(delta, orth) / (np.linalg.norm(delta) * np.linalg.norm(orth))),
                    "specificity": specificity,
                    "later_attention": later})
        append_jsonl(output, row)
        done.add(case_id)
    counts = Counter(row["status"] for row in read_jsonl(output))
    return {"phase": phase, "rows": len(done), "status_counts": dict(counts)}
