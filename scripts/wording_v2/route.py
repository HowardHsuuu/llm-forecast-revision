"""Frozen condition-specific answer-prefix cache interventions on new cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from forecast_mechanisms.attention_route import (
    PREFIX_IDS, CachePatch, capture_step, generate_with_prefix_value_intervention,
)
from forecast_mechanisms.core import append_jsonl, parse_response, read_jsonl, verify_complete_run
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION, ForecastRunner


WORK = Path.cwd()
ROOT = WORK / "cross-wording-v2"
CASES = WORK / "cross-wording-v2-cases.jsonl"
CONDITIONS = ("opposite", "paraphrase_a", "paraphrase_b", "unrelated_expectation")
CASE_SHA = "8b2dc2783229e172c730e1a4dd8a137de26b4968af4348533ea0813eb80230eb"
NUMBER_INDEX = 5
BLOCK = 21
HEAD = 20
GROUP = 2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orthogonal(delta: np.ndarray, key: str) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("invalid observed value difference")
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    draw = np.random.default_rng(seed).normal(size=delta.shape).astype(np.float32)
    draw -= float(np.dot(draw, delta) / np.dot(delta, delta)) * delta
    draw_norm = float(np.linalg.norm(draw))
    if draw_norm <= 0:
        raise ValueError("orthogonal control degenerated")
    return (draw * (norm / draw_norm)).astype(np.float32)


def prefix_values(capture: dict, prompt_len: int) -> np.ndarray:
    if capture["cache_v"].shape[-2] != prompt_len + 4:
        raise ValueError("unexpected answer-prefix cache length")
    return capture["cache_v"][0, :, prompt_len + 3, :].astype(np.float32)


def cache_patch(natural: dict, prompt_len: int, deltas: dict[int, np.ndarray]) -> CachePatch:
    position = prompt_len + 3
    values = natural["cache_v"].copy()
    for group, delta in deltas.items():
        values[0, group, position, :] += delta
    return CachePatch(natural["cache_k"], values, tuple(sorted(deltas)),
                      ((position, position + 1),), "value")


def margin(capture: dict, neutral_token: int, condition_token: int, *, raw: bool = False) -> float:
    scores = capture["raw_logits"] if raw else capture["scores"]
    return float(scores[condition_token] - scores[neutral_token])


def continue_all_groups(
    runner: ForecastRunner, messages: list[dict[str, str]], prompt_len: int,
    deltas: dict[int, np.ndarray],
) -> tuple[str, list[int], np.ndarray]:
    """Apply all-group value edit at the number, then restore it one step later."""
    torch = runner.torch
    prompt_ids = runner._ids(messages)
    if prompt_ids.shape[-1] != prompt_len or set(deltas) != {0, 1, 2, 3}:
        raise ValueError("all-group continuation inputs differ")
    position = prompt_len + 3
    state = {"step": 0, "applied": False, "restored": False, "original": {}}

    def hook(_module, _inputs, kwargs):
        step = state["step"]
        state["step"] += 1
        if step not in (NUMBER_INDEX, NUMBER_INDEX + 1):
            return None
        cache = kwargs.get("past_key_values")
        if cache is None or len(cache.layers) <= BLOCK:
            raise ValueError("block-21 past cache absent")
        layer = cache.layers[BLOCK]
        if not layer.is_initialized or not 0 <= position < layer.values.shape[-2]:
            raise ValueError("punctuation position absent")
        values = layer.values.clone()
        for group, delta in deltas.items():
            if step == NUMBER_INDEX:
                state["original"][group] = values[:, group, position, :].clone()
                values[:, group, position, :] = (
                    state["original"][group].float()
                    + torch.as_tensor(delta, dtype=torch.float32, device=values.device)
                ).to(values.dtype)
            else:
                values[:, group, position, :] = state["original"][group]
        layer.values = values
        if step == NUMBER_INDEX:
            state["applied"] = True
        else:
            state["restored"] = True
        return None

    handle = runner.model.model.layers[BLOCK].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.inference_mode():
            generated = runner.model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=runner.max_new_tokens, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                                  else runner.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        handle.remove()
    tokens = generated.sequences[0, prompt_len:].tolist()
    if not state["applied"] or len(tokens) <= NUMBER_INDEX:
        raise ValueError("all-group intervention missed number")
    if len(tokens) > NUMBER_INDEX + 1 and not state["restored"]:
        raise ValueError("all-group cached value was not restored")
    response = runner.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    scores = generated.scores[NUMBER_INDEX][0].float().cpu().numpy()
    return response, tokens, scores


def intervention(
    runner: ForecastRunner, messages: list[dict[str, str]], natural: dict,
    prompt_len: int, neutral_token: int, condition_token: int,
    deltas: dict[int, np.ndarray], *, isolate: bool,
) -> dict:
    isolated = (HEAD, natural["heads"]) if isolate else None
    patched = capture_step(
        runner, messages, NUMBER_INDEX, expect_state=natural["block20"],
        cache_patch=cache_patch(natural, prompt_len, deltas), isolate_head=isolated,
    )
    if patched["source_state_error"] != 0 or patched["tokens"][:5] != natural["tokens"][:5]:
        raise ValueError("one-step intervention changed earlier state or visible prefix")
    if isolate:
        response, tokens, scores = generate_with_prefix_value_intervention(
            runner, messages, NUMBER_INDEX, kv_group=GROUP,
            cache_position=prompt_len + 3, value_delta=deltas[GROUP],
            keep_head=HEAD, baseline_heads=natural["heads"],
        )
    else:
        response, tokens, scores = continue_all_groups(runner, messages, prompt_len, deltas)
    if tokens[:6] != patched["tokens"] or abs(float(scores[condition_token] - scores[neutral_token])
                                              - margin(patched, neutral_token, condition_token)) > 1e-5:
        raise ValueError("one-step and full-continuation interventions disagree")
    parsed = parse_response(response, "p_first")
    return {
        "processed_margin_shift": margin(patched, neutral_token, condition_token)
                                  - margin(natural, neutral_token, condition_token),
        "raw_margin_shift": margin(patched, neutral_token, condition_token, raw=True)
                            - margin(natural, neutral_token, condition_token, raw=True),
        "first_token": tokens[NUMBER_INDEX],
        "neutral_first_token_match": tokens[NUMBER_INDEX] == neutral_token,
        "parse_ok": parsed["parse_ok"],
        "full_percentage": parsed["p_accept"],
        "full_verdict": parsed["verdict"],
    }


def indexed_records() -> tuple[dict, dict, dict]:
    if digest(CASES) != CASE_SHA:
        raise ValueError("frozen third-cohort cases changed")
    baseline_completion = verify_complete_run(ROOT / "baseline")
    variant_completion = json.loads((ROOT / "variants" / "completion.json").read_text())
    if digest(ROOT / "variants" / "records.jsonl") != variant_completion["records_sha256"]:
        raise ValueError("variant collection changed after completion")
    baseline = read_jsonl(ROOT / "baseline" / "records.jsonl")
    first = {r["case_id"]: r for r in baseline if r["round"] == 1 and r["arm"] == "p_first"}
    second = {(r["case_id"], r["condition"]): r for r in baseline
              if r["round"] == 2 and r["condition"] in ("neutral", "opposite")}
    second.update({(r["case_id"], r["condition"]): r
                   for r in read_jsonl(ROOT / "variants" / "records.jsonl")})
    valid_first = sum(row["parse_ok"] for row in first.values())
    if (len(first) != 160 or len(second) != valid_first * 5
            or baseline_completion["total_records"] != 160 + valid_first * 2):
        raise ValueError("third-cohort conversation matrix incomplete")
    return first, second, {row["case_id"]: row for row in read_jsonl(CASES)}


def main(max_cases: int | None) -> None:
    first, second, cases = indexed_records()
    out = ROOT / "route"
    out.mkdir(parents=True, exist_ok=True)
    import forecast_mechanisms.attention_route as route_module
    manifest = {
        "protocol": "cross-wording-prefix-route-v2", "cases_sha256": CASE_SHA,
        "baseline_records_sha256": digest(ROOT / "baseline" / "records.jsonl"),
        "variants_records_sha256": digest(ROOT / "variants" / "records.jsonl"),
        "route_script_sha256": digest(Path(__file__)),
        "attention_code_sha256": digest(Path(route_module.__file__)),
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
        "primary_metric": "condition-specific processed first-token margin shift: paired minus orthogonal",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("route manifest differs; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    records = out / "records.jsonl"
    previous = read_jsonl(records) if records.exists() else []
    done = {row["case_id"] for row in previous}
    if len(done) != len(previous):
        raise ValueError("duplicate route case")
    runner = ForecastRunner(max_new_tokens=96)
    completed = 0
    for case_id in sorted(cases):
        if case_id in done:
            continue
        case = cases[case_id]
        origin = first[case_id]
        row = {
            "case_id": case_id, "domain": case["domain"],
            "initial_verdict": origin["verdict"], "conditions": {},
        }
        if not origin["parse_ok"]:
            row["status"] = "first_round_parse_failure"
        else:
            neutral_row = second[case_id, "neutral"]
            if not neutral_row["parse_ok"]:
                row["status"] = "neutral_parse_failure"
            else:
                row["status"] = "ready"
        if row["status"] == "ready":
            neutral = capture_step(runner, neutral_row["messages"], NUMBER_INDEX, capture_cache=True)
            if tuple(neutral["tokens"][:5]) != PREFIX_IDS:
                row["status"] = "neutral_prefix_mismatch"
            else:
                row["status"] = "valid"
                neutral_len = runner._ids(neutral_row["messages"]).shape[-1]
                neutral_values = prefix_values(neutral, neutral_len)
                neutral_token = neutral["tokens"][NUMBER_INDEX]
                row["neutral_first_token"] = neutral_token
                row["neutral_percentage"] = neutral_row["p_accept"]
                row["neutral_verdict"] = neutral_row["verdict"]
                for condition in CONDITIONS:
                    feedback = second[case_id, condition]
                    result = {
                        "natural_percentage": feedback["p_accept"],
                        "natural_verdict": feedback["verdict"],
                    }
                    if not feedback["parse_ok"]:
                        result["status"] = "parse_failure"
                    else:
                        messages = feedback["messages"]
                        natural = capture_step(runner, messages, NUMBER_INDEX, capture_cache=True)
                        if natural["tokens"][:5] != neutral["tokens"][:5]:
                            result["status"] = "prefix_mismatch"
                        else:
                            condition_token = natural["tokens"][NUMBER_INDEX]
                            result["natural_first_token"] = condition_token
                            if condition_token == neutral_token:
                                result["status"] = "same_first_token"
                            else:
                                prompt_len = runner._ids(messages).shape[-1]
                                observed = neutral_values - prefix_values(natural, prompt_len)
                                if not np.isfinite(observed).all() or any(np.linalg.norm(v) == 0 for v in observed):
                                    result["status"] = "zero_or_invalid_value_difference"
                                else:
                                    result["status"] = "valid"
                                    result["natural_processed_margin"] = margin(
                                        natural, neutral_token, condition_token)
                                    result["natural_raw_margin"] = margin(
                                        natural, neutral_token, condition_token, raw=True)
                                    paired_head = {GROUP: observed[GROUP]}
                                    control_head = {GROUP: orthogonal(
                                        observed[GROUP], f"route-v2:{case_id}:{condition}:head20")}
                                    paired_all = {g: observed[g] for g in range(4)}
                                    control_all = {g: orthogonal(
                                        observed[g], f"route-v2:{case_id}:{condition}:all:{g}")
                                        for g in range(4)}
                                    result["interventions"] = {
                                        "head20_paired": intervention(
                                            runner, messages, natural, prompt_len, neutral_token,
                                            condition_token, paired_head, isolate=True),
                                        "head20_orthogonal": intervention(
                                            runner, messages, natural, prompt_len, neutral_token,
                                            condition_token, control_head, isolate=True),
                                        "all_groups_paired": intervention(
                                            runner, messages, natural, prompt_len, neutral_token,
                                            condition_token, paired_all, isolate=False),
                                        "all_groups_orthogonal": intervention(
                                            runner, messages, natural, prompt_len, neutral_token,
                                            condition_token, control_all, isolate=False),
                                    }
                    row["conditions"][condition] = result
        append_jsonl(records, row)
        done.add(case_id)
        completed += 1
        if max_cases is not None and completed >= max_cases:
            break
    if len(done) == len(cases):
        (out / "completion.json").write_text(json.dumps({
            "records_sha256": digest(records), "total_records": len(done),
        }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"completed_this_run": completed, "total_records": len(done)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    main(args.max_cases)
