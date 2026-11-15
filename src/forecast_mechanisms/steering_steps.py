"""Single-position checks for the frozen generation-time steering vector."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .attention_route import PREFIX_IDS
from .core import parse_response
from .model import ForecastRunner


VERDICT_PREFIX = re.compile(r"\nVerdict:\s*$")
NUMBER_STEP = len(PREFIX_IDS)


def verdict_step(tokenizer, tokens: list[int]) -> int:
    candidates = [j for j in range(1, len(tokens)) if VERDICT_PREFIX.search(
        tokenizer.decode(tokens[:j], skip_special_tokens=True))]
    if len(candidates) != 1:
        raise ValueError("could not identify a unique first verdict token")
    return candidates[0]


def natural_verdict_state(runner: ForecastRunner, messages: list[dict],
                          expected_response: str, block: int) -> tuple[int, np.ndarray, str]:
    torch = runner.torch
    ids = runner._ids(messages)
    with torch.inference_mode():
        output = runner.model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=96, do_sample=False,
            pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                          else runner.tokenizer.eos_token_id),
            return_dict_in_generate=True, output_hidden_states=True,
        )
    tokens = output.sequences[0, ids.shape[-1]:].tolist()
    response = runner.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if response != expected_response:
        raise ValueError("verdict-state replay differs from saved answer")
    step = verdict_step(runner.tokenizer, tokens)
    state = output.hidden_states[step][block + 1][0, -1, :].float().cpu().numpy()
    if state.shape != (3584,) or not np.isfinite(state).all():
        raise ValueError("invalid verdict-step state")
    digest = hashlib.sha256(json.dumps(tokens[:step]).encode()).hexdigest()
    return step, state, digest


def number_once(runner: ForecastRunner, messages: list[dict], block: int,
                delta: np.ndarray) -> dict:
    torch = runner.torch
    ids = runner._ids(messages)
    prompt_len = ids.shape[-1]
    count = 0
    applied = False
    norm = None
    perturbation = torch.as_tensor(delta, dtype=torch.bfloat16,
                                   device=runner.model.device)

    def hook(_module, _inputs, output):
        nonlocal count, applied, norm
        current = count
        count += 1
        if current != NUMBER_STEP:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != 1:
            raise ValueError("first-number step was not one generated token")
        actual = hidden[0, -1, :].float()
        norm = float(torch.linalg.vector_norm(actual))
        changed = hidden.clone()
        changed[0, -1, :] += perturbation
        applied = True
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = runner.model.model.layers[block].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            generated = runner.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=96, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
            )
    finally:
        handle.remove()
    if not applied or norm is None:
        raise ValueError("first-number edit did not execute")
    tokens = generated[0, prompt_len:].tolist()
    if len(tokens) <= NUMBER_STEP or tuple(tokens[:NUMBER_STEP]) != PREFIX_IDS:
        raise ValueError("fixed pre-number answer prefix changed")
    response = runner.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    parsed = parse_response(response, "p_first")
    return {"response": response, "parse_ok": parsed["parse_ok"],
            "p_accept": parsed["p_accept"], "verdict": parsed["verdict"],
            "first_numeric_token": tokens[NUMBER_STEP],
            "pre_number_prefix_ok": True, "source_state_norm": norm,
            "relative_dose": float(np.linalg.norm(delta) / norm)}


def verdict_once(runner: ForecastRunner, messages: list[dict], block: int,
                 step: int, source: np.ndarray, delta: np.ndarray,
                 prefix_digest: str, accept_id: int, reject_id: int) -> dict:
    torch = runner.torch
    ids = runner._ids(messages)
    prompt_len = ids.shape[-1]
    count = 0
    applied = False
    source_error = None

    def hook(_module, _inputs, output):
        nonlocal count, applied, source_error
        current = count
        count += 1
        if current != step:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        actual = hidden[0, -1, :].float()
        expected = torch.as_tensor(source, dtype=torch.float32, device=actual.device)
        source_error = float(torch.linalg.vector_norm(actual - expected))
        if source_error > 1e-3:
            raise ValueError("verdict-step source state changed")
        applied = True
        if not np.any(delta):
            return output
        changed = hidden.clone()
        changed[0, -1, :] = (actual + torch.as_tensor(delta, dtype=torch.float32,
                                                      device=actual.device)).to(hidden.dtype)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = runner.model.model.layers[block].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = runner.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=96, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        handle.remove()
    if not applied or source_error is None or len(output.scores) <= step:
        raise ValueError("verdict-step edit did not execute")
    tokens = output.sequences[0, prompt_len:].tolist()
    digest = hashlib.sha256(json.dumps(tokens[:step]).encode()).hexdigest()
    if digest != prefix_digest:
        raise ValueError("pre-verdict answer prefix changed")
    response = runner.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    parsed = parse_response(response, "p_first")
    scores = output.scores[step][0].float()
    return {"response": response, "parse_ok": parsed["parse_ok"],
            "p_accept": parsed["p_accept"], "verdict": parsed["verdict"],
            "accept_minus_reject_first_logit": float(scores[accept_id] - scores[reject_id]),
            "first_verdict_token": tokens[step],
            "source_state_error": source_error,
            "prefix_sha256": digest}
