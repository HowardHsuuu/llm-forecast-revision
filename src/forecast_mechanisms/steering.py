"""Generation-time residual steering with explicit token-position checks."""

from __future__ import annotations

import hashlib

import numpy as np

from .attention_route import PREFIX_IDS
from .core import parse_response
from .model import ForecastRunner


def orthogonal_control(unit: np.ndarray, case_id: str, label: str) -> np.ndarray:
    """Draw a reproducible equal-norm control orthogonal to a unit vector."""
    seed = int.from_bytes(hashlib.sha256(
        f"preanswer:{label}:{case_id}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    draw = rng.standard_normal(unit.shape)
    draw -= np.dot(draw, unit) * unit
    length = np.linalg.norm(draw)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("degenerate orthogonal control")
    return (draw / length).astype(np.float32)


def generate_with_vector(runner: ForecastRunner, messages: list[dict],
                         block: int, delta: np.ndarray) -> dict:
    """Add ``delta`` after ``block`` on each forward pass after prefill.

    The unedited prompt prefill determines the first generated token. Every
    subsequent one-token model call receives the same vector. The returned
    response and edit count make this timing verifiable for each case.
    """
    torch = runner.torch
    ids = runner._ids(messages)
    prompt_len = ids.shape[-1]
    delta = np.asarray(delta, dtype=np.float32)
    if delta.shape != (3584,) or not np.isfinite(delta).all():
        raise ValueError("invalid steering vector")
    amount = float(np.linalg.norm(delta))
    delta_t = torch.as_tensor(delta, dtype=torch.bfloat16, device=runner.model.device)
    calls = edits = 0
    norms = []

    def hook(_module, _inputs, output):
        nonlocal calls, edits
        step = calls
        calls += 1
        if step == 0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != 1:
            raise ValueError("generation step was not a single token")
        if amount == 0:
            return output
        norms.append(torch.linalg.vector_norm(hidden[0, -1, :].float()))
        changed = hidden.clone()
        changed[0, -1, :] += delta_t
        edits += 1
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = runner.model.model.layers[block].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            generated = runner.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=runner.max_new_tokens, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
            )
    finally:
        handle.remove()
    tokens = generated[0, prompt_len:].tolist()
    response = runner.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    parsed = parse_response(response, "p_first")
    prefix_ok = len(tokens) > 5 and tuple(tokens[:5]) == PREFIX_IDS
    if amount > 0 and (edits == 0 or len(norms) != edits):
        raise ValueError("nonzero steering did not edit generated tokens")
    relative = None
    if amount > 0:
        measured = torch.stack(norms).float().cpu().numpy()
        relative = float(np.mean(amount / measured))
    return {"response": response, "parse_ok": parsed["parse_ok"],
            "p_accept": parsed["p_accept"], "verdict": parsed["verdict"],
            "prefix_ok": prefix_ok,
            "first_numeric_token": tokens[5] if prefix_ok else None,
            "n_generated_tokens": len(tokens), "model_calls": calls,
            "edited_generation_steps": edits,
            "mean_relative_dose": relative}
