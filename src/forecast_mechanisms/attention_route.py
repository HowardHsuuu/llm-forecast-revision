"""Decision-step capture and patching for the pinned Qwen2.5-7B checkpoint.

The hook contract is intentionally narrow: block 20's current-token residual,
block 21's past KV cache and query-head outputs, and attention outputs through
block 27. Calls fail when the model layout or autoregressive step differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .model import EXPECTED_HIDDEN, ForecastRunner

STATE_BLOCK = 20
ROUTE_BLOCK = 21
FINAL_BLOCK = 27
PREFIX_IDS = (47, 7, 10330, 1648, 220)  # P(accept): and the following space


@dataclass(frozen=True)
class CachePatch:
    keys: np.ndarray
    values: np.ndarray
    groups: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    kind: Literal["key", "value", "both"] = "both"


def _validate_model(runner: ForecastRunner) -> tuple[int, int, int]:
    cfg = runner.model.config
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_heads
    if (n_heads, n_kv, head_dim, cfg.hidden_size, cfg.num_hidden_layers) != (28, 4, 128, EXPECTED_HIDDEN, 28):
        raise ValueError("unexpected Qwen attention layout")
    return n_heads, n_kv, head_dim


def capture_step(
    runner: ForecastRunner,
    messages: list[dict[str, str]],
    token_index: int,
    *,
    replace_state: np.ndarray | None = None,
    expect_state: np.ndarray | None = None,
    cache_patch: CachePatch | None = None,
    capture_cache: bool = False,
    head_replacements: Mapping[int, np.ndarray] | None = None,
    isolate_head: tuple[int, np.ndarray] | None = None,
    capture_later: bool = False,
    later_attention: Mapping[int, np.ndarray] | None = None,
) -> dict:
    """Run through one generated token and return its internal state and logits.

    ``token_index=0`` is the prompt's final-token forward pass. Cached entries
    are the *past* tokens before the requested step; current Q/K/V are separate.
    All patches act only at the requested step and leave earlier output intact.
    """
    n_heads, n_kv, head_dim = _validate_model(runner)
    if not 0 <= token_index < runner.max_new_tokens:
        raise ValueError("token_index outside generation cap")
    if head_replacements and isolate_head:
        raise ValueError("choose head replacement or head isolation")
    if later_attention and not set(later_attention).issubset(range(ROUTE_BLOCK, FINAL_BLOCK + 1)):
        raise ValueError("later attention block outside 21..27")
    if cache_patch is not None:
        if cache_patch.kind not in ("key", "value", "both"):
            raise ValueError("invalid cache patch kind")
        if not cache_patch.groups or len(set(cache_patch.groups)) != len(cache_patch.groups):
            raise ValueError("invalid KV-group selection")
        if any(not 0 <= g < n_kv for g in cache_patch.groups):
            raise ValueError("KV-group index outside model")
        if not cache_patch.spans:
            raise ValueError("cache spans missing")
        capture_cache = True
    for vector in (replace_state, expect_state):
        if vector is not None and (np.asarray(vector).shape != (EXPECTED_HIDDEN,) or not np.isfinite(vector).all()):
            raise ValueError("invalid block-20 residual state")
    if head_replacements:
        if any(not 0 <= h < n_heads or np.asarray(v).shape != (head_dim,)
               for h, v in head_replacements.items()):
            raise ValueError("invalid query-head replacement")
    if isolate_head is not None:
        keep, baseline = isolate_head
        if not 0 <= keep < n_heads or np.asarray(baseline).shape != (n_heads, head_dim):
            raise ValueError("invalid query-head isolation")

    torch = runner.torch
    prompt_ids = runner._ids(messages)
    counts = {key: 0 for key in ("state", "q", "k", "v", "heads", "attn21", "logits")}
    if capture_cache:
        counts["cache"] = 0
    if capture_later or later_attention:
        counts.update({f"attn{block}": 0 for block in range(22, FINAL_BLOCK + 1)})
    captured: dict = {"qkv": {}, "attention": {}}
    handles = []

    def state_hook(_module, _inputs, output):
        step = counts["state"]
        counts["state"] += 1
        if step != token_index:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        observed = hidden[0, -1, :]
        captured["block20"] = observed.float().cpu().numpy()
        if expect_state is not None:
            expected = torch.as_tensor(expect_state, dtype=hidden.dtype, device=hidden.device)
            captured["source_state_error"] = float(torch.linalg.vector_norm(observed.float() - expected.float()))
        if replace_state is None:
            return output
        changed = hidden.clone()
        changed[0, -1, :] = torch.as_tensor(replace_state, dtype=hidden.dtype, device=hidden.device)
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def qkv_hook(name):
        def hook(_module, _inputs, output):
            step = counts[name]
            counts[name] += 1
            if step == token_index:
                captured["qkv"][name] = output[0, -1, :].float().cpu().numpy()
            return output
        return hook

    def cache_pre(_module, _inputs, kwargs):
        step = counts["cache"]
        counts["cache"] += 1
        if step != token_index:
            return None
        cache = kwargs.get("past_key_values")
        if cache is None or len(cache.layers) <= ROUTE_BLOCK:
            raise ValueError("past KV cache missing")
        layer = cache.layers[ROUTE_BLOCK]
        if not layer.is_initialized:
            raise ValueError("block-21 cache is uninitialized")
        keys, values = layer.keys, layer.values
        if keys.ndim != 4 or values.shape != keys.shape or keys.shape[1] != n_kv or keys.shape[-1] != head_dim:
            raise ValueError("unexpected KV-cache shape")
        captured["cache_k"] = keys.float().cpu().numpy()
        captured["cache_v"] = values.float().cpu().numpy()
        if cache_patch is None:
            return None
        shape = tuple(keys.shape)
        if cache_patch.keys.shape != shape or cache_patch.values.shape != shape:
            raise ValueError("target KV cache has wrong shape")
        if any(not 0 <= start < end <= shape[-2] for start, end in cache_patch.spans):
            raise ValueError("cache span outside past sequence")
        new_keys, new_values = keys.clone(), values.clone()
        target_keys = torch.as_tensor(cache_patch.keys, dtype=keys.dtype, device=keys.device)
        target_values = torch.as_tensor(cache_patch.values, dtype=values.dtype, device=values.device)
        for group in cache_patch.groups:
            for start, end in cache_patch.spans:
                if cache_patch.kind in ("key", "both"):
                    new_keys[:, group, start:end, :] = target_keys[:, group, start:end, :]
                if cache_patch.kind in ("value", "both"):
                    new_values[:, group, start:end, :] = target_values[:, group, start:end, :]
        layer.keys, layer.values = new_keys, new_values
        return None

    def heads_pre(_module, inputs):
        step = counts["heads"]
        counts["heads"] += 1
        if step != token_index:
            return None
        hidden = inputs[0]
        captured["heads"] = hidden[0, -1, :].float().cpu().numpy().reshape(n_heads, head_dim)
        if not head_replacements and isolate_head is None:
            return None
        changed = hidden.clone()
        if head_replacements:
            for head, vector in head_replacements.items():
                changed[0, -1, head * head_dim:(head + 1) * head_dim] = torch.as_tensor(
                    vector, dtype=hidden.dtype, device=hidden.device)
        if isolate_head is not None:
            keep, baseline = isolate_head
            baseline_t = torch.as_tensor(baseline, dtype=hidden.dtype, device=hidden.device)
            for head in range(n_heads):
                if head != keep:
                    changed[0, -1, head * head_dim:(head + 1) * head_dim] = baseline_t[head]
        return (changed, *inputs[1:])

    def attention_hook(block):
        key = "attn21" if block == ROUTE_BLOCK else f"attn{block}"
        def hook(_module, _inputs, output):
            step = counts[key]
            counts[key] += 1
            if step != token_index:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            captured["attention"][block] = hidden[0, -1, :].float().cpu().numpy()
            if later_attention is None or block not in later_attention:
                return output
            vector = np.asarray(later_attention[block])
            if vector.shape != (EXPECTED_HIDDEN,):
                raise ValueError("later attention vector has wrong shape")
            changed = hidden.clone()
            changed[0, -1, :] = torch.as_tensor(vector, dtype=hidden.dtype, device=hidden.device)
            return (changed, *output[1:]) if isinstance(output, tuple) else changed
        return hook

    def logits_hook(_module, _inputs, output):
        step = counts["logits"]
        counts["logits"] += 1
        if step == token_index:
            captured["raw_logits"] = output[0, -1, :].float().cpu().numpy()
        return output

    layer = runner.model.model.layers[ROUTE_BLOCK]
    handles.append(runner.model.model.layers[STATE_BLOCK].register_forward_hook(state_hook))
    for name, module in (("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
                         ("v", layer.self_attn.v_proj)):
        handles.append(module.register_forward_hook(qkv_hook(name)))
    if capture_cache:
        handles.append(layer.self_attn.register_forward_pre_hook(cache_pre, with_kwargs=True))
    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(heads_pre))
    handles.append(layer.self_attn.register_forward_hook(attention_hook(ROUTE_BLOCK)))
    if capture_later or later_attention:
        for block in range(22, FINAL_BLOCK + 1):
            handles.append(runner.model.model.layers[block].self_attn.register_forward_hook(attention_hook(block)))
    handles.append(runner.model.lm_head.register_forward_hook(logits_hook))
    try:
        with torch.inference_mode():
            generated = runner.model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=token_index + 1, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    tokens = generated.sequences[0, prompt_ids.shape[-1]:].tolist()
    if len(tokens) != token_index + 1 or len(generated.scores) != token_index + 1:
        raise ValueError("generation ended before the requested token")
    if any(count != token_index + 1 for count in counts.values()):
        raise ValueError(f"unexpected model forward counts: {counts}")
    required = {"block20", "heads", "raw_logits"}
    if not required <= set(captured) or set(captured["qkv"]) != {"q", "k", "v"}:
        raise ValueError("capture hook missing at decision step")
    if capture_cache and not {"cache_k", "cache_v"} <= set(captured):
        raise ValueError("KV cache capture missing")
    captured["tokens"] = tokens
    captured["scores"] = generated.scores[token_index][0].float().cpu().numpy()
    return captured


def evaluate_first_token(capture: dict, source_token: int, target_token: int) -> dict:
    return {
        "source_match": capture["tokens"][-1] == source_token,
        "target_match": capture["tokens"][-1] == target_token,
        "processed_margin": float(capture["scores"][target_token] - capture["scores"][source_token]),
        "raw_margin": float(capture["raw_logits"][target_token] - capture["raw_logits"][source_token]),
    }


def generate_with_state_replacement(
    runner: ForecastRunner, messages: list[dict[str, str]], token_index: int,
    source_state: np.ndarray, target_state: np.ndarray,
    *, head_replacements: Mapping[int, np.ndarray] | None = None,
) -> tuple[str, list[int], float]:
    """Generate the complete answer after a one-step block-20/head replacement."""
    n_heads, _, head_dim = _validate_model(runner)
    if head_replacements and any(
        not 0 <= head < n_heads or np.asarray(vector).shape != (head_dim,)
        for head, vector in head_replacements.items()
    ):
        raise ValueError("invalid query-head replacement")
    torch = runner.torch
    prompt_ids = runner._ids(messages)
    step = 0
    applied = False
    error = None

    def hook(_module, _inputs, output):
        nonlocal step, applied, error
        current = step
        step += 1
        if current != token_index:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        observed = hidden[0, -1, :]
        source = torch.as_tensor(source_state, dtype=hidden.dtype, device=hidden.device)
        error = float(torch.linalg.vector_norm(observed.float() - source.float()))
        changed = hidden.clone()
        changed[0, -1, :] = torch.as_tensor(target_state, dtype=hidden.dtype, device=hidden.device)
        applied = True
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    head_step = 0
    head_applied = False

    def head_hook(_module, inputs):
        nonlocal head_step, head_applied
        current = head_step
        head_step += 1
        if current != token_index or not head_replacements:
            return None
        hidden = inputs[0]
        changed = hidden.clone()
        for head, vector in head_replacements.items():
            changed[0, -1, head * head_dim:(head + 1) * head_dim] = torch.as_tensor(
                vector, dtype=hidden.dtype, device=hidden.device
            )
        head_applied = True
        return (changed, *inputs[1:])

    handles = [runner.model.model.layers[STATE_BLOCK].register_forward_hook(hook)]
    if head_replacements:
        handles.append(runner.model.model.layers[ROUTE_BLOCK].self_attn.o_proj.register_forward_pre_hook(head_hook))
    try:
        with torch.inference_mode():
            generated = runner.model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=runner.max_new_tokens, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
            )
    finally:
        for handle in handles:
            handle.remove()
    if not applied or error is None:
        raise ValueError("state replacement hook did not run")
    if head_replacements and not head_applied:
        raise ValueError("query-head replacement hook did not run")
    tokens = generated[0, prompt_ids.shape[-1]:].tolist()
    response = runner.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return response, tokens, error


def generate_with_prefix_value_intervention(
    runner: ForecastRunner,
    messages: list[dict[str, str]],
    token_index: int,
    *,
    kv_group: int,
    cache_position: int,
    value_delta: np.ndarray,
    keep_head: int,
    baseline_heads: np.ndarray,
) -> tuple[str, list[int], np.ndarray]:
    """Continue a complete answer after a value edit at one numeric decision step.

    The edited cached value is restored before the next decoding step. Other
    query heads are held at their natural decision-step outputs, so subsequent
    digits inherit only the changed first numeric token and its consequences.
    """
    n_heads, n_kv, head_dim = _validate_model(runner)
    vector = np.asarray(value_delta, dtype=np.float32)
    heads = np.asarray(baseline_heads, dtype=np.float32)
    if (not 0 <= kv_group < n_kv or not 0 <= keep_head < n_heads
            or vector.shape != (head_dim,) or heads.shape != (n_heads, head_dim)
            or not np.isfinite(vector).all() or not np.isfinite(heads).all()):
        raise ValueError("invalid prefix-value intervention")
    if not 0 <= token_index < runner.max_new_tokens:
        raise ValueError("token index outside generation cap")
    torch = runner.torch
    prompt_ids = runner._ids(messages)
    cache_step = head_step = 0
    applied = restored = head_applied = False
    original_value = None

    def cache_hook(_module, _inputs, kwargs):
        nonlocal cache_step, applied, restored, original_value
        current = cache_step
        cache_step += 1
        if current not in (token_index, token_index + 1):
            return None
        cache = kwargs.get("past_key_values")
        if cache is None or len(cache.layers) <= ROUTE_BLOCK:
            raise ValueError("past KV cache missing")
        layer = cache.layers[ROUTE_BLOCK]
        if not layer.is_initialized or not 0 <= cache_position < layer.values.shape[-2]:
            raise ValueError("prefix position outside initialized cache")
        values = layer.values.clone()
        if current == token_index:
            original_value = values[:, kv_group, cache_position, :].clone()
            values[:, kv_group, cache_position, :] = (
                original_value.float()
                + torch.as_tensor(vector, dtype=torch.float32, device=values.device)
            ).to(values.dtype)
            applied = True
        else:
            if original_value is None:
                raise ValueError("cannot restore an unapplied value edit")
            values[:, kv_group, cache_position, :] = original_value
            restored = True
        layer.values = values
        return None

    def heads_hook(_module, inputs):
        nonlocal head_step, head_applied
        current = head_step
        head_step += 1
        if current != token_index:
            return None
        hidden = inputs[0]
        changed = hidden.clone()
        baseline = torch.as_tensor(heads, dtype=hidden.dtype, device=hidden.device)
        for head in range(n_heads):
            if head != keep_head:
                changed[0, -1, head * head_dim:(head + 1) * head_dim] = baseline[head]
        head_applied = True
        return (changed, *inputs[1:])

    attention = runner.model.model.layers[ROUTE_BLOCK].self_attn
    handles = [
        attention.register_forward_pre_hook(cache_hook, with_kwargs=True),
        attention.o_proj.register_forward_pre_hook(heads_hook),
    ]
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
        for handle in handles:
            handle.remove()
    tokens = generated.sequences[0, prompt_ids.shape[-1]:].tolist()
    if not applied or not head_applied or len(tokens) <= token_index:
        raise ValueError("prefix-value intervention did not reach the numeric step")
    if len(tokens) > token_index + 1 and not restored:
        raise ValueError("edited cache value was not restored after the numeric step")
    response = runner.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    scores = generated.scores[token_index][0].float().cpu().numpy()
    return response, tokens, scores
