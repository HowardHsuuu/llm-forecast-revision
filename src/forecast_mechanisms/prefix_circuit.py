"""Replay the local answer-prefix cache interventions for Qwen2.5-7B.

A block-20 state replacement acts while the last punctuation token is
processed. Cache restorations and head-output replacements act one step
later, immediately before the first number is generated. The five visible
prefix tokens are checked on every replay.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .attention_route import PREFIX_IDS
from .model import ForecastRunner

LAYERS = (20,)
PUNCT_STEP = 4
NUMBER_STEP = 5
KV_BLOCK = 21
KV_GROUP = 2
HEAD = 20
LATE_HEAD = 26
BLOCK = 26
N_HEADS = 28
HEAD_DIM = 128
LATE_BLOCKS = tuple(range(22, 28))
CONDITIONS = ("base", "value21", "key_value21", "later22_27",
              "key_value21_later22_27")

def orthogonal(delta: np.ndarray, key: str) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid state difference")
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    draw = rng.normal(size=delta.shape).astype(np.float32)
    draw -= float(np.dot(draw, delta) / np.dot(delta, delta)) * delta
    return (draw * (norm / float(np.linalg.norm(draw)))).astype(np.float32)


def prefix_states(runner: ForecastRunner, messages: list[dict]) -> dict[int, np.ndarray]:
    torch = runner.torch
    prompt_ids = runner._ids(messages)
    counts = {block: 0 for block in LAYERS}
    observed: dict[int, np.ndarray] = {}
    handles = []
    for block in LAYERS:
        def make_hook(which):
            def hook(_module, _inputs, output):
                step = counts[which]
                counts[which] += 1
                if step == PUNCT_STEP:
                    hidden = output[0] if isinstance(output, tuple) else output
                    observed[which] = hidden[0, -1, :].float().cpu().numpy()
                return output
            return hook
        handles.append(runner.model.model.layers[block].register_forward_hook(make_hook(block)))
    try:
        with torch.inference_mode():
            generation = runner.model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=PUNCT_STEP + 1, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
            )
    finally:
        for handle in handles:
            handle.remove()
    tokens = generation[0, prompt_ids.shape[-1]:].tolist()
    if tuple(tokens) != PREFIX_IDS or set(observed) != set(LAYERS):
        raise ValueError("natural answer prefix or state capture differs")
    if any(count != PUNCT_STEP + 1 for count in counts.values()):
        raise ValueError("unexpected natural hook counts")
    return observed


def variant(
    runner: ForecastRunner, messages: list[dict], natural: dict,
    block: int, source_state: np.ndarray, target_state: np.ndarray,
    source_value: np.ndarray, recipient_token: int, donor_token: int,
    *, rescue_value: bool,
) -> dict:
    torch = runner.torch
    prompt_ids = runner._ids(messages)
    prompt_len = prompt_ids.shape[-1]
    punctuation_position = prompt_len + 3
    counts = {"layer": 0, "v": 0, "cache": 0, "heads": 0, "logits": 0}
    captured: dict = {}

    def layer_hook(_module, _inputs, output):
        step = counts["layer"]
        counts["layer"] += 1
        if step != PUNCT_STEP:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        observed = hidden[0, -1, :]
        error = float(torch.linalg.vector_norm(observed.float() - torch.as_tensor(
            source_state, dtype=torch.float32, device=hidden.device)))
        if error > 1e-3:
            raise ValueError(f"natural block-{block} punctuation state changed: {error}")
        patched = hidden.clone()
        patched[0, -1, :] = torch.as_tensor(target_state, dtype=hidden.dtype, device=hidden.device)
        captured["state_patched"] = True
        return (patched, *output[1:]) if isinstance(output, tuple) else patched

    def value_hook(_module, _inputs, output):
        step = counts["v"]
        counts["v"] += 1
        if step == PUNCT_STEP:
            captured["punctuation_value"] = output[0, -1, :].float().cpu().numpy().reshape(4, 128)[KV_GROUP]
        return output

    def cache_hook(_module, _inputs, kwargs):
        step = counts["cache"]
        counts["cache"] += 1
        if step != NUMBER_STEP or not rescue_value:
            return None
        cache = kwargs.get("past_key_values")
        if cache is None or len(cache.layers) <= KV_BLOCK:
            raise ValueError("block-21 cache absent")
        layer = cache.layers[KV_BLOCK]
        if not layer.is_initialized or punctuation_position >= layer.values.shape[-2]:
            raise ValueError("punctuation cache absent")
        values = layer.values.clone()
        values[:, KV_GROUP, punctuation_position, :] = torch.as_tensor(
            source_value, dtype=values.dtype, device=values.device)
        layer.values = values
        captured["value_restored"] = True
        return None

    def heads_hook(_module, inputs):
        step = counts["heads"]
        counts["heads"] += 1
        if step != NUMBER_STEP:
            return None
        hidden = inputs[0]
        baseline = torch.as_tensor(natural["heads"], dtype=hidden.dtype, device=hidden.device)
        patched = hidden.clone()
        for head in range(28):
            if head != HEAD:
                patched[0, -1, head * 128:(head + 1) * 128] = baseline[head]
        captured["head_isolated"] = True
        return (patched, *inputs[1:])

    def logits_hook(_module, _inputs, output):
        step = counts["logits"]
        counts["logits"] += 1
        if step == NUMBER_STEP:
            captured["raw_logits"] = output[0, -1, :].float().cpu().numpy()
        return output

    attention = runner.model.model.layers[KV_BLOCK].self_attn
    handles = [
        runner.model.model.layers[block].register_forward_hook(layer_hook),
        attention.v_proj.register_forward_hook(value_hook),
        attention.register_forward_pre_hook(cache_hook, with_kwargs=True),
        attention.o_proj.register_forward_pre_hook(heads_hook),
        runner.model.lm_head.register_forward_hook(logits_hook),
    ]
    try:
        with torch.inference_mode():
            generated = runner.model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=NUMBER_STEP + 1, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    tokens = generated.sequences[0, prompt_len:].tolist()
    if (not captured.get("state_patched") or not captured.get("head_isolated") or
            "raw_logits" not in captured or "punctuation_value" not in captured or
            rescue_value != bool(captured.get("value_restored")) or
            any(c != NUMBER_STEP + 1 for c in counts.values())):
        raise ValueError("intervention hook timing or capture failed")
    if tuple(tokens[:5]) != PREFIX_IDS:
        return {"status": "prefix_changed", "first_five_tokens": tokens[:5]}
    scores = generated.scores[NUMBER_STEP][0].float().cpu().numpy()
    margin = float(scores[recipient_token] - scores[donor_token])
    raw_margin = float(captured["raw_logits"][recipient_token]
                       - captured["raw_logits"][donor_token])
    value = captured["punctuation_value"]
    return {
        "status": "valid", "first_token": tokens[NUMBER_STEP],
        "donor_first_token_match": tokens[NUMBER_STEP] == donor_token,
        "processed_margin": margin, "raw_margin": raw_margin,
        "punctuation_value_distance_to_natural": float(np.linalg.norm(value - source_value)),
    }


def capture_natural_caches(runner: ForecastRunner, messages: list[dict]) -> dict[int, dict[str, np.ndarray]]:
    torch = runner.torch
    prompt_ids = runner._ids(messages)
    position = prompt_ids.shape[-1] + 3
    counts = {b: 0 for b in (KV_BLOCK, *LATE_BLOCKS)}
    captured: dict[int, dict[str, np.ndarray]] = {}
    handles = []
    for block in counts:
        def make_hook(which):
            def hook(_module, _inputs, kwargs):
                step = counts[which]
                counts[which] += 1
                if step != NUMBER_STEP:
                    return None
                cache = kwargs.get("past_key_values")
                if cache is None or not cache.layers[which].is_initialized:
                    raise ValueError("natural cache absent")
                layer = cache.layers[which]
                captured[which] = {
                    "key": layer.keys[0, :, position, :].float().cpu().numpy(),
                    "value": layer.values[0, :, position, :].float().cpu().numpy(),
                }
                return None
            return hook
        handles.append(runner.model.model.layers[block].self_attn.register_forward_pre_hook(
            make_hook(block), with_kwargs=True))
    try:
        with torch.inference_mode():
            generated = runner.model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=NUMBER_STEP + 1, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
            )
    finally:
        for handle in handles:
            handle.remove()
    tokens = generated[0, prompt_ids.shape[-1]:].tolist()
    if tuple(tokens[:5]) != PREFIX_IDS or set(captured) != set(counts) or any(
            c != NUMBER_STEP + 1 for c in counts.values()):
        raise ValueError("natural cache capture or prefix mismatch")
    return captured


def run_condition(
    runner: ForecastRunner, messages: list[dict], natural: dict,
    source_state: np.ndarray, donor_state: np.ndarray,
    caches: dict[int, dict[str, np.ndarray]],
    recipient_token: int, donor_token: int, condition: str,
    *, late_blocks: tuple[int, ...] = LATE_BLOCKS,
) -> dict:
    if len(set(late_blocks)) != len(late_blocks) or not set(late_blocks).issubset(LATE_BLOCKS):
        raise ValueError("late-block selection outside frozen set")
    if condition not in CONDITIONS:
        raise ValueError("unknown rescue condition")
    torch = runner.torch
    prompt_ids = runner._ids(messages)
    prompt_len = prompt_ids.shape[-1]
    position = prompt_len + 3
    counts = {"state": 0, "head": 0, "logits": 0}
    counts.update({f"cache{block}": 0 for block in (KV_BLOCK, *late_blocks)})
    captured: dict = {}
    restore_key21 = condition in ("key_value21", "key_value21_later22_27")
    restore_value21 = condition in ("value21", "key_value21", "key_value21_later22_27")
    restore_later = condition in ("later22_27", "key_value21_later22_27")

    def state_hook(_module, _inputs, output):
        step = counts["state"]
        counts["state"] += 1
        if step != PUNCT_STEP:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        error = float(torch.linalg.vector_norm(hidden[0, -1, :].float() - torch.as_tensor(
            source_state, dtype=torch.float32, device=hidden.device)))
        if error > 1e-3:
            raise ValueError("source punctuation state changed")
        changed = hidden.clone()
        changed[0, -1, :] = torch.as_tensor(donor_state, dtype=hidden.dtype, device=hidden.device)
        captured["punctuation_patched"] = True
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    def cache_hook(which):
        key = f"cache{which}"
        def hook(_module, _inputs, kwargs):
            step = counts[key]
            counts[key] += 1
            if step != NUMBER_STEP:
                return None
            if not (restore_key21 or restore_value21) and which == KV_BLOCK:
                return None
            if which in late_blocks and not restore_later:
                return None
            cache = kwargs.get("past_key_values")
            if cache is None or not cache.layers[which].is_initialized:
                raise ValueError("intervention cache absent")
            layer = cache.layers[which]
            if which == KV_BLOCK:
                if restore_key21:
                    keys = layer.keys.clone()
                    keys[:, KV_GROUP, position, :] = torch.as_tensor(
                        caches[which]["key"][KV_GROUP], dtype=keys.dtype, device=keys.device)
                    layer.keys = keys
                    captured["key21_restored"] = True
                if restore_value21:
                    values = layer.values.clone()
                    values[:, KV_GROUP, position, :] = torch.as_tensor(
                        caches[which]["value"][KV_GROUP], dtype=values.dtype, device=values.device)
                    layer.values = values
                    captured["value21_restored"] = True
            else:
                keys, values = layer.keys.clone(), layer.values.clone()
                keys[:, :, position, :] = torch.as_tensor(caches[which]["key"],
                                                           dtype=keys.dtype, device=keys.device)
                values[:, :, position, :] = torch.as_tensor(caches[which]["value"],
                                                             dtype=values.dtype, device=values.device)
                layer.keys, layer.values = keys, values
                captured.setdefault("later_restored", []).append(which)
            return None
        return hook

    def head_hook(_module, inputs):
        step = counts["head"]
        counts["head"] += 1
        if step != NUMBER_STEP:
            return None
        hidden = inputs[0]
        baseline = torch.as_tensor(natural["heads"], dtype=hidden.dtype, device=hidden.device)
        changed = hidden.clone()
        for head in range(28):
            if head != 20:
                changed[0, -1, head * 128:(head + 1) * 128] = baseline[head]
        captured["head_isolated"] = True
        return (changed, *inputs[1:])

    def logits_hook(_module, _inputs, output):
        step = counts["logits"]
        counts["logits"] += 1
        if step == NUMBER_STEP:
            captured["raw_logits"] = output[0, -1, :].float().cpu().numpy()
        return output

    handles = [runner.model.model.layers[20].register_forward_hook(state_hook)]
    for block in (KV_BLOCK, *late_blocks):
        handles.append(runner.model.model.layers[block].self_attn.register_forward_pre_hook(
            cache_hook(block), with_kwargs=True))
    handles.append(runner.model.model.layers[KV_BLOCK].self_attn.o_proj.register_forward_pre_hook(head_hook))
    handles.append(runner.model.lm_head.register_forward_hook(logits_hook))
    try:
        with torch.inference_mode():
            generated = runner.model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=NUMBER_STEP + 1, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if (not captured.get("punctuation_patched") or not captured.get("head_isolated") or
            "raw_logits" not in captured or any(v != NUMBER_STEP + 1 for v in counts.values()) or
            bool(captured.get("key21_restored")) != restore_key21 or
            bool(captured.get("value21_restored")) != restore_value21 or
            set(captured.get("later_restored", [])) != (set(late_blocks) if restore_later else set())):
        raise ValueError("rescue hook timing or components differ")
    tokens = generated.sequences[0, prompt_len:].tolist()
    if tuple(tokens[:5]) != PREFIX_IDS:
        return {"status": "prefix_changed"}
    scores = generated.scores[NUMBER_STEP][0].float().cpu().numpy()
    return {"status": "valid", "first_token": tokens[NUMBER_STEP],
            "donor_first_token_match": tokens[NUMBER_STEP] == donor_token,
            "processed_margin": float(scores[recipient_token] - scores[donor_token]),
            "raw_margin": float(captured["raw_logits"][recipient_token]
                                - captured["raw_logits"][donor_token])}


def selective_rescue(runner, messages, natural, source_state, donor_state,
                     caches, recipient_token, donor_token, selected):
    """Restore block-21 group 2 plus the specified later punctuation caches."""
    return run_condition(runner, messages, natural, source_state, donor_state,
                         caches, recipient_token, donor_token,
                         "key_value21_later22_27", late_blocks=selected)


def run_with_heads(
    runner: ForecastRunner, messages: list[dict], natural: dict,
    source_state: np.ndarray, donor_state: np.ndarray, caches: dict,
    recipient_token: int, donor_token: int, *,
    rescue_block26: bool, replacement: dict[int, np.ndarray] | None = None,
) -> tuple[dict, np.ndarray]:
    torch = runner.torch
    counts = 0
    observed = None

    def hook(_module, inputs):
        nonlocal counts, observed
        step = counts
        counts += 1
        if step != NUMBER_STEP:
            return None
        hidden = inputs[0]
        observed = hidden[0, -1, :].float().cpu().numpy().reshape(N_HEADS, HEAD_DIM)
        if replacement is None:
            return None
        changed = hidden.clone()
        for head, vector in replacement.items():
            if not 0 <= head < N_HEADS or np.asarray(vector).shape != (HEAD_DIM,):
                raise ValueError("invalid block-26 head replacement")
            changed[0, -1, head * HEAD_DIM:(head + 1) * HEAD_DIM] = torch.as_tensor(
                vector, dtype=hidden.dtype, device=hidden.device)
        return (changed, *inputs[1:])

    handle = runner.model.model.layers[26].self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        result = selective_rescue(
            runner, messages, natural, source_state, donor_state, caches,
            recipient_token, donor_token, (26,) if rescue_block26 else (),
        )
    finally:
        handle.remove()
    if counts != NUMBER_STEP + 1 or observed is None or result["status"] != "valid":
        raise ValueError("block-26 head hook or visible prefix failed")
    return result, observed


def exact(a: dict, b: dict) -> bool:
    return (a["status"] == b["status"] == "valid" and
            a["first_token"] == b["first_token"] and
            a["processed_margin"] == b["processed_margin"] and
            a["raw_margin"] == b["raw_margin"])


def capture_base(
    runner: ForecastRunner, messages: list[dict], natural: dict,
    source_state: np.ndarray, donor_state: np.ndarray, caches: dict,
    recipient_token: int, donor_token: int,
) -> tuple[dict, dict[str, np.ndarray], np.ndarray]:
    prompt_len = runner._ids(messages).shape[-1]
    position = prompt_len + 3
    counts = {"cache": 0, "heads": 0}
    captured: dict = {}

    def cache_hook(_module, _inputs, kwargs):
        step = counts["cache"]
        counts["cache"] += 1
        if step == NUMBER_STEP:
            cache = kwargs.get("past_key_values")
            layer = cache.layers[BLOCK]
            captured["cache"] = {
                "key": layer.keys[0, :, position, :].float().cpu().numpy(),
                "value": layer.values[0, :, position, :].float().cpu().numpy(),
            }
        return None

    def head_hook(_module, inputs):
        step = counts["heads"]
        counts["heads"] += 1
        if step == NUMBER_STEP:
            captured["heads"] = inputs[0][0, -1, :].float().cpu().numpy().reshape(28, 128)
        return None

    attention = runner.model.model.layers[BLOCK].self_attn
    handles = [attention.register_forward_pre_hook(cache_hook, with_kwargs=True),
               attention.o_proj.register_forward_pre_hook(head_hook)]
    try:
        result = selective_rescue(
            runner, messages, natural, source_state, donor_state, caches,
            recipient_token, donor_token, (),
        )
    finally:
        for handle in handles:
            handle.remove()
    if result["status"] != "valid" or any(c != NUMBER_STEP + 1 for c in counts.values()) or set(captured) != {"cache", "heads"}:
        raise ValueError("edited base capture or visible prefix failed")
    return result, captured["cache"], captured["heads"]


def run_isolated(
    runner: ForecastRunner, messages: list[dict], natural: dict,
    source_state: np.ndarray, donor_state: np.ndarray, caches: dict,
    recipient_token: int, donor_token: int, base_heads: np.ndarray,
) -> dict:
    torch = runner.torch
    count = 0

    def hook(_module, inputs):
        nonlocal count
        step = count
        count += 1
        if step != NUMBER_STEP:
            return None
        hidden = inputs[0]
        changed = hidden.clone()
        source = torch.as_tensor(base_heads, dtype=hidden.dtype, device=hidden.device)
        for head in range(28):
            if head != LATE_HEAD:
                changed[0, -1, head * 128:(head + 1) * 128] = source[head]
        return (changed, *inputs[1:])

    handle = runner.model.model.layers[BLOCK].self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        result = selective_rescue(
            runner, messages, natural, source_state, donor_state, caches,
            recipient_token, donor_token, (BLOCK,),
        )
    finally:
        handle.remove()
    if count != NUMBER_STEP + 1 or result["status"] != "valid":
        raise ValueError("head-26 isolation or prefix failed")
    return result
