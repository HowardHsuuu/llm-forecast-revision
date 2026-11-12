"""Qwen2.5 generation, pre-answer states, and single-position intervention."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
NLA_LAYER = 20
EXPECTED_HIDDEN = 3584
EXPECTED_BLOCKS = 28


class ForecastRunner:
    def __init__(self, model_name: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION, max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for the pinned 7B target model")
        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        cfg = self.model.config
        if cfg.model_type != "qwen2" or cfg.hidden_size != EXPECTED_HIDDEN or cfg.num_hidden_layers != EXPECTED_BLOCKS:
            raise ValueError("target checkpoint is incompatible with the pinned Qwen2.5 NLA layer/dimension contract")
        if getattr(cfg, "_name_or_path", None) is None:
            raise ValueError("missing model identity")

    def _ids(self, messages: list[dict[str, str]]):
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("expected one unpadded prompt")
        return ids.to(self.model.device)

    def preanswer_states(self, messages: list[dict[str, str]]) -> np.ndarray:
        torch = self.torch
        ids = self._ids(messages)
        with torch.inference_mode():
            output = self.model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                output_hidden_states=True, use_cache=False,
            )
        states = output.hidden_states
        if len(states) != EXPECTED_BLOCKS + 1:
            raise ValueError(f"unexpected number of hidden states: {len(states)}")
        # HF Qwen hidden_states[21] is raw block-20 output; index 28 is final norm.
        matrix = torch.stack([s[0, -1, :].float().cpu() for s in states]).numpy()
        if matrix.shape != (EXPECTED_BLOCKS + 1, EXPECTED_HIDDEN) or not np.isfinite(matrix).all():
            raise ValueError("invalid pre-answer state matrix")
        return matrix

    def generate(self, messages: list[dict[str, str]], intervention: tuple[int, np.ndarray] | None = None) -> str:
        torch = self.torch
        ids = self._ids(messages)
        handle = None
        applied = False
        if intervention is not None:
            block_index, vector = intervention
            if not 0 <= block_index < EXPECTED_BLOCKS:
                raise ValueError("intervention block must be a transformer block index")
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (EXPECTED_HIDDEN,) or not np.isfinite(vector).all():
                raise ValueError("invalid intervention vector")
            delta = torch.as_tensor(vector, dtype=torch.bfloat16, device=self.model.device)
            prompt_len = ids.shape[-1]

            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                nonlocal applied
                hidden = output[0] if isinstance(output, tuple) else output
                if applied or hidden.shape[1] != prompt_len:
                    return output
                changed = hidden.clone()
                changed[:, -1, :] += delta
                applied = True
                if isinstance(output, tuple):
                    return (changed, *output[1:])
                return changed

            handle = self.model.model.layers[block_index].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                )
        finally:
            if handle is not None:
                handle.remove()
        if intervention is not None and not applied:
            raise RuntimeError("intervention hook was not applied to the prompt's final token")
        return self.tokenizer.decode(generated[0, ids.shape[-1]:], skip_special_tokens=True).strip()

    def predict(self, messages: list[dict[str, str]]) -> tuple[str, np.ndarray]:
        states = self.preanswer_states(messages)
        response = self.generate(messages)
        return response, states

    def generate_at_step(
        self, messages: list[dict[str, str]], *, block_index: int,
        token_index: int, delta: np.ndarray,
    ) -> tuple[str, list[int], np.ndarray]:
        """Add a residual delta just before generated token ``token_index``.

        Step zero is the prompt's final token; later steps each consume the
        preceding generated token. The returned token IDs allow callers to
        verify an unchanged prefix; the step scores allow a positive control
        even when the discrete answer does not change.
        """
        torch = self.torch
        if not 0 <= block_index < EXPECTED_BLOCKS:
            raise ValueError("intervention block must be a transformer block index")
        if not 0 <= token_index < self.max_new_tokens:
            raise ValueError("generated token index is outside the generation cap")
        vector = np.asarray(delta, dtype=np.float32)
        if vector.shape != (EXPECTED_HIDDEN,) or not np.isfinite(vector).all():
            raise ValueError("invalid intervention vector")
        prompt_ids = self._ids(messages)
        perturbation = torch.as_tensor(vector, dtype=torch.bfloat16, device=self.model.device)
        current_step = 0
        applied = False

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            nonlocal current_step, applied
            step = current_step
            current_step += 1
            if step != token_index:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            changed = hidden.clone()
            changed[:, -1, :] += perturbation
            applied = True
            if isinstance(output, tuple):
                return (changed, *output[1:])
            return changed

        handle = self.model.model.layers[block_index].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                    max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                  else self.tokenizer.eos_token_id),
                    return_dict_in_generate=True, output_scores=True,
                )
        finally:
            handle.remove()
        if not applied:
            raise RuntimeError("intervention hook was not applied at the requested generated step")
        tokens = generated.sequences[0, prompt_ids.shape[-1]:].tolist()
        if len(generated.scores) <= token_index:
            raise RuntimeError("generation ended before the requested score step")
        step_scores = generated.scores[token_index][0].float().cpu().numpy()
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), tokens, step_scores
