"""Replay completed answers and save generation-time decision states.

The trace uses generated token IDs and the hidden states returned by the same
autoregressive generation, rather than re-tokenizing or recomputing prefixes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .core import append_jsonl, read_jsonl, verify_complete_run
from .model import EXPECTED_BLOCKS, EXPECTED_HIDDEN, NLA_LAYER, ForecastRunner


_NUMBER = re.compile(r"P\(accept\):\s*[0-9]", re.I)
_VERDICT = re.compile(r"Verdict:\s*[AR]", re.I)


def decision_token_indices(decoded_prefixes: list[str]) -> dict[str, int]:
    """Find first token containing a numeric or verdict value, zero-indexed."""
    result: dict[str, int] = {}
    for index, prefix in enumerate(decoded_prefixes):
        if "number" not in result and _NUMBER.search(prefix):
            result["number"] = index
        if "verdict" not in result and _VERDICT.search(prefix):
            result["verdict"] = index
    if set(result) != {"number", "verdict"}:
        raise ValueError("generated response lacks a numeric or verdict value token")
    return result


def _selected(row: dict[str, Any]) -> bool:
    if not row["parse_ok"]:
        return False
    if row["round"] == 1:
        return row["arm"] in ("p_first", "v_first")
    return row["round"] == 2 and row["arm"] == "p_first" and row.get("condition") in ("neutral", "opposite")


def collect_decision_traces(
    run_dir: str | Path, output_dir: str | Path, runner: ForecastRunner,
    *, max_records: int | None = None,
) -> dict[str, Any]:
    run, out = Path(run_dir), Path(output_dir)
    completion = verify_complete_run(run)
    source_manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (runner.model_name, runner.revision, runner.max_new_tokens) != (
        source_manifest["model"], source_manifest["revision"], source_manifest["max_new_tokens"]
    ):
        raise ValueError("trace runner differs from completed collection")
    settings = {
        "version": 2, "source_records_sha256": completion["records_sha256"],
        "source_manifest": source_manifest, "positions": ["pre_number", "pre_verdict"],
        "neighbor_offsets": [-2, -1, 0], "layer_convention": "index 21 = block 20 output",
        "measurement": "generation_return_hidden_states_and_scores",
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("trace settings changed; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    records_path = out / "traces.jsonl"
    existing_rows = read_jsonl(records_path) if records_path.exists() else []
    existing = {row["key"] for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("duplicate trace keys")
    source = [row for row in read_jsonl(run / "records.jsonl") if _selected(row)]
    eligible = len(source)
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        source = source[:max_records]
    torch = runner.torch
    for row in source:
        if row["key"] in existing:
            trace_file = out / next(r["trace_file"] for r in existing_rows if r["key"] == row["key"])
            if not trace_file.is_file():
                raise ValueError(f"missing saved trace for {row['key']}")
            continue
        prompt_ids = runner._ids(row["messages"])
        with torch.inference_mode():
            generation = runner.model.generate(
                input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=runner.max_new_tokens, do_sample=False,
                pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                              else runner.tokenizer.eos_token_id),
                return_dict_in_generate=True, output_hidden_states=True, output_scores=True,
            )
        generated_ids = generation.sequences[0, prompt_ids.shape[-1]:]
        steps = generation.hidden_states
        scores = generation.scores
        if len(steps) != len(scores) or len(steps) != len(generated_ids):
            raise ValueError("generation did not return one hidden-state/score set per token")
        if any(len(step) != EXPECTED_BLOCKS + 1 for step in steps):
            raise ValueError("unexpected hidden-state count")
        if any(int(score[0].argmax()) != int(generated_ids[i])
               for i, score in enumerate(scores)):
            raise ValueError("generation scores disagree with emitted token IDs")
        replay = runner.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if replay != row["response"]:
            raise ValueError(f"baseline replay does not match collection for {row['key']}")
        prefixes = [runner.tokenizer.decode(generated_ids[:i + 1], skip_special_tokens=True)
                    for i in range(len(generated_ids))]
        targets = decision_token_indices(prefixes)
        prompt_len = prompt_ids.shape[-1]
        if min(targets.values()) < 2:
            raise ValueError("trace lacks enough preceding tokens")
        verdict_states = steps[targets["verdict"]]
        accept_ids = runner.tokenizer.encode(" ACCEPT", add_special_tokens=False)
        reject_ids = runner.tokenizer.encode(" RE", add_special_tokens=False)
        if len(accept_ids) != 1 or len(reject_ids) != 1:
            raise ValueError("verdict first-token candidates changed under this tokenizer")
        candidate_ids = [accept_ids[0], reject_ids[0]]
        with torch.inference_mode():
            candidate_weights = runner.model.lm_head.weight[candidate_ids]
            verdict_lens_margin = []
            for index, hidden in enumerate(verdict_states):
                vector = hidden[0, -1, :].unsqueeze(0)
                if index < EXPECTED_BLOCKS:
                    vector = runner.model.model.norm(vector)
                logits = torch.matmul(vector, candidate_weights.T)
                verdict_lens_margin.append(float((logits[0, 0] - logits[0, 1]).float().cpu()))
            true_margin = float((scores[targets["verdict"]][0, candidate_ids[0]]
                                 - scores[targets["verdict"]][0, candidate_ids[1]]).float().cpu())
        lens_numeric_difference = float(verdict_lens_margin[-1] - true_margin)
        # BF16 small-matrix and full-vocabulary kernels can round differently;
        # retain that difference for audit and use the actual generation score.
        verdict_lens_margin[-1] = true_margin

        def layer_matrix(hidden_states) -> np.ndarray:
            return torch.stack([h[0, -1, :].float().cpu() for h in hidden_states]).numpy()

        prompt_matrix = layer_matrix(steps[0])
        collected = np.load(run / row["activation_file"], allow_pickle=False)
        relative_error = float(np.linalg.norm(prompt_matrix - collected) / np.linalg.norm(collected))
        if not np.isfinite(relative_error) or relative_error > 0.01:
            raise ValueError(f"trace disagrees with collected prompt state for {row['key']}: {relative_error}")
        number_states = layer_matrix(steps[targets["number"]])
        verdict_matrix = layer_matrix(verdict_states)
        neighbors = np.stack([
            np.stack([steps[targets[name] + offset][NLA_LAYER + 1]
                      [0, -1, :].float().cpu().numpy()
                      for offset in (-2, -1, 0)])
            for name in ("number", "verdict")
        ])
        if number_states.shape != (EXPECTED_BLOCKS + 1, EXPECTED_HIDDEN):
            raise ValueError("invalid decision trace shape")
        if not (np.isfinite(number_states).all() and np.isfinite(verdict_matrix).all()
                and np.isfinite(neighbors).all()):
            raise ValueError("nonfinite decision trace")
        filename = hashlib.sha256(row["key"].encode()).hexdigest()[:24] + ".npz"
        trace_path = out / "states" / filename
        trace_path.parent.mkdir(exist_ok=True)
        np.savez(trace_path, pre_number=number_states, pre_verdict=verdict_matrix,
                 block20_neighbors=neighbors,
                 verdict_logit_lens_margin=np.asarray(verdict_lens_margin, dtype=np.float32))
        append_jsonl(records_path, {
            "key": row["key"], "case_id": row["case_id"], "domain": row["domain"],
            "split": row["split"], "round": row["round"], "arm": row["arm"],
            "condition": row.get("condition"), "p_accept": row["p_accept"],
            "verdict": row["verdict"], "trace_file": str(trace_path.relative_to(out)),
            "number_token_index": targets["number"], "verdict_token_index": targets["verdict"],
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_ids.tolist(),
            "verdict_candidate_token_ids": candidate_ids,
            "actual_verdict_first_token_id": int(generated_ids[targets["verdict"]]),
            "prompt_state_relative_error": relative_error,
            "final_lens_numeric_difference": lens_numeric_difference,
            "next_token_replay_top1": {name: True for name in ("number", "verdict")},
        })
        existing.add(row["key"])
    return {"eligible": eligible, "attempted": len(source), "saved": len(existing),
            "new": len(existing) - len(existing_rows)}
