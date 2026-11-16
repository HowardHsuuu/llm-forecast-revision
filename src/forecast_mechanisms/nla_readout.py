"""NLA readout with checkpoint compatibility checks and matched-vector controls."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .core import ARMS, read_jsonl, verify_complete_run, write_jsonl
from .model import EXPECTED_HIDDEN, NLA_LAYER

_EXPLANATION = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.S)


def _load_official_client(module_path: str | Path):
    path = Path(module_path)
    if not path.is_file():
        raise ValueError("pass the official kitft/nla-inference nla_inference.py file")
    spec = importlib.util.spec_from_file_location("nla_inference", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nla_inference"] = module
    spec.loader.exec_module(module)
    return module


def _check_sidecar(checkpoint: Path) -> None:
    sidecar = checkpoint / "nla_meta.yaml"
    if not sidecar.is_file():
        raise ValueError("NLA actor checkpoint is missing nla_meta.yaml")
    metadata = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    if metadata.get("role") != "av" or metadata.get("d_model") != EXPECTED_HIDDEN:
        raise ValueError("NLA actor role or hidden dimension does not match the target model")
    if metadata.get("extraction_layer_index") != NLA_LAYER:
        raise ValueError("NLA actor was not trained on block-20 output")


def _source_rows(run: Path, trace_dir: Path | None) -> list[dict[str, Any]]:
    completion = verify_complete_run(run)
    if trace_dir is None:
        return read_jsonl(run / "records.jsonl")
    trace_manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    if trace_manifest.get("source_records_sha256") != completion["records_sha256"]:
        raise ValueError("decision trace does not match completed collection")
    return read_jsonl(trace_dir / "traces.jsonl")


def _vector(
    row: dict[str, Any], run: Path, trace_dir: Path | None,
    position: str, offset: int,
) -> np.ndarray:
    if trace_dir is None:
        vector = np.load(run / row["activation_file"], allow_pickle=False)[NLA_LAYER + 1]
    else:
        with np.load(trace_dir / row["trace_file"], allow_pickle=False) as trace:
            if offset == 0:
                vector = trace[position][NLA_LAYER + 1]
            else:
                position_index = 0 if position == "pre_number" else 1
                vector = trace["block20_neighbors"][position_index, offset + 2]
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (EXPECTED_HIDDEN,) or not np.isfinite(vector).all():
        raise ValueError("invalid NLA target vector")
    return vector


def decode_nla(
    run_dir: str | Path, output_dir: str | Path, *, actor_checkpoint: str | Path,
    official_module: str | Path, server_url: str = "http://localhost:30000",
    critic_checkpoint: str | Path | None = None, max_per_group: int = 20,
    start_per_group: int = 0,
    seed: int = 42, temperature: float = 0.0,
    arms: tuple[str, ...] = ("p_first", "v_only"),
    trace_dir: str | Path | None = None, position: str = "pre_answer", offset: int = 0,
) -> dict[str, Any]:
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("NLA temperature must be between 0 and 2")
    if max_per_group < 1 or start_per_group < 0:
        raise ValueError("NLA group count must be positive and start offset nonnegative")
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError("NLA arms must be a nonempty subset of the prompt arms")
    if trace_dir is None and (position != "pre_answer" or offset != 0):
        raise ValueError("decision positions require a trace directory")
    if trace_dir is not None and (position not in ("pre_number", "pre_verdict") or offset not in (-2, -1, 0)):
        raise ValueError("trace NLA requires a decision position and offset -2, -1, or 0")
    run, out = Path(run_dir), Path(output_dir)
    trace = Path(trace_dir) if trace_dir is not None else None
    out.mkdir(parents=True, exist_ok=True)
    verify_complete_run(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != "Qwen/Qwen2.5-7B-Instruct":
        raise ValueError("released Qwen NLA requires Qwen/Qwen2.5-7B-Instruct target")
    actor_path = Path(actor_checkpoint)
    _check_sidecar(actor_path)
    official = _load_official_client(official_module)
    client = official.NLAClient(actor_path, sglang_url=server_url)
    critic = official.NLACritic(critic_checkpoint, device="cuda:0") if critic_checkpoint else None
    records = [r for r in _source_rows(run, trace) if r["split"] == "test" and r["arm"] in arms and r.get("parse_ok", True)]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row["round"], row["arm"], row.get("condition"),
                row["domain"] if trace is not None else None)
    for row in records:
        groups[group_key(row)].append(row)
    rng = random.Random(seed)
    selected = []
    for group, members in sorted(groups.items(), key=lambda pair: str(pair[0])):
        rng.shuffle(members)
        selected.extend(members[start_per_group:start_per_group + max_per_group])
    decoded = []
    blind = []
    blind_key = []
    for row in selected:
        vector = _vector(row, run, trace, position, offset)
        raw = client.generate(vector, temperature=temperature, max_new_tokens=200, extract_explanation=False)
        match = _EXPLANATION.search(raw)
        explanation = match.group(1).strip() if match else None
        fidelity = None
        if explanation is not None and critic is not None:
            mse, cosine = critic.score(explanation, vector)
            fidelity = {"mse": float(mse), "cosine": float(cosine)}
        group = groups[group_key(row)]
        donors = [candidate for candidate in group if candidate["case_id"] != row["case_id"]]
        control = None
        if donors:
            donor = rng.choice(donors)
            donor_vec = _vector(donor, run, trace, position, offset)
            donor_raw = client.generate(donor_vec, temperature=temperature, max_new_tokens=200, extract_explanation=False)
            donor_match = _EXPLANATION.search(donor_raw)
            donor_explanation = donor_match.group(1).strip() if donor_match else None
            control_fidelity = None
            if donor_explanation is not None and critic is not None:
                donor_mse, donor_cosine = critic.score(donor_explanation, donor_vec)
                control_fidelity = {"mse": float(donor_mse), "cosine": float(donor_cosine)}
            control = {"donor_key": donor["key"], "raw": donor_raw, "explanation": donor_explanation, "fidelity": control_fidelity}
        decoded.append({
            "key": row["key"], "case_id": row["case_id"], "domain": row["domain"],
            "round": row["round"],
            "arm": row["arm"], "condition": row.get("condition"),
            "p_accept": row.get("p_accept"), "verdict": row.get("verdict"),
            "raw": raw, "explanation": explanation, "fidelity": fidelity,
            "control": control,
        })
        for kind, text in (("target", explanation), ("matched_control", control["explanation"] if control else None)):
            if text is not None:
                decode_id = f"D{len(blind_key):05d}"
                blind.append({"decode_id": decode_id, "text": text})
                blind_key.append({"decode_id": decode_id, "kind": kind, "key": row["key"]})
    rng.shuffle(blind)
    write_jsonl(out / "nla_records.jsonl", decoded)
    write_jsonl(out / "blind_coding.jsonl", blind)
    write_jsonl(out / "blind_key.jsonl", blind_key)
    summary = {
        "temperature": temperature,
        "start_per_group": start_per_group,
        "max_per_group": max_per_group,
        "arms": list(arms),
        "position": position,
        "offset": offset,
        "domain_stratified": trace is not None,
        "trace_manifest_sha256": (hashlib.sha256((trace / "manifest.json").read_bytes()).hexdigest()
                                  if trace is not None else None),
        "decoded": len(decoded),
        "parsed_explanations": sum(r["explanation"] is not None for r in decoded),
        "controls": sum(r["control"] is not None for r in decoded),
        "critic_scored": sum(r["fidelity"] is not None for r in decoded),
        "actor_sidecar_sha256": hashlib.sha256((actor_path / "nla_meta.yaml").read_bytes()).hexdigest(),
        "official_module_sha256": hashlib.sha256(Path(official_module).read_bytes()).hexdigest(),
    }
    (out / "nla_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def score_saved_nla(
    run_dir: str | Path, nla_dir: str | Path, output_dir: str | Path,
    *, critic_checkpoint: str | Path, official_module: str | Path,
    trace_dir: str | Path | None = None, position: str = "pre_answer", offset: int = 0,
) -> dict[str, int]:
    """Score actor text in an AR-only phase when actor+AR do not fit together."""
    run, nla, out = Path(run_dir), Path(nla_dir), Path(output_dir)
    trace = Path(trace_dir) if trace_dir is not None else None
    if trace is None and (position != "pre_answer" or offset != 0):
        raise ValueError("decision positions require a trace directory")
    if trace is not None and (position not in ("pre_number", "pre_verdict") or offset not in (-2, -1, 0)):
        raise ValueError("trace NLA requires a decision position and offset -2, -1, or 0")
    out.mkdir(parents=True, exist_ok=True)
    verify_complete_run(run)
    decoded_summary = json.loads((nla / "nla_summary.json").read_text(encoding="utf-8"))
    if decoded_summary.get("position", "pre_answer") != position or decoded_summary.get("offset", 0) != offset:
        raise ValueError("NLA scoring position differs from actor decoding position")
    expected_trace_sha = hashlib.sha256((trace / "manifest.json").read_bytes()).hexdigest() if trace else None
    if decoded_summary.get("trace_manifest_sha256") != expected_trace_sha:
        raise ValueError("NLA scoring trace differs from actor decoding trace")
    official = _load_official_client(official_module)
    critic = official.NLACritic(critic_checkpoint, device="cuda:0")
    source = {row["key"]: row for row in _source_rows(run, trace)}
    scores = []
    for decoded in read_jsonl(nla / "nla_records.jsonl"):
        explanation = decoded.get("explanation")
        if explanation is None:
            continue
        row = source[decoded["key"]]
        vector = _vector(row, run, trace, position, offset)
        mse, cosine = critic.score(explanation, vector)
        result = {"key": decoded["key"], "mse": float(mse), "cosine": float(cosine)}
        control = decoded.get("control")
        if control and control.get("explanation"):
            donor = source[control["donor_key"]]
            donor_vec = _vector(donor, run, trace, position, offset)
            control_mse, control_cosine = critic.score(control["explanation"], donor_vec)
            target_on_donor_mse, target_on_donor_cosine = critic.score(explanation, donor_vec)
            control_on_target_mse, control_on_target_cosine = critic.score(control["explanation"], vector)
            result["control_mse"] = float(control_mse)
            result["control_cosine"] = float(control_cosine)
            result["target_on_donor_mse"] = float(target_on_donor_mse)
            result["target_on_donor_cosine"] = float(target_on_donor_cosine)
            result["control_on_target_mse"] = float(control_on_target_mse)
            result["control_on_target_cosine"] = float(control_on_target_cosine)
            result["target_specificity_mse_margin"] = float(target_on_donor_mse - mse)
            result["control_specificity_mse_margin"] = float(control_on_target_mse - control_mse)
        scores.append(result)
    write_jsonl(out / "nla_fidelity.jsonl", scores)
    return {"scored": len(scores)}
