"""Verify the paired steering direction from the first new cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from forecast_mechanisms.core import read_jsonl
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION


TRAIN_SHA = "76c1954d3a44c345ef3556f79adf840a93bd0446cc34160ee61ccd87daa64a05"
INDEX = 22  # HF hidden_states index: output of decoder block 21


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(train: Path, frozen: Path) -> None:
    manifest = json.loads((train / "manifest.json").read_text())
    completion = json.loads((train / "completion.json").read_text())
    records = train / "records.jsonl"
    if (sha(records) != TRAIN_SHA or completion["records_sha256"] != TRAIN_SHA or
            (manifest["model"], manifest["revision"]) !=
            (DEFAULT_MODEL, DEFAULT_REVISION) or
            manifest.get("prompt_voice") != "first_person"):
        raise ValueError("frozen training run differs")
    cases: dict[str, dict] = {}
    for row in read_jsonl(records):
        if row["arm"] != "p_first":
            raise ValueError("unexpected output arm")
        key = "first" if row["round"] == 1 else row["condition"]
        cases.setdefault(row["case_id"], {})[key] = row
    if len(cases) != 160:
        raise ValueError("training cohort changed")
    contrasts = []
    for case_id in sorted(cases):
        pair = cases[case_id]
        if pair["first"]["verdict"] != "accept":
            continue
        if not all(pair[k]["parse_ok"] for k in ("first", "neutral", "opposite")):
            raise ValueError("training pair did not parse")
        neutral = np.load(train / pair["neutral"]["activation_file"],
                          allow_pickle=False)[INDEX].astype(np.float64)
        opposite = np.load(train / pair["opposite"]["activation_file"],
                           allow_pickle=False)[INDEX].astype(np.float64)
        if neutral.shape != (3584,) or opposite.shape != (3584,):
            raise ValueError("unexpected hidden-state shape")
        contrasts.append(neutral - opposite)
    if len(contrasts) != 147:
        raise ValueError("initial-accept training count changed")
    mean = np.stack(contrasts).mean(axis=0)
    dose = float(np.linalg.norm(mean))
    unit = mean / dose
    saved = np.load(frozen, allow_pickle=False)
    if (int(saved["state_index"]) != INDEX or
            abs(float(saved["high_dose"]) - dose) > 1e-4 or
            abs(float(saved["low_dose"]) - dose / 2) > 1e-4 or
            np.max(np.abs(np.asarray(saved["paired_neutral"], dtype=np.float64) - unit)) > 1e-5):
        raise ValueError("recomputed direction differs from frozen steering input")
    print(json.dumps({"training_initial_accepts": len(contrasts),
                      "state_index": INDEX, "high_dose_norm": dose,
                      "direction_sha256": sha(frozen)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()
    main(args.train, args.frozen)
