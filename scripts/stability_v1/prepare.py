"""Reconstruct frozen stability cohorts from explicit imported data paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forecast_mechanisms.core import read_jsonl
from forecast_stability_protocol_v1 import select_development


FIELDS = ("case_id", "domain", "context", "source_ref", "audit_status")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded_rows(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in rows).encode("utf-8")


def guarded_write(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"refusing to replace different prepared input: {path.name}")
    if not path.exists():
        path.write_bytes(content)


def baseline_records(path: Path, expected: str) -> tuple[Path, list[dict]]:
    records = path / "records.jsonl" if path.is_dir() else path
    if digest(records) != expected:
        raise ValueError("baseline records do not match the frozen selection")
    marker = records.parent / "completion.json"
    rows = read_jsonl(records)
    if marker.exists() and (path.is_dir() or records.name == "records.jsonl"):
        completion = json.loads(marker.read_text())
        if (completion.get("records_sha256") != expected or
                completion.get("total_records") != len(rows)):
            raise ValueError("baseline completion marker differs")
    return records, rows


def prepare(train_baseline: Path, dev_baseline: Path, gradcafe: Path,
            peerread: Path, selection_path: Path, output: Path) -> dict:
    selection = json.loads(selection_path.read_text())
    if (selection.get("protocol") != "forecast-stability-v1" or
            selection.get("excluded_prior_cases") != 780 or
            selection.get("train_n") != 160 or selection.get("dev_n") != 25 or
            len(selection.get("test_ids", [])) != 117 or
            len(set(selection["test_ids"])) != 117 or
            len(set(selection.get("dev_ids", []))) != 25):
        raise ValueError("selection metadata differs from this frozen protocol")
    sources = {"gradcafe-cs.jsonl": gradcafe, "peerread-iclr2017.jsonl": peerread}
    for name, path in sources.items():
        if digest(path) != selection["source_sha256"][name]:
            raise ValueError(f"imported source revision differs: {name}")
    _, train_rows = baseline_records(train_baseline, selection["train_baseline_sha256"])
    _, dev_rows = baseline_records(dev_baseline, selection["dev_baseline_sha256"])
    train = [r for r in train_rows if r["round"] == 1 and r["parse_ok"]]
    development = select_development(
        [r for r in dev_rows if r["round"] == 1 and r["parse_ok"]], per_stratum=8)
    if (len(train) != selection["train_n"] or len(development) != selection["dev_n"] or
            [r["case_id"] for r in development] != selection["dev_ids"] or
            len({r["case_id"] for r in train}) != len(train)):
        raise ValueError("training or development case flow differs")
    desired = set(selection["test_ids"])
    selected = {}
    for source in sources.values():
        for row in read_jsonl(source):
            if row["case_id"] not in desired:
                continue
            if row["case_id"] in selected:
                raise ValueError("test case occurs more than once in the sources")
            selected[row["case_id"]] = {k: row[k] for k in FIELDS}
    if set(selected) != desired:
        raise ValueError("a selected test case is absent from the imported sources")
    test = sorted(selected.values(), key=lambda r: (r["domain"], r["case_id"]))
    if ([r["case_id"] for r in test] != selection["test_ids"] or
            sum(r["domain"] == "grad_admissions" for r in test) != 80 or
            sum(r["domain"] == "paper_acceptance" for r in test) != 37 or
            hashlib.sha256(encoded_rows(test)).hexdigest() != selection["test_cases_sha256"]):
        raise ValueError("reconstructed test cases differ from the frozen digest")
    prior_ids = {r["case_id"] for r in train_rows + dev_rows}
    if desired & prior_ids:
        raise ValueError("test overlaps the training/development inputs")
    outputs = {
        "train-first.jsonl": encoded_rows(train), "dev-first.jsonl": encoded_rows(development),
        "test-cases.jsonl": encoded_rows(test), "train-baseline.jsonl": encoded_rows(train_rows),
        "dev-baseline.jsonl": encoded_rows(dev_rows),
        "selection.json": (json.dumps(selection, sort_keys=True, indent=2) + "\n").encode(),
    }
    # Check all existing files before writing any new prepared file.
    for name, content in outputs.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"prepared input differs: {name}")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        guarded_write(output / name, content)
    return {"train": len(train), "dev": len(development), "test": len(test),
            "prepared_sha256": {name: hashlib.sha256(content).hexdigest()
                                for name, content in outputs.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-baseline", type=Path, required=True)
    parser.add_argument("--dev-baseline", type=Path, required=True)
    parser.add_argument("--gradcafe", type=Path, required=True)
    parser.add_argument("--peerread", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.train_baseline, args.dev_baseline, args.gradcafe,
                     args.peerread, args.selection, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
