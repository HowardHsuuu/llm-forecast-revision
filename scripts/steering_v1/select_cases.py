"""Reconstruct the two fixed case files used by steering confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIELDS = ("case_id", "domain", "context", "source_ref", "audit_status")
SOURCE_SHA = {"grad_admissions":
              "945a7c99f0f04e18f3f243c02bd47651234044aea60c8f3a8ce24141075c8d05",
              "paper_acceptance":
              "8195b87b0493782ea01195519467b58196cd87825e9e8093c4acf107900fe8c9"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def main(selection: Path, gradcafe: Path, peerread: Path, output: Path) -> None:
    spec = json.loads(selection.read_text())
    sources = {"grad_admissions": gradcafe, "paper_acceptance": peerread}
    for domain, path in sources.items():
        if sha(path) != SOURCE_SHA[domain]:
            raise ValueError(f"{domain} source revision differs")
    candidates = {r["case_id"]: r for path in sources.values() for r in rows(path)}
    ids = spec["case_ids"]
    if len(ids) != 160 or len(set(ids)) != 160 or any(k not in candidates for k in ids):
        raise ValueError("frozen case IDs are missing or repeated")
    fields = (FIELDS if spec["protocol"] == "generation-steering-confirmation-v1"
              else (*FIELDS, "audit_note", "outcome"))
    selected = [{field: candidates[key][field] for field in fields} for key in ids]
    expected_counts = spec["counts"] if "counts" in spec else {
        d: spec["per_domain"] for d in sources}
    if (selected != sorted(selected, key=lambda r: (r["domain"], r["case_id"])) or
            any(not all(isinstance(r[f], str) and r[f] for f in FIELDS)
                for r in selected) or
            {d: sum(r["domain"] == d for r in selected) for d in sources} != expected_counts):
        raise ValueError("frozen case order, domain, or fields differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                              for r in selected))
    if sha(output) != spec["cases_sha256"]:
        output.unlink()
        raise ValueError("reconstructed case-file SHA-256 differs")
    print(json.dumps({"cases": len(selected), "cases_sha256": sha(output)},
                     sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--gradcafe", type=Path, required=True)
    parser.add_argument("--peerread", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.selection, args.gradcafe, args.peerread, args.output)
