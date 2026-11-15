"""Regenerate the frozen, outcome-free second wording cohort from source imports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SALT = "cross-wording-prefix-route-v2-2026-11-15"
EXPECTED_CASE_SHA = "8b2dc2783229e172c730e1a4dd8a137de26b4968af4348533ea0813eb80230eb"
EXPECTED_ORIGINAL_SHA = "bd61ddf20c0c54fa80f9be78709a2f7389cfd678bf5c543d9e4ef030c0d15ca0"
EXPECTED_SOURCE_SHA = {
    "grad_admissions": "945a7c99f0f04e18f3f243c02bd47651234044aea60c8f3a8ce24141075c8d05",
    "paper_acceptance": "8195b87b0493782ea01195519467b58196cd87825e9e8093c4acf107900fe8c9",
}
OUTPUT_FIELDS = ("case_id", "domain", "context", "source_ref", "audit_status")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True,
                        help="The 300-case original sample made with seed 42")
    parser.add_argument("--prior-ids", type=Path, required=True,
                        help="configs/validation-160-selection.json")
    parser.add_argument("--gradcafe", type=Path, required=True,
                        help="Imported GradCafe source JSONL")
    parser.add_argument("--peerread", type=Path, required=True,
                        help="Imported PeerRead source JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = {"grad_admissions": args.gradcafe, "paper_acceptance": args.peerread}
    if sha(args.original) != EXPECTED_ORIGINAL_SHA:
        raise ValueError("original 300-case sample differs from the reference import")
    for domain, path in sources.items():
        if sha(path) != EXPECTED_SOURCE_SHA[domain]:
            raise ValueError(f"{domain} import differs from the reference import")

    original = read(args.original)
    source_rows = {domain: read(path) for domain, path in sources.items()}
    source_by_id = {row["case_id"]: row for rows in source_rows.values() for row in rows}
    prior_ids = json.loads(args.prior_ids.read_text())["case_ids"]
    if len(prior_ids) != 160 or len(set(prior_ids)) != 160:
        raise ValueError("prior wording cohort must contain 160 distinct IDs")
    missing = set(prior_ids) - set(source_by_id)
    if missing:
        raise ValueError(f"{len(missing)} prior cohort IDs absent from source imports")
    previous = original + [source_by_id[case_id] for case_id in prior_ids]
    excluded_ids = {row["case_id"] for row in previous}
    excluded_contexts = {fingerprint(row["context"]) for row in previous}

    selected: list[dict] = []
    for domain, rows in source_rows.items():
        eligible = [row for row in rows
                    if row["case_id"] not in excluded_ids
                    and fingerprint(row["context"]) not in excluded_contexts]
        eligible.sort(key=lambda row: hashlib.sha256(
            f"{SALT}:{row['case_id']}".encode()).hexdigest())
        domain_rows: list[dict] = []
        for row in eligible:
            fp = fingerprint(row["context"])
            if fp in excluded_contexts:
                continue
            if row["domain"] != domain or any(
                not isinstance(row.get(key), str) or not row[key] for key in OUTPUT_FIELDS
            ):
                raise ValueError(f"invalid source row {row.get('case_id')}")
            domain_rows.append({key: row[key] for key in OUTPUT_FIELDS})
            excluded_contexts.add(fp)
            excluded_ids.add(row["case_id"])
            if len(domain_rows) == 80:
                break
        if len(domain_rows) != 80:
            raise ValueError(f"fewer than 80 eligible cases in {domain}")
        selected.extend(domain_rows)
    selected.sort(key=lambda row: (row["domain"], row["case_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in selected
    ))
    if sha(args.output) != EXPECTED_CASE_SHA:
        raise ValueError("selected cohort differs from the frozen experiment")
    print(json.dumps({"cases": len(selected), "sha256": EXPECTED_CASE_SHA}, sort_keys=True))


if __name__ == "__main__":
    main()
