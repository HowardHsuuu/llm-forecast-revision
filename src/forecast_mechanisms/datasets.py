"""Loss-aware importers for historical GradCafe and PeerRead records."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any

from .core import read_jsonl, validate_cases, write_jsonl


def _field(row: dict[str, str], *needles: str) -> str:
    for key, value in row.items():
        key_norm = re.sub(r"[^a-z0-9]", "", key.lower())
        if any(re.sub(r"[^a-z0-9]", "", needle.lower()) in key_norm for needle in needles):
            return (value or "").strip()
    return ""


def _decision(value: Any) -> str | None:
    if isinstance(value, bool):
        return "accept" if value else "reject"
    if isinstance(value, str):
        value = value.strip().lower()
        if value.startswith(("accept", "admit")):
            return "accept"
        if value.startswith(("reject", "deny", "declin")):
            return "reject"
    return None


def import_gradcafe(source: str | Path, output: str | Path, limit: int | None = None) -> dict[str, int]:
    paths = sorted(Path(source).glob("*.csv")) if Path(source).is_dir() else [Path(source)]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("GradCafe source must be a CSV file or directory of CSV files")
    cases: list[dict[str, Any]] = []
    rejected_invalid = rejected_duplicate = 0
    seen_context = set()
    positional_headers = [
        "Institution", "Subject", "Degree type and admission semester",
        "Acception/rejection", "Method of communication", "Decision date",
        "GPA", "GRE Verbal", "GRE Quantitative", "GRE Writing", "GRE Subject",
        "Student classification", "Date posted", "Comments",
    ]
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
            reader = csv.reader(handle)
            first = next(reader, None)
            if first is None:
                continue
            has_header = first[0].strip().lower() in ("institution", "university")
            headers = first if has_header else positional_headers
            numbered = enumerate(reader, 2) if has_header else itertools.chain([(1, first)], enumerate(reader, 2))
            for line_no, values in numbered:
                row = dict(zip(headers, values))
                outcome = _decision(_field(row, "acception/rejection", "acceptance/rejection", "decision"))
                institution = _field(row, "institution", "university")
                subject = _field(row, "subject", "major") or path.stem
                degree = _field(row, "degree type and admission semester", "degree", "program")
                if outcome is None or not institution or not degree:
                    rejected_invalid += 1
                    continue
                fields = [
                    ("Institution", institution), ("Subject", subject),
                    ("Degree and admission term", degree),
                    ("GPA", _field(row, "gpa")),
                    ("GRE verbal", _field(row, "gre verbal")),
                    ("GRE quantitative", _field(row, "gre quantitative")),
                    ("GRE writing", _field(row, "gre writing")),
                    ("GRE subject", _field(row, "gre subject")),
                    ("Applicant classification", _field(row, "student classification")),
                ]
                context = "Graduate admissions application:\n" + "\n".join(
                    f"{name}: {value}" for name, value in fields if value
                )
                if context in seen_context:
                    rejected_duplicate += 1
                    continue
                seen_context.add(context)
                source_ref = f"{path.name}:{line_no}"
                case_id = "gradcafe-" + hashlib.sha256(
                    f"{path.name}:{line_no}:{institution}:{degree}".encode()
                ).hexdigest()[:16]
                cases.append({
                    "case_id": case_id, "domain": "grad_admissions", "context": context,
                    "source_ref": source_ref, "outcome": outcome,
                    "audit_status": "legacy_unverified",
                    "audit_note": "Self-reported historical row; only application fields used. No predecision snapshot proof.",
                })
                if limit is not None and len(cases) >= limit:
                    write_jsonl(output, cases)
                    return {"written": len(cases), "skipped": rejected_invalid + rejected_duplicate, "skipped_invalid": rejected_invalid, "skipped_duplicate": rejected_duplicate}
    write_jsonl(output, cases)
    return {"written": len(cases), "skipped": rejected_invalid + rejected_duplicate, "skipped_invalid": rejected_invalid, "skipped_duplicate": rejected_duplicate}


def import_peerread(source: str | Path, output: str | Path, limit: int | None = None) -> dict[str, int]:
    paths = sorted(Path(source).rglob("*.json"))
    if not paths:
        raise ValueError("PeerRead source directory contains no JSON files")
    cases: list[dict[str, Any]] = []
    rejected = 0
    seen_context = set()
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            rejected += 1
            continue
        if not isinstance(row, dict):
            rejected += 1
            continue
        outcome = _decision(row.get("accepted", row.get("decision")))
        title = row.get("title")
        abstract = row.get("abstract") or row.get("abstractText")
        if outcome is None or not isinstance(title, str) or not isinstance(abstract, str):
            rejected += 1
            continue
        # Never place reviews, ratings, author response, or decision in context.
        venue = path.parts[-4] if len(path.parts) >= 4 else "conference"
        context = f"A paper submitted to {venue} awaits a decision.\nTitle: {title.strip()}\nAbstract: {abstract.strip()}"
        if context in seen_context:
            rejected += 1
            continue
        seen_context.add(context)
        source_ref = str(path.relative_to(source))
        case_id = "peerread-" + hashlib.sha256(source_ref.encode()).hexdigest()[:16]
        cases.append({
            "case_id": case_id, "domain": "paper_acceptance", "context": context,
            "source_ref": source_ref, "outcome": outcome,
            "audit_status": "legacy_unverified",
            "audit_note": "Historical paper. Title/abstract only; predecision version and model contamination not independently audited.",
        })
        if limit is not None and len(cases) >= limit:
            break
    write_jsonl(output, cases)
    return {"written": len(cases), "skipped": rejected}


def sample_cases(inputs: list[str | Path], output: str | Path, per_domain: int, seed: int = 42) -> dict[str, Any]:
    """Combine sources and sample without selecting on outcome labels."""
    if per_domain < 1:
        raise ValueError("per_domain must be positive")
    pool: dict[str, list[dict[str, Any]]] = {}
    seen_ids, seen_context = set(), set()
    duplicates = 0
    for source in inputs:
        for row in read_jsonl(source):
            fingerprint = hashlib.sha256(" ".join(row["context"].lower().split()).encode()).hexdigest()
            if row["case_id"] in seen_ids or fingerprint in seen_context:
                duplicates += 1
                continue
            seen_ids.add(row["case_id"])
            seen_context.add(fingerprint)
            pool.setdefault(row["domain"], []).append(row)
    rng = random.Random(seed)
    selected = []
    for domain, members in sorted(pool.items()):
        rng.shuffle(members)
        selected.extend(members[:per_domain])
    selected.sort(key=lambda row: row["case_id"])
    validate_cases(selected)
    write_jsonl(output, selected)
    return {"written": len(selected), "by_domain": {domain: min(len(members), per_domain) for domain, members in pool.items()}, "duplicates_removed": duplicates}
