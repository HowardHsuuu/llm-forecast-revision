"""Replace the written percentage and read the model's next verdict token."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forecast_mechanisms.core import append_jsonl, read_jsonl
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION, ForecastRunner
from forecast_mechanisms.steering_steps import verdict_step


CONDITIONS = ("opposite_natural_p", "opposite_first_p", "neutral_first_p")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(baseline: Path, expected_sha256: str,
               expected_cases: int, expected_primary: int) -> dict[str, dict]:
    records = baseline / "records.jsonl"
    completion = json.loads((baseline / "completion.json").read_text())
    if (digest(records) != expected_sha256 or
            completion["records_sha256"] != expected_sha256):
        raise ValueError("baseline records differ from the expected frozen run")
    grouped = {}
    baseline_rows = read_jsonl(records)
    if completion['total_records'] != len(baseline_rows):
        raise ValueError('baseline completion count differs from its records')
    for row in baseline_rows:
        case = grouped.setdefault(row["case_id"], {})
        key = "first" if row["round"] == 1 else row["condition"]
        if key in case:
            raise ValueError("duplicate case-condition")
        case[key] = row
    cases = {case_id: rows for case_id, rows in grouped.items()
             if set(rows) == {"first", "neutral", "opposite"} and
             all(row["parse_ok"] for row in rows.values())}
    primary = sum(c["first"]["verdict"] == "accept" and
                  c["opposite"]["verdict"] == "reject" and
                  c["first"]["p_accept"] != c["opposite"]["p_accept"]
                  for c in cases.values())
    if len(cases) != expected_cases or primary != expected_primary:
        raise ValueError("eligible case flow differs from the frozen run")
    return cases


def replay(runner: ForecastRunner, messages: list[dict],
           expected: str, percentage: int) -> list[int]:
    torch = runner.torch
    ids = runner._ids(messages)
    with torch.inference_mode():
        generated = runner.model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=96, do_sample=False,
            pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                          else runner.tokenizer.eos_token_id),
        )
    tokens = generated[0, ids.shape[-1]:].tolist()
    text = runner.tokenizer.decode(tokens, skip_special_tokens=True).strip()
    if (text != expected or verdict_step(runner.tokenizer, tokens) != 11 or
            runner.tokenizer.decode(tokens[5:7]) != str(percentage) or
            runner.tokenizer.decode(tokens[:11]) !=
            f"P(accept): {percentage}%\nVerdict:"):
        raise ValueError("natural answer or numeric-token layout changed")
    return tokens


def numeric_tokens(runner: ForecastRunner, percentage: int) -> list[int]:
    tokens = runner.tokenizer.encode(str(percentage), add_special_tokens=False)
    if len(tokens) != 2 or runner.tokenizer.decode(tokens) != str(percentage):
        raise ValueError("reference percentage is not two digit tokens")
    return tokens


def next_verdict(runner: ForecastRunner, messages: list[dict],
                 prefix: list[int], accept_id: int, reject_id: int) -> dict:
    torch = runner.torch
    ids = runner._ids(messages)
    full = torch.cat((ids, torch.tensor([prefix], dtype=ids.dtype, device=ids.device)), dim=1)
    with torch.inference_mode():
        out = runner.model.generate(
            input_ids=full, attention_mask=torch.ones_like(full),
            max_new_tokens=1, do_sample=False,
            pad_token_id=(runner.tokenizer.pad_token_id if runner.tokenizer.pad_token_id is not None
                          else runner.tokenizer.eos_token_id),
            return_dict_in_generate=True, output_scores=True,
        )
    token = int(out.sequences[0, -1])
    scores = out.scores[0][0].float()
    return {
        "token": token,
        "candidate": ("accept" if token == accept_id else
                      "reject" if token == reject_id else "other"),
        "accept_minus_reject_logit": float(scores[accept_id] - scores[reject_id]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-records-sha256", required=True)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument("--expected-primary", required=True, type=int)
    parser.add_argument("--only-case")
    args = parser.parse_args()
    if (DEFAULT_MODEL, DEFAULT_REVISION) != (
            "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"):
        raise ValueError("model revision changed")
    cases = load_cases(args.baseline, args.expected_records_sha256,
                       args.expected_cases, args.expected_primary)
    ordered = sorted(cases)
    if args.only_case:
        if args.only_case not in cases:
            raise ValueError("requested case is not eligible")
        ordered = [args.only_case]
    runner = ForecastRunner(max_new_tokens=96)
    accept = runner.tokenizer.encode(" ACCEPT", add_special_tokens=False)
    reject = runner.tokenizer.encode(" REJECT", add_special_tokens=False)
    if len(accept) != 1 or not reject or accept[0] == reject[0]:
        raise ValueError("verdict candidate tokenization changed")
    accept_id, reject_id = accept[0], reject[0]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "forced-number-verdict-v1", "script_sha256": digest(Path(__file__)),
        "baseline_records_sha256": args.expected_records_sha256,
        "model": DEFAULT_MODEL, "revision": DEFAULT_REVISION,
        "conditions": list(CONDITIONS), "case_ids": ordered,
        "full_cohort_cases": args.expected_cases,
        "full_cohort_primary": args.expected_primary,
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("manifest differs from an existing run")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    record_file = args.output / "records.jsonl"
    existing = read_jsonl(record_file) if record_file.exists() else []
    done = {row["case_id"] for row in existing}
    if len(done) != len(existing) or not done <= set(ordered):
        raise ValueError("existing record IDs differ")
    for row in existing:
        case = cases[row['case_id']]
        expected = {'domain': case['first']['domain']}
        expected.update({f'{name}_{field}': case[name][source]
                         for name in ('first','neutral','opposite')
                         for field,source in (('p','p_accept'),('verdict','verdict'))})
        if (any(row.get(key) != value for key,value in expected.items()) or
                set(row.get('conditions', {})) != set(CONDITIONS)):
            raise ValueError('saved intervention record differs from its baseline case')
    for case_id in ordered:
        if case_id in done:
            continue
        case = cases[case_id]
        first, neutral, opposite = (case[name] for name in
                                    ("first", "neutral", "opposite"))
        opposite_tokens = replay(runner, opposite["messages"], opposite["response"],
                                 opposite["p_accept"])
        neutral_tokens = replay(runner, neutral["messages"], neutral["response"],
                                neutral["p_accept"])
        conditions = {}
        for name in CONDITIONS:
            source = neutral if name == "neutral_first_p" else opposite
            natural = neutral_tokens if name == "neutral_first_p" else opposite_tokens
            percentage = int(opposite["p_accept"] if name == "opposite_natural_p"
                             else first["p_accept"])
            prefix = natural[:5] + numeric_tokens(runner, percentage) + natural[7:11]
            if runner.tokenizer.decode(prefix) != f"P(accept): {percentage}%\nVerdict:":
                raise ValueError("forced prefix changed a nonnumeric token")
            conditions[name] = {"forced_p": percentage, "prefix_token_ids": prefix,
                                **next_verdict(runner, source["messages"], prefix,
                                               accept_id, reject_id)}
        if conditions["opposite_natural_p"]["token"] != opposite_tokens[11]:
            raise ValueError("natural-percentage sham failed")
        if first["p_accept"] == neutral["p_accept"] and (
                conditions["neutral_first_p"]["token"] != neutral_tokens[11]):
            raise ValueError("neutral-percent replay failed")
        append_jsonl(record_file, {
            "case_id": case_id, "domain": first["domain"],
            "first_p": first["p_accept"], "first_verdict": first["verdict"],
            "neutral_p": neutral["p_accept"], "neutral_verdict": neutral["verdict"],
            "opposite_p": opposite["p_accept"], "opposite_verdict": opposite["verdict"],
            "natural_opposite_verdict_token": opposite_tokens[11],
            "natural_neutral_verdict_token": neutral_tokens[11],
            "conditions": conditions,
        })
        done.add(case_id)
    if done == set(ordered):
        (args.output / "completion.json").write_text(json.dumps({
            "total_records": len(done), "records_sha256": digest(record_file),
        }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"total_records": len(done), "expected": len(ordered)}, sort_keys=True))


if __name__ == "__main__":
    main()
