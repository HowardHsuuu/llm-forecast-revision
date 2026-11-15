"""Collect the two fixed first-person baselines for steering confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forecast_mechanisms.collect import collect
from forecast_mechanisms.core import read_jsonl, verify_complete_run
from forecast_mechanisms.model import ForecastRunner


EXPECTED = {
    "train": ("40419a3b935fe7dfa877b4586662d49f77abef4071d739b1434705c7f996f78a",
              "76c1954d3a44c345ef3556f79adf840a93bd0446cc34160ee61ccd87daa64a05"),
    "confirm": ("15f8f9bb81d4c94ea9a9f10b94ef42119f53e33b3fe8de286171665a7b144aeb",
                "0d25bdfbdf0652cdff66de68df074539dda7a24e332a6b8c67201208c3a61f92"),
}


def main(cases: Path, output: Path, cohort: str) -> None:
    case_sha, baseline_sha = EXPECTED[cohort]
    if hashlib.sha256(cases.read_bytes()).hexdigest() != case_sha:
        raise ValueError("frozen case file changed")
    runner = ForecastRunner(max_new_tokens=96)
    result = collect(cases, output, runner, seed=42, arms=("p_first",),
                     second_sources=("p_first",),
                     followups=("neutral", "opposite"),
                     prompt_voice="first_person")
    completed = verify_complete_run(output)
    rows = read_jsonl(output / "records.jsonl")
    if (len(rows) != 480 or completed["records_sha256"] != baseline_sha or
            any(not r["parse_ok"] for r in rows)):
        raise ValueError("frozen baseline count, parsing, or hash differs")
    print(json.dumps({"collection": result, "records_sha256": baseline_sha},
                     sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.cases, args.output, args.cohort)
