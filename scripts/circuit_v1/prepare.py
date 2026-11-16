"""Reconstruct the two fixed circuit samples from collected conversations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forecast_mechanisms.core import prompt_digest, read_jsonl


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(baseline: Path, selection: Path, cohort: str, output: Path) -> dict:
    spec = json.loads(selection.read_text())
    if (spec['protocol'] != 'answer-prefix-circuit-v1' or
            spec['model'] != 'Qwen/Qwen2.5-7B-Instruct' or
            spec['revision'] != 'a09a35458c702b33eeacc393d103063234e8bc28'):
        raise ValueError('model or protocol differs from the fixed experiment')
    selected = spec['cohorts'][cohort]
    path = baseline / 'records.jsonl' if baseline.is_dir() else baseline
    if digest(path) != selected['baseline_records_sha256']:
        raise ValueError('baseline records differ from the fixed sample')
    rows = read_jsonl(path)
    if baseline.is_dir():
        completion = json.loads((baseline / 'completion.json').read_text())
        if (completion['records_sha256'] != digest(path) or
                completion['total_records'] != len(rows)):
            raise ValueError('baseline completion marker differs')
    ids = selected['case_ids']
    if len(ids) != 75 or len(set(ids)) != 75 or ids != sorted(ids):
        raise ValueError('expected 75 unique ordered cases')
    second = {}
    for row in rows:
        if row['round'] == 2 and row['condition'] in ('neutral', 'opposite'):
            key = row['case_id'], row['condition']
            if key in second:
                raise ValueError('duplicate baseline continuation')
            second[key] = row
    prepared = []
    for case_id in ids:
        neutral, opposite = (second[case_id, condition] for condition in ('neutral', 'opposite'))
        if (not neutral['parse_ok'] or not opposite['parse_ok'] or
                neutral['domain'] != opposite['domain'] or
                neutral['p_accept'] == opposite['p_accept']):
            raise ValueError('selected pair is no longer eligible')
        for row in (neutral, opposite):
            if prompt_digest(row['messages']) != row['prompt_sha256']:
                raise ValueError('saved conversation hash differs')
        prepared.append({
            'case_id': case_id, 'domain': neutral['domain'],
            'neutral_messages': neutral['messages'], 'opposite_messages': opposite['messages'],
            'neutral_p': neutral['p_accept'], 'opposite_p': opposite['p_accept'],
            'neutral_prompt_sha256': neutral['prompt_sha256'],
            'opposite_prompt_sha256': opposite['prompt_sha256'],
        })
    counts = {domain: sum(r['domain'] == domain for r in prepared)
              for domain in selected['counts']}
    content = ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n'
                      for r in prepared).encode()
    if counts != selected['counts'] or hashlib.sha256(content).hexdigest() != selected['cases_sha256']:
        raise ValueError('prepared sample differs from the frozen sample')
    if output.exists() and output.read_bytes() != content:
        raise ValueError('refusing to replace a different prepared sample')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return {'cohort': cohort, 'n_cases': len(prepared), 'counts': counts,
            'cases_sha256': digest(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--cohort', required=True, choices=('case_paired', 'initial_wording'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.baseline, args.selection, args.cohort, args.output), sort_keys=True))
