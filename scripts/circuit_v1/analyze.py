"""Validate and summarize the complete two-sample circuit experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from forecast_mechanisms.core import read_jsonl
from forecast_mechanisms.prefix_circuit import exact


ROUTES = ('base', 'orthogonal', 'value21', 'later22_27', 'key_value21',
          'key_value21_later22_27', 'single26', 'leave26_out')
KV_CONDITIONS = ('edited_kv', 'natural_k', 'natural_v', 'natural_kv')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=256)
def bootstrap_indices(n: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.fromiter((rng.randrange(n) for _ in range(10000 * n)),
                       dtype=np.int32, count=10000 * n).reshape(10000, n)


def estimate(values: list[float], seed: int = 42) -> dict:
    if not values:
        return {'n': 0}
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError('nonfinite score')
    means = np.sort(arr[bootstrap_indices(len(values), seed)].mean(axis=1))
    return {'n': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'ci95': [float(means[249]), float(means[9749])]}


def validate_rows(rows: list[dict]) -> None:
    if len(rows) != 75 or len({r['case_id'] for r in rows}) != 75:
        raise ValueError('analysis requires all 75 unique cases')
    for r in rows:
        if r['domain'] not in ('grad_admissions', 'paper_acceptance'):
            raise ValueError('unknown task domain')
        routes, head, kv = r['routes'], r['head_map'], r['kv']
        if (set(routes) != set(ROUTES) or set(head['heads']) != {str(i) for i in range(28)}
                or set(kv['conditions']) != set(KV_CONDITIONS)):
            raise ValueError('incomplete intervention matrix')
        results = [*routes.values(), head['base'], head['block26_rescue'],
                   head['all_head_reconstruction'], *kv['conditions'].values()]
        if any(x['status'] != 'valid' for x in results):
            raise ValueError('a visible answer prefix changed')
        if (not exact(head['base'], routes['key_value21'])
                or not exact(head['block26_rescue'], routes['single26'])
                or not exact(head['all_head_reconstruction'], head['block26_rescue'])
                or not exact(kv['conditions']['edited_kv'], head['base'])):
            raise ValueError('route, head and K/V anchors disagree')
        full = routes['key_value21_later22_27']
        if (full['processed_margin'] != r['natural_processed_margin'] or
                full['raw_margin'] != r['natural_raw_margin']):
            raise ValueError('combined cache restoration differs from the natural scores')
        joint = kv['conditions']['natural_kv']
        h = head['heads']['26']
        if (joint['processed_margin'] - head['base']['processed_margin'] != h['processed_margin_shift']
                or joint['raw_margin'] - head['base']['raw_margin'] != h['raw_margin_shift']
                or joint['donor_first_token_match'] != h['donor_first_token_match']):
            raise ValueError('joint K/V result differs from head-26 restoration')
        for x in results:
            if not np.isfinite([x['processed_margin'], x['raw_margin']]).all():
                raise ValueError('nonfinite score')


def summarize(rows: list[dict]) -> dict:
    def difference(a, b):
        return estimate([r['routes'][a]['processed_margin'] -
                         r['routes'][b]['processed_margin'] for r in rows])
    route = {
        'base_minus_natural': estimate([r['routes']['base']['processed_margin'] -
                                       r['natural_processed_margin'] for r in rows]),
        'base_minus_orthogonal': difference('base', 'orthogonal'),
        'value21_restoration': difference('value21', 'base'),
        'later22_27_restoration': difference('later22_27', 'base'),
        'key_value21_restoration': difference('key_value21', 'base'),
        'all_cache_restoration': difference('key_value21_later22_27', 'base'),
        'block26_single_after_key_value21': difference('single26', 'key_value21'),
        'block26_leave_out_effect': difference('key_value21_later22_27', 'leave26_out'),
        'neutral_first_token_matches': {name: sum(r['routes'][name]['donor_first_token_match']
            for r in rows) for name in ROUTES},
        'exact_natural_score_restorations': sum(
            r['routes']['key_value21_later22_27']['processed_margin'] == r['natural_processed_margin']
            and r['routes']['key_value21_later22_27']['raw_margin'] == r['natural_raw_margin'] for r in rows),
    }
    heads = {str(i): {
        'processed_effect': estimate([r['head_map']['heads'][str(i)]['processed_margin_shift']
                                      for r in rows], seed=42+i),
        'raw_effect': estimate([r['head_map']['heads'][str(i)]['raw_margin_shift']
                                for r in rows], seed=42+i),
        'neutral_first_token_matches': sum(r['head_map']['heads'][str(i)]['donor_first_token_match']
                                           for r in rows),
    } for i in range(28)}
    head_map = {'all_28_heads_effect': estimate([
        r['head_map']['block26_rescue']['processed_margin'] -
        r['head_map']['base']['processed_margin'] for r in rows]), 'heads': heads}
    kv = {name: {
        'processed_shift': estimate([r['kv']['conditions'][name]['processed_margin'] -
            r['kv']['conditions']['edited_kv']['processed_margin'] for r in rows]),
        'raw_shift': estimate([r['kv']['conditions'][name]['raw_margin'] -
            r['kv']['conditions']['edited_kv']['raw_margin'] for r in rows]),
        'neutral_first_token_matches': sum(r['kv']['conditions'][name]['donor_first_token_match']
                                           for r in rows),
    } for name in KV_CONDITIONS}
    return {'n_cases': len(rows), 'routes': route, 'block26_heads': head_map,
            'head26_key_value': kv}


def load_run(path: Path, cohort: str) -> list[dict]:
    manifest = json.loads((path / 'manifest.json').read_text())
    completion = json.loads((path / 'completion.json').read_text())
    records = path / 'records.jsonl'
    rows = read_jsonl(records)
    if (manifest['protocol'] != 'answer-prefix-circuit-v1' or manifest['cohort'] != cohort
            or manifest['model'] != 'Qwen/Qwen2.5-7B-Instruct'
            or manifest['revision'] != 'a09a35458c702b33eeacc393d103063234e8bc28'
            or manifest['full_cohort_cases'] != 75 or len(manifest['case_ids']) != 75
            or len(set(manifest['case_ids'])) != 75
            or completion['total_records'] != 75
            or completion['records_sha256'] != digest(records)
            or completion['manifest_sha256'] != digest(path / 'manifest.json')
            or {r['case_id'] for r in rows} != set(manifest['case_ids'])):
        raise ValueError('incomplete or changed circuit run')
    validate_rows(rows)
    return rows


def analyze(case_paired: Path, initial_wording: Path) -> dict:
    rows = {name: load_run(path, name) for name, path in
            (('case_paired', case_paired), ('initial_wording', initial_wording))}
    if {r['case_id'] for r in rows['case_paired']} & {r['case_id'] for r in rows['initial_wording']}:
        raise ValueError('case samples overlap')
    cohorts = {}
    for name, data in rows.items():
        cohorts[name] = {'all': summarize(data)}
        for domain in ('grad_admissions', 'paper_acceptance'):
            cohorts[name][domain] = summarize([r for r in data if r['domain'] == domain])
    a, b = ([cohorts[name]['all']['block26_heads']['heads'][str(i)]['processed_effect']['mean']
             for i in range(28)] for name in ('case_paired', 'initial_wording'))
    return {'protocol': 'answer-prefix-circuit-v1-summary', 'cohorts': cohorts,
            'cross_sample_head_effect_pearson': float(np.corrcoef(a, b)[0, 1]),
            'bootstrap': {'resamples': 10000, 'unit': 'case', 'seed': '42; head i uses 42+i'},
            'interpretation': 'Local propagation of the specified punctuation-state replacement into the first number, with non-target head outputs held fixed as described in the reproduction guide.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-paired', type=Path, required=True)
    parser.add_argument('--initial-wording', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.case_paired, args.initial_wording)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({name: result['cohorts'][name]['all']['head26_key_value']['natural_v']['processed_shift']
                      for name in result['cohorts']}, sort_keys=True))
