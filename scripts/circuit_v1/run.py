"""Replay the prefix-state intervention, cache restorations and head dissection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from forecast_mechanisms.attention_route import PREFIX_IDS, capture_step
from forecast_mechanisms.core import append_jsonl, prompt_digest, read_jsonl
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION, ForecastRunner
import forecast_mechanisms.prefix_circuit as circuit


ROUTES = ('base', 'orthogonal', 'value21', 'later22_27', 'key_value21',
          'key_value21_later22_27', 'single26', 'leave26_out')
KV_CONDITIONS = ('edited_kv', 'natural_k', 'natural_v', 'natural_kv')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(runner: ForecastRunner, case: dict, orthogonal_seed: str) -> dict:
    messages = case['opposite_messages']
    for name in ('neutral', 'opposite'):
        if prompt_digest(case[name + '_messages']) != case[name + '_prompt_sha256']:
            raise ValueError('prepared conversation differs')
    natural = capture_step(runner, messages, circuit.NUMBER_STEP, capture_cache=True)
    donor = capture_step(runner, case['neutral_messages'], circuit.NUMBER_STEP, capture_cache=True)
    for name, result in (('neutral', donor), ('opposite', natural)):
        expected = runner.tokenizer.encode(str(case[name + '_p']), add_special_tokens=False)[0]
        if tuple(result['tokens'][:5]) != PREFIX_IDS or result['tokens'][5] != expected:
            raise ValueError('natural prefix or first number differs from the saved answer')
    recipient_token, donor_token = natural['tokens'][5], donor['tokens'][5]
    if recipient_token == donor_token:
        raise ValueError('frozen first-token pair is no longer distinct')
    source_state = circuit.prefix_states(runner, messages)[20]
    donor_state = circuit.prefix_states(runner, case['neutral_messages'])[20]
    caches = circuit.capture_natural_caches(runner, messages)
    routes = {}
    for condition in ('base', 'value21', 'later22_27', 'key_value21',
                      'key_value21_later22_27'):
        routes[condition] = circuit.run_condition(runner, messages, natural,
            source_state, donor_state, caches, recipient_token, donor_token, condition)
    position = runner._ids(messages).shape[-1] + 3
    source_value = natural['cache_v'][0, circuit.KV_GROUP, position, :].astype(np.float32)
    control = source_state + circuit.orthogonal(donor_state - source_state,
                                               orthogonal_seed.format(case_id=case['case_id']))
    routes['orthogonal'] = circuit.variant(runner, messages, natural, 20,
        source_state, control, source_value, recipient_token, donor_token, rescue_value=False)
    for name, layers in (('single26', (26,)), ('leave26_out', (22, 23, 24, 25, 27))):
        routes[name] = circuit.selective_rescue(runner, messages, natural,
            source_state, donor_state, caches, recipient_token, donor_token, layers)
    if set(routes) != set(ROUTES) or any(v['status'] != 'valid' for v in routes.values()):
        raise ValueError('route matrix or visible answer prefix differs')
    natural_margin = float(natural['scores'][recipient_token] - natural['scores'][donor_token])
    natural_raw = float(natural['raw_logits'][recipient_token] - natural['raw_logits'][donor_token])
    full = routes['key_value21_later22_27']
    if (full['first_token'] != recipient_token or full['processed_margin'] != natural_margin
            or full['raw_margin'] != natural_raw):
        raise ValueError('combined cache restoration fails exact natural replay')

    base, base_heads = circuit.run_with_heads(runner, messages, natural,
        source_state, donor_state, caches, recipient_token, donor_token, rescue_block26=False)
    restored, restored_heads = circuit.run_with_heads(runner, messages, natural,
        source_state, donor_state, caches, recipient_token, donor_token, rescue_block26=True)
    if not circuit.exact(base, routes['key_value21']) or not circuit.exact(restored, routes['single26']):
        raise ValueError('head scan differs from route anchors')
    reconstructed, _ = circuit.run_with_heads(runner, messages, natural,
        source_state, donor_state, caches, recipient_token, donor_token,
        rescue_block26=False, replacement={head: restored_heads[head] for head in range(28)})
    if not circuit.exact(reconstructed, restored):
        raise ValueError('all 28 heads fail exact cache-restoration reconstruction')
    heads = {}
    for head in range(28):
        output, _ = circuit.run_with_heads(runner, messages, natural, source_state,
            donor_state, caches, recipient_token, donor_token, rescue_block26=False,
            replacement={head: restored_heads[head]})
        heads[str(head)] = {
            'processed_margin_shift': output['processed_margin'] - base['processed_margin'],
            'raw_margin_shift': output['raw_margin'] - base['raw_margin'],
            'donor_first_token_match': output['donor_first_token_match'],
            'head_delta_norm': float(np.linalg.norm(restored_heads[head] - base_heads[head])),
        }

    kv_base, edited26, isolation_heads = circuit.capture_base(runner, messages, natural,
        source_state, donor_state, caches, recipient_token, donor_token)
    if not circuit.exact(kv_base, base):
        raise ValueError('K/V dissection base differs from the head scan')
    conditions = {}
    for name, use_k, use_v in (('edited_kv', False, False), ('natural_k', True, False),
                               ('natural_v', False, True), ('natural_kv', True, True)):
        mixed = dict(caches)
        mixed[26] = {'key': caches[26]['key'] if use_k else edited26['key'],
                     'value': caches[26]['value'] if use_v else edited26['value']}
        conditions[name] = circuit.run_isolated(runner, messages, natural,
            source_state, donor_state, mixed, recipient_token, donor_token, isolation_heads)
    if not circuit.exact(conditions['edited_kv'], base):
        raise ValueError('edited K/V no-op differs from the base')
    joint = conditions['natural_kv']
    head26 = heads['26']
    if (joint['processed_margin'] - base['processed_margin'] != head26['processed_margin_shift']
            or joint['raw_margin'] - base['raw_margin'] != head26['raw_margin_shift']
            or joint['donor_first_token_match'] != head26['donor_first_token_match']):
        raise ValueError('K/V restoration fails head-26 reconstruction')
    return {
        'case_id': case['case_id'], 'domain': case['domain'],
        'natural_processed_margin': natural_margin, 'natural_raw_margin': natural_raw,
        'routes': routes,
        'head_map': {'base': base, 'block26_rescue': restored,
                     'all_head_reconstruction': reconstructed, 'heads': heads},
        'kv': {'conditions': conditions,
               'key_delta_norm': float(np.linalg.norm(caches[26]['key'] - edited26['key'])),
               'value_delta_norm': float(np.linalg.norm(caches[26]['value'] - edited26['value']))},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--cohort', choices=('case_paired', 'initial_wording'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only-case', help='Run one selected case for implementation verification.')
    args = parser.parse_args()
    spec = json.loads(args.selection.read_text())
    cohort = spec['cohorts'][args.cohort]
    if (DEFAULT_MODEL, DEFAULT_REVISION) != (spec['model'], spec['revision']):
        raise ValueError('model revision differs')
    if spec['protocol'] != 'answer-prefix-circuit-v1' or digest(args.cases) != cohort['cases_sha256']:
        raise ValueError('prepared case file differs')
    cases = read_jsonl(args.cases)
    if [r['case_id'] for r in cases] != cohort['case_ids'] or len(cases) != 75:
        raise ValueError('case sample differs')
    if args.only_case:
        cases = [r for r in cases if r['case_id'] == args.only_case]
        if len(cases) != 1:
            raise ValueError('requested case is not selected')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': 'answer-prefix-circuit-v1', 'cohort': args.cohort,
        'cases_sha256': digest(args.cases), 'selection_sha256': digest(args.selection),
        'script_sha256': digest(Path(__file__)), 'circuit_code_sha256': digest(Path(circuit.__file__)),
        'model': DEFAULT_MODEL, 'revision': DEFAULT_REVISION,
        'case_ids': [r['case_id'] for r in cases], 'full_cohort_cases': 75,
        'route_conditions': list(ROUTES), 'heads': list(range(28)),
        'kv_conditions': list(KV_CONDITIONS)}
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('manifest differs from the existing run')
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    path = args.output / 'records.jsonl'
    previous = read_jsonl(path) if path.exists() else []
    done = {r['case_id'] for r in previous}
    if len(done) != len(previous) or not done <= set(manifest['case_ids']):
        raise ValueError('existing output IDs differ')
    runner = ForecastRunner(max_new_tokens=96)
    for case in cases:
        if case['case_id'] in done:
            continue
        append_jsonl(path, run_case(runner, case, cohort['orthogonal_seed_template']))
        done.add(case['case_id'])
        print(json.dumps({'cohort': args.cohort, 'completed': len(done), 'expected': len(cases)}), flush=True)
    if done == set(manifest['case_ids']):
        (args.output / 'completion.json').write_text(json.dumps({
            'total_records': len(done), 'records_sha256': digest(path),
            'manifest_sha256': digest(args.output / 'manifest.json')}, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
