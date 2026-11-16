"""Train, develop and confirm steering for forecast stability and responsiveness."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from forecast_mechanisms.core import read_jsonl, append_jsonl, first_messages, parse_response
from forecast_stability_protocol_v1 import (
    build_messages, contrast_messages, verdict_prefix,
    TRAIN_PRESSURE, DEV_CONDITIONS, TEST_CONDITIONS,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def freeze(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"frozen manifest differs: {path.name}")
    save(path, value)


def baseline_lookup(path):
    grouped = defaultdict(dict)
    for r in read_jsonl(path):
        if r['round'] == 2:
            if r['condition'] in grouped[r['case_id']]:
                raise ValueError('duplicate baseline case-condition')
            grouped[r['case_id']][r['condition']] = r
    return grouped


def derive(data, out, old_direction):
    import numpy as np
    from forecast_mechanisms.model import ForecastRunner, DEFAULT_MODEL, DEFAULT_REVISION
    train = read_jsonl(data / 'train-first.jsonl')
    baselines = baseline_lookup(data / 'train-baseline.jsonl')
    first_by_id = {r['case_id']: r for r in train}
    if len(first_by_id) != len(train) or not train or any(not r['parse_ok'] for r in train):
        raise ValueError('training first-answer collection is empty, duplicated or unparsed')
    out.mkdir(parents=True, exist_ok=True)
    block = 21
    manifest = {'protocol': 'forecast-stability-direction-v1', 'block': block,
        'train_sha256': sha(data / 'train-first.jsonl'),
        'train_baseline_sha256': sha(data / 'train-baseline.jsonl'),
        'protocol_sha256': sha(Path(__file__).with_name('forecast_stability_protocol_v1.py')),
        'script_sha256': sha(__file__), 'model': DEFAULT_MODEL, 'revision': DEFAULT_REVISION,
        'train_wordings': list(TRAIN_PRESSURE), 'old_direction_sha256': sha(old_direction)}
    freeze(out / 'manifest.json', manifest)
    path = out / 'pairs.jsonl'
    old = read_jsonl(path) if path.exists() else []
    done = {(r['case_id'], r['condition']) for r in old}
    expected_pairs = {(r['case_id'], condition) for r in train for condition in TRAIN_PRESSURE}
    if len(done) != len(old) or not done <= expected_pairs:
        raise ValueError('saved training pairs contain duplicate or unexpected keys')
    for row in old:
        first = first_by_id[row['case_id']]
        delta = np.asarray(row['delta'])
        if (row['domain'] != first['domain'] or row['first_verdict'] != first['verdict']
                or delta.shape != (3584,) or not np.isfinite(delta).all()):
            raise ValueError('saved training pair differs from its case or state shape')
    runner = ForecastRunner() if done != expected_pairs else None
    def state(messages, prefix):
        torch = runner.torch
        ids = runner._ids(messages)
        extra = runner.tokenizer.encode(prefix, add_special_tokens=False)
        ids = torch.cat((ids, torch.tensor([extra], device=ids.device)), dim=1)
        captured = []
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[0, -1].float().detach().cpu().numpy())
        handle = runner.model.model.layers[block].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                runner.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (3584,):
            raise ValueError('state hook invalid')
        return captured[0]
    started = time.monotonic()
    for first in train:
        for condition in TRAIN_PRESSURE:
            key = (first['case_id'], condition)
            if key in done:
                continue
            positive, negative = contrast_messages(first, condition, baseline=baselines[first['case_id']])
            diff = state(positive, verdict_prefix(first)) - state(negative, verdict_prefix(first))
            append_jsonl(path, {'case_id': first['case_id'], 'domain': first['domain'],
                'first_verdict': first['verdict'], 'condition': condition, 'delta': diff.tolist()})
            done.add(key)
        if len(done) % 30 == 0:
            print(json.dumps({'derived_pairs': len(done), 'elapsed_s': time.monotonic()-started}), flush=True)
    rows = read_jsonl(path)
    if len(rows) != len(expected_pairs) or {(r['case_id'],r['condition']) for r in rows} != expected_pairs:
        raise ValueError('direction pairs incomplete')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['domain'], r['first_verdict'])].append(r['delta'])
    if {key[1] for key in groups} != {'accept', 'reject'}:
        raise ValueError('training lacks initial-verdict classes')
    # Equal total weight per initial verdict; equal nonempty domains within it.
    mean = np.mean([np.mean([np.mean(v, axis=0) for key,v in groups.items()
        if key[1] == verdict], axis=0) for verdict in ('accept','reject')], axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('degenerate mean')
    legacy = np.load(old_direction)
    amount = float(legacy['high_dose'])
    np.savez(out / 'directions.npz', instruction_unit=(mean/norm).astype(np.float32),
        legacy_unit=legacy['paired_neutral'], dose=amount, block=block)
    save(out / 'completion.json', {'pairs_sha256': sha(path),
        'directions_sha256': sha(out / 'directions.npz'), 'n_pairs': len(rows),
        'group_counts': {':'.join(k):len(v) for k,v in groups.items()},
        'raw_mean_norm': float(norm), 'fixed_reference_dose': amount})


def verdict_only(runner, messages, delta, baseline):
    import numpy as np
    from forecast_mechanisms.core import parse_response
    if not baseline['parse_ok'] or baseline['response'].count('\nVerdict:') != 1:
        return {'response':baseline['response'],'parse_ok':False,'p_accept':None,
            'verdict':None,'edited_generation_steps':0,'mean_relative_dose':None,
            'failure':'unparseable_base_prefix'}
    torch = runner.torch
    # Greedy output before the verdict is held fixed at the unsteered run.
    prefix = baseline['response'].split('\nVerdict:')[0] + '\nVerdict:'
    tokens = runner.tokenizer.encode(prefix, add_special_tokens=False)
    ids = runner._ids(messages)
    full = torch.cat((ids, torch.tensor([tokens], device=ids.device)), dim=1)
    count = 0
    relative = None
    dt = torch.tensor(delta, device=ids.device, dtype=torch.bfloat16)
    def hook(_m, _i, output):
        nonlocal count, relative
        count += 1
        if count != 1:
            return output
        h = output[0] if isinstance(output, tuple) else output
        relative = float(np.linalg.norm(delta)/torch.linalg.vector_norm(h[0,-1].float()).item())
        changed = h.clone(); changed[0,-1] += dt
        return (changed,*output[1:]) if isinstance(output,tuple) else changed
    handle=runner.model.model.layers[21].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            generated=runner.model.generate(input_ids=full,attention_mask=torch.ones_like(full),
                max_new_tokens=24,do_sample=False,pad_token_id=runner.tokenizer.eos_token_id)
    finally:
        handle.remove()
    tail=runner.tokenizer.decode(generated[0,full.shape[-1]:],skip_special_tokens=True)
    response=(prefix+tail).strip()
    parsed=parse_response(response,'p_first')
    if parsed['parse_ok'] and parsed['p_accept'] != baseline['p_accept']:
        raise ValueError('verdict-only intervention changed the written percentage')
    return {'response':response,**parsed,'edited_generation_steps':1,'mean_relative_dose':relative}


DEV_ARMS = ('zero','reminder','legacy','new_half','new_one','new_two','new_verdict')


def evaluate(data, direction_dir, out, phase, choice=None, max_cases=None):
    import numpy as np
    from forecast_mechanisms.model import ForecastRunner, DEFAULT_MODEL, DEFAULT_REVISION
    from forecast_mechanisms.steering import generate_with_vector, orthogonal_control
    if phase not in ('dev', 'test') or max_cases is not None and max_cases < 1:
        raise ValueError('invalid evaluation phase or case limit')
    if phase == 'test' and (not isinstance(choice, dict) or choice.get('arm') not in
                           ('new_half','new_one','new_two','new_verdict')):
        raise ValueError('test requires a frozen development candidate')
    arrays=np.load(direction_dir/'directions.npz')
    if sha(direction_dir/'directions.npz') != json.loads((direction_dir/'completion.json').read_text())['directions_sha256']:
        raise ValueError('direction checksum changed')
    amount=float(arrays['dose'])
    if int(arrays['block'])!=21:
        raise ValueError('wrong steering block')
    for name in ('instruction_unit','legacy_unit'):
        vector=arrays[name]
        if (vector.shape != (3584,) or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector),1.,rtol=1e-5,atol=1e-6)):
            raise ValueError('invalid steering direction')
    if not np.isfinite(amount) or amount <= 0:
        raise ValueError('invalid steering dose')
    conditions=DEV_CONDITIONS if phase=='dev' else TEST_CONDITIONS
    arms=DEV_ARMS if phase=='dev' else ('zero','reminder','legacy','legacy_orthogonal','selected','orthogonal')
    first_path=data/('dev-first.jsonl' if phase=='dev' else 'test-first.jsonl')
    runner=None
    if phase=='test' and not (data/'test-first-completion.json').exists():
        # First answers are collected once, before any test intervention, and never used in selection.
        output=read_jsonl(first_path) if first_path.exists() else []
        completed_ids={r['case_id'] for r in output}
        cases=read_jsonl(data/'test-cases.jsonl')
        cases_by_id={r['case_id']:r for r in cases}
        if (len(cases_by_id)!=len(cases) or len(completed_ids)!=len(output)
                or not completed_ids <= set(cases_by_id)):
            raise ValueError('test case or saved initial-answer IDs are duplicated or unexpected')
        for row in output:
            case=cases_by_id[row['case_id']]
            if row['domain']!=case['domain'] or row['messages']!=first_messages(case,'p_first','first_person'):
                raise ValueError('saved initial answer does not match its case')
        if completed_ids != set(cases_by_id):
            runner=ForecastRunner()
        for case in cases:
            if case['case_id'] in completed_ids:
                continue
            messages=first_messages(case,'p_first','first_person')
            result=generate_with_vector(runner,messages,21,np.zeros(3584,dtype=np.float32))
            row={'case_id':case['case_id'],'domain':case['domain'],'messages':messages,**result}
            append_jsonl(first_path,row)
            output.append(row)
        save(data/'test-first-completion.json',{'sha256':sha(first_path),'n':len(output)})
    first_rows=read_jsonl(first_path)
    first_by_id={r['case_id']:r for r in first_rows}
    if not first_rows or len(first_by_id)!=len(first_rows):
        raise ValueError('initial-answer collection is empty or duplicated')
    for row in first_rows:
        parsed=parse_response(row['response'],'p_first')
        if any(row.get(key)!=parsed[key] for key in ('parse_ok','p_accept','verdict')):
            raise ValueError('saved initial parse differs from its response')
    if phase=='test':
        marker=json.loads((data/'test-first-completion.json').read_text())
        expected_ids={r['case_id'] for r in read_jsonl(data/'test-cases.jsonl')}
        if (marker['sha256']!=sha(first_path) or marker['n']!=len(expected_ids)
                or len(first_rows)!=len(expected_ids)
                or {r['case_id'] for r in first_rows}!=expected_ids):
            raise ValueError('test initial collection hash or IDs changed')
        cases_by_id={r['case_id']:r for r in read_jsonl(data/'test-cases.jsonl')}
        for row in first_rows:
            case=cases_by_id[row['case_id']]
            if row['domain']!=case['domain'] or row['messages']!=first_messages(case,'p_first','first_person'):
                raise ValueError('initial answer does not match the frozen test case')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'protocol':'forecast-stability-evaluation-v1','phase':phase,
        'first_sha256':sha(first_path),'direction_sha256':sha(direction_dir/'directions.npz'),
        'case_inputs_sha256':sha(data/'test-cases.jsonl') if phase=='test' else sha(data/'dev-baseline.jsonl'),
        'protocol_sha256':sha(Path(__file__).with_name('forecast_stability_protocol_v1.py')),
        'script_sha256':sha(__file__),'model':DEFAULT_MODEL,'revision':DEFAULT_REVISION,
        'conditions':list(conditions),'arms':list(arms),'choice':choice,
        'case_ids':[r['case_id'] for r in first_rows]}
    freeze(out/'manifest.json',manifest)
    records=out/'records.jsonl'
    existing=read_jsonl(records) if records.exists() else []
    done={(r['case_id'],r['condition']) for r in existing}
    expected_keys={(r['case_id'],c) for r in first_rows if r['parse_ok'] for c in conditions}
    if len(done)!=len(existing) or not done <= expected_keys:
        raise ValueError('saved evaluation contains duplicate or unexpected case-condition keys')
    for row in existing:
        first=first_by_id[row['case_id']]
        if (set(row['results'])!=set(arms) or row['domain']!=first['domain']
                or row['first_p']!=first['p_accept'] or row['first_verdict']!=first['verdict']):
            raise ValueError('saved evaluation row differs from its case or intervention arms')
    if done != expected_keys and runner is None:
        runner=ForecastRunner()
    lookup=baseline_lookup(data/'dev-baseline.jsonl') if phase=='dev' else {}
    started=time.monotonic(); completed=0
    for first in first_rows:
        if not first['parse_ok']:
            continue
        for condition in conditions:
            if (first['case_id'],condition) in done:
                continue
            messages=build_messages(first,condition,baseline=lookup.get(first['case_id']))
            results={}
            for arm in arms:
                mode=arm
                if arm=='legacy_orthogonal':
                    mode='legacy'
                if arm in ('selected','orthogonal'):
                    mode=choice['arm']
                instruction=arm=='reminder'
                current=build_messages(first,condition,instruction='robust',baseline=lookup.get(first['case_id'])) if instruction else messages
                unit=np.asarray(arrays['legacy_unit'] if mode=='legacy' else arrays['instruction_unit'],dtype=np.float32)
                factor={'new_half':.5,'new_one':1.,'new_two':2.,'new_verdict':1.,'legacy':1.}.get(mode,0.)
                if arm in ('orthogonal','legacy_orthogonal'):
                    unit=orthogonal_control(unit,first['case_id'],'forecast-stability')
                delta=unit*amount*factor
                if mode=='new_verdict':
                    result=verdict_only(runner,current,delta,results['zero'])
                else:
                    result=generate_with_vector(runner,current,21,delta)
                results[arm]=result
                if phase=='dev' and arm=='zero' and condition in ('neutral','original'):
                    key='opposite' if condition=='original' else 'neutral'
                    if result['response']!=lookup[first['case_id']][key]['response']:
                        raise ValueError('zero-dose replay differs from frozen baseline')
            append_jsonl(records,{'case_id':first['case_id'],'domain':first['domain'],
                'first_p':first['p_accept'],'first_verdict':first['verdict'],
                'condition':condition,'results':results})
        completed+=1
        print(json.dumps({'phase':phase,'new_cases':completed,'last_case':first['case_id'],
            'elapsed_s':time.monotonic()-started}),flush=True)
        if max_cases and completed>=max_cases:
            return
    expected=sum(r['parse_ok'] for r in first_rows)*len(conditions)
    actual=read_jsonl(records)
    if len(actual)!=expected or {(r['case_id'],r['condition']) for r in actual}!=expected_keys:
        raise ValueError('evaluation incomplete or duplicate')
    save(out/'completion.json',{'records_sha256':sha(records),'n_records':expected,
        'initial_cases':len(first_rows),'first_parse_failures':sum(not r['parse_ok'] for r in first_rows)})


def verify_direction(path, reference):
    """Check array content against the frozen numerical reference."""
    import numpy as np
    expected_keys={'instruction_unit','legacy_unit','dose','block'}
    actual=np.load(path,allow_pickle=False)
    target=np.load(reference,allow_pickle=False)
    if set(actual.files)!=expected_keys or set(target.files)!=expected_keys:
        raise ValueError('direction archive fields differ')
    for archive in (actual,target):
        for name in ('instruction_unit','legacy_unit'):
            vector=archive[name]
            if (vector.shape!=(3584,) or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector),1.,rtol=1e-5,atol=1e-6)):
                raise ValueError('invalid direction shape, values or norm')
        if int(archive['block'])!=21 or not np.isfinite(archive['dose']) or float(archive['dose'])<=0:
            raise ValueError('invalid block or reference dose')
    for name in expected_keys:
        np.testing.assert_allclose(actual[name],target[name],rtol=1e-6,atol=1e-6,
            err_msg=f'reconstructed {name} differs from reference')


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest='command',required=True)
    for command in ('derive','dev','test'):
        sub=commands.add_parser(command)
        sub.add_argument('--data',type=Path,required=True)
        sub.add_argument('--output',type=Path,required=True)
        if command=='derive':
            sub.add_argument('--old-direction',type=Path,required=True)
            sub.add_argument('--reference',type=Path)
        else:
            sub.add_argument('--directions',type=Path,required=True)
            sub.add_argument('--max-cases',type=int)
            if command=='test':
                sub.add_argument('--selection',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='derive':
        derive(a.data,a.output,a.old_direction)
        if a.reference:
            verify_direction(a.output/'directions.npz',a.reference)
    else:
        choice=json.loads(a.selection.read_text()) if a.command=='test' else None
        evaluate(a.data,a.directions,a.output,a.command,choice,a.max_cases)


if __name__=='__main__':main()
