"""Matched NLA descriptions and reconstructions of neutral/pushback states."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from forecast_mechanisms.core import read_jsonl, write_jsonl, verify_complete_run
from forecast_mechanisms.model import DEFAULT_MODEL, DEFAULT_REVISION, EXPECTED_HIDDEN, NLA_LAYER
from forecast_mechanisms.nla_readout import _load_official_client


PHASES = ('pilot', 'replication')
EXPLANATION = re.compile(r'<explanation>\s*(.*?)\s*</explanation>', re.S)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('saved inputs or settings differ; use a new output directory')
    save(path, value)


def load_spec(path):
    spec = json.loads(path.read_text())
    cases = spec['cases']
    if (spec['protocol'] != 'matched-nla-v1' or
            (spec['target_model'], spec['target_revision']) != (DEFAULT_MODEL, DEFAULT_REVISION) or
            (spec['block'], spec['hidden_size'], spec['position'], spec['prompt_voice']) !=
            (NLA_LAYER, EXPECTED_HIDDEN, 'pre_number', 'third_person') or
            len(cases) != 24 or len({r['case_id'] for r in cases}) != 24 or
            any(sum(r['phase'] == phase for r in cases) != 12 for phase in PHASES)):
        raise ValueError('frozen NLA selection or target model differs')
    return spec


def normalized(value):
    length = np.linalg.norm(value)
    if not np.isfinite(length) or length <= 0:
        raise ValueError('cannot normalize a nonfinite or zero vector')
    return value / length


def state(trace_dir, row):
    path = (trace_dir / row['trace_file']).resolve()
    if not path.is_relative_to(trace_dir.resolve()):
        raise ValueError('trace array is outside the trace directory')
    with np.load(path, allow_pickle=False) as data:
        value = data['pre_number'][NLA_LAYER + 1].astype(np.float32)
    if value.shape != (EXPECTED_HIDDEN,) or not np.isfinite(value).all():
        raise ValueError('invalid NLA target state')
    return value


def guarded_bytes(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f'prepared file differs: {path.name}')
    if not path.exists():
        path.write_bytes(content)


def prepare(run, traces, selection, out):
    spec = load_spec(selection)
    marker = verify_complete_run(run)
    trace_manifest = json.loads((traces / 'manifest.json').read_text())
    if (marker['records_sha256'] != spec['source_records_sha256'] or
            trace_manifest['source_records_sha256'] != marker['records_sha256'] or
            sha(traces / 'manifest.json') != spec['trace_manifest_sha256']):
        raise ValueError('source collection or third-person trace revision differs')
    rows = read_jsonl(traces / 'traces.jsonl')
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('duplicate trace keys')
    first = {(r['split'], r['case_id']): r for r in rows
             if r['round'] == 1 and r['arm'] == 'p_first'}
    pairs = defaultdict(dict)
    for row in rows:
        if row['round'] == 2 and row['arm'] == 'p_first' and row.get('condition') in ('neutral','opposite'):
            pairs[row['split'], row['case_id']][row['condition']] = row
    train = defaultdict(list)
    for (split, case_id), pair in pairs.items():
        if set(pair) != {'neutral','opposite'}:
            raise ValueError('incomplete neutral/pushback trace pair')
        if split == 'train':
            delta = state(traces, pair['opposite']) - state(traces, pair['neutral'])
            train[first[split, case_id]['verdict']].append(normalized(delta))
    shared = {verdict: normalized(np.mean(units, axis=0)) for verdict, units in train.items()}
    key, vectors, directions = [], [], []
    for selected in spec['cases']:
        case_id = selected['case_id']
        pair = pairs['test', case_id]
        initial = first['test', case_id]
        neutral, opposite = (state(traces, pair[c]) for c in ('neutral','opposite'))
        common = shared[initial['verdict']]
        own = normalized(opposite - neutral)
        residual = normalized(own - np.dot(own, common) * common)
        seed = int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)
        orth = np.random.default_rng(seed).normal(size=EXPECTED_HIDDEN).astype(np.float32)
        orth -= np.dot(orth, common) * common + np.dot(orth, residual) * residual
        orth = normalized(orth)
        directions.append(np.stack([common, orth]))
        for condition, vector in (('neutral', neutral), ('opposite', opposite)):
            source = pair[condition]
            key.append({'vector_id':f'V{len(vectors):04d}', 'case_id':case_id,
                'domain':source['domain'], 'phase':selected['phase'], 'condition':condition,
                'first_verdict':initial['verdict'], 'p_accept':source['p_accept'], 'verdict':source['verdict']})
            vectors.append(vector)
    out.mkdir(parents=True, exist_ok=True)
    payloads = {'key.jsonl':''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in key).encode()}
    for name, values in (('vectors.npy', vectors), ('directions.npy', directions)):
        buffer = io.BytesIO(); np.save(buffer, np.stack(values), allow_pickle=False)
        payloads[name] = buffer.getvalue()
    digests = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    for name in ('key.jsonl', 'vectors.npy'):
        if digests[name] != spec['prepared_sha256'][name]:
            raise ValueError(f'reconstructed NLA input differs from reference: {name}')
    freeze(out / 'manifest.json', {'protocol':'matched-nla-preparation-v1',
        'selection_sha256':sha(selection), 'script_sha256':sha(__file__),
        'source_records_sha256':marker['records_sha256'],
        'trace_manifest_sha256':sha(traces/'manifest.json'),
        # Float32 projection reductions can differ slightly across BLAS implementations.
        'prepared_sha256':digests, 'cases':24, 'vectors':48})
    for name, content in payloads.items():
        guarded_bytes(out / name, content)
    guarded_bytes(out / 'selection.json', selection.read_bytes())
    print(json.dumps({'cases':24,'vectors':48,'reference_inputs_match':True,
                      'reference_directions_bitwise_match':digests['directions.npy']==spec['prepared_sha256']['directions.npy']}))


def prepared(out):
    spec = load_spec(out / 'selection.json')
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['selection_sha256'] != sha(out / 'selection.json'):
        raise ValueError('NLA selection changed')
    for name in ('key.jsonl', 'vectors.npy'):
        if manifest['prepared_sha256'][name] != spec['prepared_sha256'][name]:
            raise ValueError(f'prepared NLA reference differs: {name}')
    for name, digest in manifest['prepared_sha256'].items():
        if sha(out / name) != digest:
            raise ValueError(f'NLA input changed: {name}')
    return spec, read_jsonl(out/'key.jsonl')


def official_client(spec, module, checkpoint, role):
    settings = spec[role]
    sidecar = checkpoint / 'nla_meta.yaml'
    if sha(sidecar) != settings['sidecar_sha256']:
        raise ValueError(f'{role} metadata differs from its pinned checkpoint')
    metadata = yaml.safe_load(sidecar.read_text())
    roles = ('av',) if role == 'actor' else ('ar','critic')
    if metadata.get('role') not in roles or metadata.get('d_model') != EXPECTED_HIDDEN:
        raise ValueError(f'{role} role or hidden dimension is incompatible')
    if sha(module) != spec['official_client']['module_sha256']:
        raise ValueError('official NLA inference source differs from the pinned commit')
    return _load_official_client(module)


def decode(out, phase, actor, module, server_url):
    spec, key = prepared(out)
    official = official_client(spec,module,actor,'actor')
    output = out / f'{phase}-actor.jsonl'
    settings = {'action':'actor','phase':phase,'selection_sha256':sha(out/'selection.json'),
        'actor':spec['actor'],'official_client':spec['official_client'],
        'temperature':0,'max_new_tokens':200,'script_sha256':sha(__file__)}
    marker = out / f'{phase}-actor-manifest.json'
    if output.exists() and not marker.exists():
        raise ValueError('saved actor output has no matching run manifest')
    freeze(marker, settings)
    previous = read_jsonl(output) if output.exists() else []
    completion = out / f'{phase}-actor-completion.json'
    if completion.exists() and json.loads(completion.read_text()) != {
            'records_sha256':sha(output), 'descriptions':len(previous)}:
        raise ValueError('completed actor output changed')
    done = {r['vector_id'] for r in previous}
    expected = {r['vector_id'] for r in key if r['phase'] == phase}
    if len(done) != len(previous) or not done <= expected:
        raise ValueError('saved actor output has duplicate or unexpected vector IDs')
    order = [r for r in key if r['phase'] == phase]
    random.Random(spec['seed'] + PHASES.index(phase)).shuffle(order)
    if done != expected:
        client = official.NLAClient(actor, sglang_url=server_url)
        vectors = np.load(out/'vectors.npy', mmap_mode='r', allow_pickle=False)
        for row in order:
            if row['vector_id'] in done:
                continue
            raw = client.generate(np.asarray(vectors[int(row['vector_id'][1:])]),
                temperature=0,max_new_tokens=200,extract_explanation=False)
            match = EXPLANATION.search(raw)
            explanation = match.group(1).strip() if match else None
            with output.open('a') as handle:
                handle.write(json.dumps({'vector_id':row['vector_id'],'explanation':explanation,'raw':raw})+'\n')
    final = read_jsonl(output)
    write_jsonl(out/f'{phase}-blind.jsonl', [{'readout_id':f'R{i:04d}','text':r['explanation']}
                                            for i,r in enumerate(final)])
    save(out/f'{phase}-actor-completion.json',{'records_sha256':sha(output),'descriptions':len(final)})
    print(json.dumps({'phase':phase,'descriptions':len(final)}))


def reconstruct(out, phase, checkpoint, module, device):
    spec, key = prepared(out)
    official = official_client(spec,module,checkpoint,'reconstructor')
    actor = read_jsonl(out/f'{phase}-actor.jsonl')
    expected = {r['vector_id'] for r in key if r['phase'] == phase}
    if len(actor)!=24 or {r['vector_id'] for r in actor}!=expected:
        raise ValueError('actor phase is incomplete or duplicated')
    completion = json.loads((out/f'{phase}-actor-completion.json').read_text())
    if completion != {'records_sha256':sha(out/f'{phase}-actor.jsonl'),'descriptions':len(actor)}:
        raise ValueError('completed actor output changed')
    freeze(out/f'{phase}-reconstruction-manifest.json',{'action':'reconstruction','phase':phase,
        'actor_sha256':sha(out/f'{phase}-actor.jsonl'),'reconstructor':spec['reconstructor'],
        'official_client':spec['official_client'],'script_sha256':sha(__file__)})
    critic = official.NLACritic(checkpoint,device=device)
    recon = np.full((len(key),EXPECTED_HIDDEN),np.nan,dtype=np.float32)
    for row in actor:
        if row['explanation'] is None:
            continue
        vector = critic.reconstruct(row['explanation']).detach().cpu().numpy().astype(np.float32)
        if vector.shape!=(EXPECTED_HIDDEN,) or not np.isfinite(vector).all():
            raise ValueError('reconstructor returned an invalid activation')
        recon[int(row['vector_id'][1:])] = vector
    path=out/f'{phase}-reconstruction.npy'
    np.save(path,recon,allow_pickle=False)
    save(out/f'{phase}-reconstruction-completion.json',{'reconstruction_sha256':sha(path),
        'parsed_descriptions':sum(r['explanation'] is not None for r in actor)})
    print(json.dumps({'phase':phase,'reconstructed':int(np.isfinite(recon[:,0]).sum())}))


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest='action',required=True)
    prepare_parser=commands.add_parser('prepare')
    for name in ('run','traces','selection','output'):
        prepare_parser.add_argument('--'+name,type=Path,required=True)
    for action in ('decode','reconstruct'):
        sub=commands.add_parser(action)
        sub.add_argument('--output',type=Path,required=True)
        sub.add_argument('--phase',choices=PHASES,required=True)
        sub.add_argument('--official-module',type=Path,required=True)
        if action=='decode':
            sub.add_argument('--actor',type=Path,required=True)
            sub.add_argument('--server-url',default='http://localhost:30000')
        else:
            sub.add_argument('--reconstructor',type=Path,required=True)
            sub.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.run,args.traces,args.selection,args.output)
    elif args.action=='decode':decode(args.output,args.phase,args.actor,args.official_module,args.server_url)
    else:reconstruct(args.output,args.phase,args.reconstructor,args.official_module,args.device)


if __name__=='__main__':main()
