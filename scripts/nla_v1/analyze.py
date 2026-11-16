"""Reconstruction alignment and fixed lexical checks for matched NLA readouts."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from forecast_mechanisms.core import read_jsonl, write_jsonl, verify_complete_run
from run import PHASES, EXPLANATION, normalized, prepared, sha, save


FIRST_NUMBER = re.compile(r'\b(\d{1,3})\s*%')
PATTERNS = {
    'format_or_score': re.compile(r'\b(format|score|metric|output|token)\b',re.I),
    'numeric_percentage': re.compile(r'\b\d{1,3}\s*%'),
    'explicit_feedback': re.compile(r'\b(reassess|reconsider|disagree|pushback|definitely|no chance)\b|\bI thought\b',re.I),
}


def ci(values):
    if not values:
        return None
    rng=random.Random(42)
    draws=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(10000))
    return [draws[250],draws[9750]]


def mean(values):
    return statistics.mean(values) if values else None


def analyze(out, run):
    spec, key_rows=prepared(out)
    completion=verify_complete_run(run)
    if completion['records_sha256']!=spec['source_records_sha256']:
        raise ValueError('training baseline differs from the NLA source collection')
    source=[r for r in read_jsonl(run/'records.jsonl') if r['arm']=='p_first' and r['parse_ok']]
    first={(r['split'],r['case_id']):r for r in source if r['round']==1}
    train=defaultdict(list)
    for row in source:
        if row['split']=='train' and row['round']==2 and row['condition'] in ('neutral','opposite'):
            train[row['domain'],row['condition'],first['train',row['case_id']]['verdict']].append(row['p_accept'])
    training_means={group:statistics.mean(values) for group,values in train.items()}
    key={r['vector_id']:r for r in key_rows}
    vectors=np.load(out/'vectors.npy',allow_pickle=False)
    directions=np.load(out/'directions.npy',allow_pickle=False)
    case_order=[r['case_id'] for r in key_rows if r['condition']=='neutral']
    result={'protocol':'matched-nla-summary-v1','phases':{},
        'lexical_audit':'Fixed regular expressions on explanation text; no human or semantic annotations.',
        'lexical_patterns':{name:p.pattern for name,p in PATTERNS.items()},
        'source_records_sha256':completion['records_sha256'],'selection_sha256':sha(out/'selection.json'),
        'analysis_script_sha256':sha(__file__)}
    all_cases=[]
    for phase in PHASES:
        actor=read_jsonl(out/f'{phase}-actor.jsonl')
        expected={r['vector_id'] for r in key_rows if r['phase']==phase}
        if len(actor)!=24 or {r['vector_id'] for r in actor}!=expected:
            raise ValueError('actor phase is incomplete or duplicated')
        for row in actor:
            match=EXPLANATION.search(row['raw'])
            parsed=match.group(1).strip() if match else None
            if row['explanation']!=parsed:
                raise ValueError('stored explanation differs from its raw actor output')
        descriptions={r['vector_id']:r['explanation'] for r in actor}
        recon=np.load(out/f'{phase}-reconstruction.npy',allow_pickle=False)
        if recon.shape!=vectors.shape:
            raise ValueError('reconstruction shape differs from prepared vectors')
        grouped=defaultdict(dict)
        for row in key_rows:
            if row['phase']==phase:
                grouped[row['case_id']][row['condition']]=row
        cases=[]
        for case_id,pair in grouped.items():
            a,b=pair['neutral'],pair['opposite']
            i,j=(int(r['vector_id'][1:]) for r in (a,b))
            shared,orth=directions[case_order.index(case_id)]
            valid=(descriptions[a['vector_id']] is not None and descriptions[b['vector_id']] is not None
                   and np.isfinite(recon[[i,j]]).all())
            row={'case_id':case_id,'domain':a['domain'],'first_verdict':a['first_verdict'],
                 'output_p_shift':b['p_accept']-a['p_accept'],'parsed_pair':bool(valid)}
            if valid:
                vn,vo=normalized(vectors[i]),normalized(vectors[j])
                rn,ro=normalized(recon[i]),normalized(recon[j])
                row.update({'pair_margin':float((ro-rn)@(vo-vn)),
                    'reconstruction_shared_projection':float((ro-rn)@shared),
                    'reconstruction_orthogonal_projection':float((ro-rn)@orth),
                    'actual_shared_projection':float((vo-vn)@shared),
                    'same_text':descriptions[a['vector_id']]==descriptions[b['vector_id']]})
            cases.append(row)
        lexical={name:sum(bool(pattern.search(r['explanation'] or '')) for r in actor)
                 for name,pattern in PATTERNS.items()}
        number_pairs=defaultdict(dict)
        for row in actor:
            meta=key[row['vector_id']]
            match=FIRST_NUMBER.search(row['explanation'] or '')
            number=int(match.group(1)) if match else None
            baseline=training_means[meta['domain'],meta['condition'],meta['first_verdict']]
            number_pairs[meta['case_id']][meta['condition']]=(number,meta['p_accept'],baseline)
        number_audit=[]
        for pair in number_pairs.values():
            a,b=pair['neutral'],pair['opposite']
            if a[0] is not None and b[0] is not None:
                number_audit.append({'text_change':b[0]-a[0],'output_change':b[1]-a[1],
                                     'baseline_change':b[2]-a[2]})
        valid_cases=[r for r in cases if r['parsed_pair']]
        margins=[r['pair_margin'] for r in valid_cases]
        shared=[r['reconstruction_shared_projection'] for r in valid_cases]
        orth=[r['reconstruction_orthogonal_projection'] for r in valid_cases]
        advantage=[abs(r['baseline_change']-r['output_change'])-abs(r['text_change']-r['output_change'])
                   for r in number_audit]
        result['phases'][phase]={'n_cases':len(cases),'parsed_pairs':len(valid_cases),
            'n_descriptions':len(actor),'parsed_descriptions':sum(r['explanation'] is not None for r in actor),
            'lexical_counts':lexical,'pair_margin_mean':mean(margins),
            'pair_margin_positive':sum(x>0 for x in margins),'pair_margin_ci95':ci(margins),
            'shared_projection_mean':mean(shared),'shared_projection_positive':sum(x>0 for x in shared),
            'shared_projection_ci95':ci(shared),'orthogonal_projection_mean':mean(orth),
            'n_first_number_pairs':len(number_audit),
            'first_number_change_mae':mean([abs(r['text_change']-r['output_change']) for r in number_audit]),
            'train_group_change_mae':mean([abs(r['baseline_change']-r['output_change']) for r in number_audit]),
            'first_number_mae_improvement_ci95':ci(advantage),
            'actor_sha256':sha(out/f'{phase}-actor.jsonl'),
            'reconstruction_sha256':sha(out/f'{phase}-reconstruction.npy')}
        write_jsonl(out/f'{phase}-analysis-cases.jsonl',cases)
        all_cases.extend(valid_cases)
    result['pooled']={'n_cases':24,'parsed_pairs':len(all_cases),
        'pair_margin_mean':mean([r['pair_margin'] for r in all_cases]),
        'pair_margin_positive':sum(r['pair_margin']>0 for r in all_cases),
        'shared_projection_mean':mean([r['reconstruction_shared_projection'] for r in all_cases]),
        'shared_projection_positive':sum(r['reconstruction_shared_projection']>0 for r in all_cases),
        'orthogonal_projection_mean':mean([r['reconstruction_orthogonal_projection'] for r in all_cases])}
    save(out/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.output,args.run)
    print(json.dumps({'cases':24,'positive_pair_margins':result['pooled']['pair_margin_positive'],
                     'lexical_counts':{p:result['phases'][p]['lexical_counts'] for p in PHASES}},sort_keys=True))


if __name__=='__main__':main()
