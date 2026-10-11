#!/usr/bin/env python3
"""Analyze retained arrival/terminal journals without assuming unknown usage or cache hits."""
import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def quantile(xs,q):
    if not xs:return None
    xs=sorted(xs);index=(len(xs)-1)*q
    low=int(index);high=math.ceil(index)
    return xs[low]+(xs[high]-xs[low])*(index-low)


def analyze(path):
    header=None;arrivals={};terminals={};summary=None
    for i,line in enumerate(Path(path).read_text().splitlines(),1):
        row=json.loads(line)
        kind=row['type']
        if kind=='block':
            if header is not None:raise ValueError('Duplicate block header')
            header=row
        elif kind in ('arrival','terminal'):
            target=arrivals if kind=='arrival' else terminals
            key=row['arrival_id']
            if key in target:raise ValueError(f'Duplicate {kind} {key} at line{i}')
            target[key]=row
        elif kind=='summary':
            if summary is not None:raise ValueError('Duplicate block summary')
            summary=row
    if header is None:raise ValueError('Missing block header')
    orphan=set(terminals)-set(arrivals)
    pending=set(arrivals)-set(terminals)
    if orphan:raise ValueError('Terminal without arrival')
    rows=list(terminals.values())
    outcomes=Counter(row['result']['status'] for row in rows)
    complete=[row for row in rows if row['result']['status']=='complete']
    ttf=[row['result']['arrival_ttft_s'] for row in complete if row['result'].get('arrival_ttft_s') is not None]
    tp=[row['result']['tpot_s'] for row in complete if row['result'].get('tpot_s') is not None]
    usage=[row for row in complete if row['result'].get('usage_complete')]
    cache_observed=[row for row in complete if isinstance(row['result'].get('cached_tokens'),int)]
    durations=header['duration_s']
    slices={}
    for label,predicate in [('target2K',lambda row:row['target_prompt_tokens_estimate']==2048),
                            ('target8K',lambda row:row['target_prompt_tokens_estimate']==8192),
                            ('intended_cold',lambda row:row['cold']),('shared',lambda row:not row['cold'])]:
        selected=[row for row in rows if predicate(row)]
        slices[label]={'offered':sum(predicate(row) for row in arrivals.values()),
                       'terminal':len(selected),'complete':sum(row['result']['status']=='complete' for row in selected),
                       'slo_pass':sum(row['result'].get('slo_pass',False) for row in selected)}
    return {'file':str(path),'policy':header['policy'],'seed':header['seed'],'rate':header['rate'],
            'endpoints':header['endpoints'],'planned_seconds':durations,'offered':len(arrivals),
            'terminal':len(terminals),'unresolved_arrivals':len(pending),'outcomes':dict(outcomes),
            'summary_present':summary is not None,'complete':len(complete),
            'stopped_by_lease':bool(summary and summary.get('stopped_by_lease')),
            'actual_exposure_seconds':durations if summary and not summary.get('stopped_by_lease') and summary.get('wall_seconds',0)>=durations else None,
            'exposure_lower_bound_seconds':max((row.get('observed_arrival_offset_s',row.get('arrival_offset_s',0)) for row in arrivals.values()),default=0),
            'exposure_upper_bound_seconds':min(durations,summary['wall_seconds']) if summary else None,
            'exposure_scope':'Full unrevoked window exact planned duration; truncated source-v3 has no exact admission-close clock, so only arrival/wall bounds are available.',
            'full_planned_window_qualified':bool(summary and not summary.get('stopped_by_lease') and summary.get('wall_seconds',0)>=durations and not pending),
            'complete_with_done_and_finish':sum(bool(row['result'].get('done') and row['result'].get('finish_reason')) for row in complete),
            'complete_with_usage':len(usage),'complete_correct_json':sum(row['result'].get('needle_correct',False) and row['result'].get('exact_schema',False) for row in complete),
            'slo_pass':sum(row['result'].get('slo_pass',False) for row in rows),
            'goodput_per_planned_second':sum(row['result'].get('slo_pass',False) for row in rows)/durations,
            'arrival_ttft_median_s':quantile(ttf,.5),'arrival_ttft_p95_s':quantile(ttf,.95),'ttft_observed':len(ttf),
            'tpot_median_s':quantile(tp,.5),'tpot_p95_s':quantile(tp,.95),'tpot_observed':len(tp),
            'cache_usage_observed':len(cache_observed),'cache_usage_unknown':len(complete)-len(cache_observed),
            'positive_cached_token_requests':sum(row['result']['cached_tokens']>0 for row in cache_observed),
            'slices':slices,'scope':'All offered demand retained. TPOT is server-token-normalized visible-output approximation; missing values remain missing.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
    blocks=[analyze(path) for path in sorted(Path(a.input).glob('block-*.jsonl'))]
    per_policy={}
    for policy in sorted({b['policy'] for b in blocks}):
        selected=[b for b in blocks if b['policy']==policy]
        seconds=sum(b['planned_seconds'] for b in selected)
        per_policy[policy]={'blocks':len(selected),'planned_seconds':seconds,
                            'full_planned_window_blocks':sum(b['full_planned_window_qualified'] for b in selected),
                            'actual_exposure_seconds':sum(b['actual_exposure_seconds'] or 0 for b in selected),
                            'descriptive_scope':'Includes all offered demand and truncated blocks; not the causal inference population',
                            'offered':sum(b['offered'] for b in selected),
                            'complete':sum(b['complete'] for b in selected),
                            'slo_pass':sum(b['slo_pass'] for b in selected),
                            'goodput_per_planned_second':sum(b['slo_pass'] for b in selected)/seconds if seconds else None,
                            'outcomes':dict(sum((Counter(b['outcomes']) for b in selected),Counter()))}
    comparisons=[];baseline='round_robin';rng=random.Random(20261011)
    by_trace=defaultdict(dict)
    for b in blocks:
        if not b['full_planned_window_qualified']:continue
        key=(b['seed'],b['rate'],b['planned_seconds'])
        if b['policy'] in by_trace[key]:raise ValueError('Duplicate policy trace binding')
        by_trace[key][b['policy']]=b
    for policy in sorted(set(per_policy)-{baseline}):
        differences=[]
        for bundle in by_trace.values():
            if baseline in bundle and policy in bundle:
                differences.append(bundle[policy]['goodput_per_planned_second']-bundle[baseline]['goodput_per_planned_second'])
        ci=None
        if len(differences)>=3:
            draws=[statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(a.bootstrap)]
            ci=[quantile(draws,.025),quantile(draws,.975)]
        comparisons.append({'policy':policy,'baseline':baseline,'matched_complete_cycles':len(differences),
                            'mean_goodput_difference':statistics.mean(differences) if differences else None,
                            'paired_cycle_bootstrap95':ci,
                            'scope':'Matched cycle-level offered-load comparisons. At least3cycles for interval. Hot prefixes are matched; intended-cold namespaces differ explicitly. Sequential order/warm cache remain potential carryover.'})
    report={'status':'analyzed_retained_journals','blocks':blocks,'policies':per_policy,'comparisons':comparisons,
             'pending_journal_blocks':sum(not b['summary_present'] or b['unresolved_arrivals']>0 for b in blocks),
             'truncated_or_unqualified_blocks':sum(not b['full_planned_window_qualified'] for b in blocks),
             'inference_scope':'Paired CIs consume only full planned windows with summary, no lease stop, and all arrivals terminal. Truncated windows remain in descriptive tables.',
             'unknown_values_policy':'No imputation of token counts, cache hits, or latency.'}
    with Path(a.output).open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'policies':per_policy,'comparisons':comparisons,'blocks':len(blocks)}))


if __name__=='__main__':main()
