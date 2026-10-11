#!/usr/bin/env python3
"""CPU-only final reporting from compact v4 aggregates; never reads request journals."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

SERVING_SHA = '001747849623dcea5a9dbb1245e35141e960c7c898ae1246d202d5b551336e7f'
ANALYZER_SHA = '4cf94a0502b7d95693ce80d65bb11255975dffafabcafb5391002978403368d3'
POLICIES = ['round_robin', 'least_queue', 'cache_aware', 'cache_feedback']
DRAW_COUNT = 10000
BOOTSTRAP_SEED = 2026801103


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, label):
    require(type(value) is int and value >= 0, 'Invalid count: ' + label)
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid number: ' + label)
    return value


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def build_report(data, protocol, manifest, completion=None, analyzer_verified=False):
    require(data.get('status') == 'analyzed_retained_journals', 'Not an analyzer-v4 report')
    args = manifest['args']
    require(manifest.get('source_sha256') == SERVING_SHA, 'Serving source SHA mismatch')
    require(protocol.get('selected_before_holdout') is True, 'Protocol not frozen before holdout')
    require(args['policies'] == POLICIES, 'Unexpected planned policy family')
    require(protocol['holdout_seed'] == args['seed'] == 20268011, 'Holdout seed mismatch')
    for key, arg in [('base_rate', 'rate'), ('ttft_slo_s', 'ttft_slo'),
                     ('tpot_slo_s', 'tpot_slo'), ('hours_requested', 'hours')]:
        require(close(protocol[key], args[arg]), 'Protocol/manifest mismatch: ' + key)
    require(protocol['load_levels'] == [0.7, 1.3] and protocol['burst_multiplier'] == 2,
            'Unexpected offered-load envelope')
    require(close(args['hours'], 8) and close(args['block_seconds'], 600), 'Unexpected runtime/window')
    require(close(args['rate'], 1.68) and close(args['ttft_slo'], 10) and close(args['tpot_slo'], .15),
            'Unexpected frozen rate/SLO')
    blocks = data['blocks']
    require(bool(blocks), 'No retained blocks')
    bundles = {}
    policy_stats = {}
    seen_files = set()
    for b in blocks:
        policy = b['policy']
        require(policy in POLICIES, 'Unknown policy')
        require(b['file'] not in seen_files, 'Duplicate block file')
        seen_files.add(b['file'])
        cycle = b['seed'] - args['seed']
        require(type(cycle) is int and cycle >= 0, 'Invalid held-out cycle seed')
        expected_rate = args['rate'] * (0.7 if cycle % 2 == 0 else 1.3)
        require(close(number(b['rate'], 'rate'), expected_rate), 'Cycle load level mismatch')
        require(close(b['planned_seconds'], args['block_seconds']), 'Unexpected block duration')
        endpoints = args['endpoints']
        shift = cycle % len(endpoints)
        require(b['endpoints'] == endpoints[shift:] + endpoints[:shift], 'Replica crossover mismatch')
        key = (b['seed'], b['rate'], b['planned_seconds'])
        bundle = bundles.setdefault(key, {})
        require(policy not in bundle, 'Duplicate policy/trace binding')
        bundle[policy] = b
        for name in ('offered', 'terminal', 'unresolved_arrivals', 'complete', 'slo_pass',
                     'complete_correct_json', 'complete_with_done_and_finish', 'complete_with_usage'):
            count(b[name], name)
        require(b['offered'] == b['terminal'] + b['unresolved_arrivals'], 'Unbalanced arrival ledger')
        require(sum(count(v, k) for k, v in b['outcomes'].items()) == b['terminal'], 'Outcome total mismatch')
        require(b['outcomes'].get('complete', 0) == b['complete'], 'Complete count mismatch')
        require(b['slo_pass'] <= b['complete_correct_json'] <= b['complete'], 'Quality/SLO contradiction')
        require(b['complete_with_done_and_finish'] == b['complete'], 'Complete without DONE/finish')
        require(b['complete_with_usage'] <= b['complete'], 'Usage count contradiction')
        require(close(b['goodput_per_planned_second'], b['slo_pass']/b['planned_seconds']), 'Goodput denominator mismatch')
        expected_full = bool(b['summary_present'] and not b['stopped_by_lease'] and
                             b['actual_exposure_seconds'] == b['planned_seconds'] and not b['unresolved_arrivals'])
        require(b['full_planned_window_qualified'] == expected_full, 'Full-window flag contradiction')
        if b['actual_exposure_seconds'] is not None:
            require(close(b['actual_exposure_seconds'], b['planned_seconds']), 'Unsupported exact exposure')
        stats = policy_stats.setdefault(policy, {'blocks': 0, 'offered': 0, 'terminal': 0,
            'pending': 0, 'complete': 0, 'correct': 0, 'slo_pass': 0, 'planned_seconds': 0,
            'exact_exposure_seconds': 0, 'unknown_exposure_blocks': 0, 'outcomes': {}})
        stats['blocks'] += 1
        for name in ('offered', 'terminal', 'complete', 'slo_pass', 'planned_seconds'):
            stats[name] += b[name]
        stats['pending'] += b['unresolved_arrivals']
        stats['correct'] += b['complete_correct_json']
        stats['exact_exposure_seconds'] += b['actual_exposure_seconds'] or 0
        stats['unknown_exposure_blocks'] += b['actual_exposure_seconds'] is None
        for name, value in b['outcomes'].items():
            stats['outcomes'][name] = stats['outcomes'].get(name, 0) + value
    require(set(policy_stats) == set(POLICIES), 'Missing planned policy')
    for policy, stats in policy_stats.items():
        stats['complete_wrong_quality'] = stats['complete'] - stats['correct']
        stats['correct_without_recorded_slo_pass'] = stats['correct'] - stats['slo_pass']
        stats['goodput_per_planned_second'] = stats['slo_pass']/stats['planned_seconds']
        original = data['policies'][policy]
        for name in ('blocks', 'offered', 'complete', 'slo_pass', 'planned_seconds', 'outcomes'):
            require(original[name] == stats[name], 'Analyzer policy aggregate mismatch: ' + policy + '/' + name)
    pending = sum(not b['summary_present'] or b['unresolved_arrivals'] > 0 for b in blocks)
    unqualified = sum(not b['full_planned_window_qualified'] for b in blocks)
    require(data['pending_journal_blocks'] == pending and data['truncated_or_unqualified_blocks'] == unqualified,
            'Analyzer qualification totals mismatch')
    # All four policies must be present and full for final planned-family inference.
    matched = [(key, bundle) for key, bundle in sorted(bundles.items())
               if set(bundle) == set(POLICIES) and all(b['full_planned_window_qualified'] for b in bundle.values())]
    cycles = [{'cycle': key[0]-args['seed'], 'seed': key[0], 'rate': key[1],
               'planned_seconds_per_policy': key[2], 'offered_by_policy': {p: bundle[p]['offered'] for p in POLICIES},
               'goodput': {p: bundle[p]['goodput_per_planned_second'] for p in POLICIES}} for key, bundle in matched]
    for cycle in cycles:
        require(len(set(cycle['offered_by_policy'].values())) == 1, 'Matched cycle offered trace counts differ')
    v4 = {c['policy']: c for c in data['comparisons']}
    require(set(v4) == set(POLICIES[1:]), 'Unexpected v4 comparisons')
    comparisons = []
    for index, policy in enumerate(POLICIES[1:]):
        require(v4[policy]['baseline'] == POLICIES[0], 'Unexpected baseline')
        paired_count = sum(POLICIES[0] in bundle and policy in bundle and
            bundle[POLICIES[0]]['full_planned_window_qualified'] and bundle[policy]['full_planned_window_qualified']
            for bundle in bundles.values())
        require(paired_count == v4[policy]['matched_complete_cycles'], 'v4 paired count mismatch')
        original_diffs = [bundle[policy]['goodput_per_planned_second']-bundle[POLICIES[0]]['goodput_per_planned_second']
            for bundle in bundles.values() if POLICIES[0] in bundle and policy in bundle and
            bundle[POLICIES[0]]['full_planned_window_qualified'] and bundle[policy]['full_planned_window_qualified']]
        require((not original_diffs and v4[policy]['mean_goodput_difference'] is None) or
                (original_diffs and close(statistics.mean(original_diffs), v4[policy]['mean_goodput_difference'])),
                'v4 paired mean mismatch')
        diffs = [c['goodput'][policy]-c['goodput'][POLICIES[0]] for c in cycles]
        interval = None
        if len(diffs) >= 3:
            rng = random.Random(BOOTSTRAP_SEED + index)
            draws = [statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(DRAW_COUNT)]
            tail = 0.05/(2*3)
            interval = [quantile(draws, tail), quantile(draws, 1-tail)]
        comparisons.append({'policy': policy, 'baseline': POLICIES[0],
            'v4_unadjusted_preserved': v4[policy], 'all_policy_matched_cycles': len(diffs),
            'matched_cycle_differences_req_per_s': diffs,
            'mean_difference_req_per_s': statistics.mean(diffs) if diffs else None,
            'bonferroni_familywise95_paired_cycle_percentile': interval,
            'draws': DRAW_COUNT, 'seed': BOOTSTRAP_SEED+index, 'planned_family_size': 3,
            'tails': [0.05/(2*3), 1-0.05/(2*3)],
            'by_load_level_descriptive': {str(rate): {'cycles': len(selected),
                'mean_difference_req_per_s': statistics.mean(selected)}
                for rate in sorted({c['rate'] for c in cycles})
                for selected in [[c['goodput'][policy]-c['goodput'][POLICIES[0]] for c in cycles if c['rate'] == rate]]}})
    wall = None
    finished = None
    if completion:
        require(completion['exit_code'] == 0, 'Coordinator completion exit not zero')
        require(completion['analyzer_sha256'] == ANALYZER_SHA, 'Completion analyzer SHA mismatch')
        wall = timestamp(completion['finished_at_utc']) - timestamp(completion['started_at_utc'])
        require(wall >= 0, 'Negative wall duration')
        require(abs(timestamp(completion['started_at_utc'])-manifest['created_at']) <= 30, 'Completion/manifest start mismatch')
        finished = completion['finished_at_utc']
    exposure = sum(b['actual_exposure_seconds'] or 0 for b in blocks)
    planned = sum(b['planned_seconds'] for b in blocks)
    if wall is not None:
        require(wall >= exposure, 'Completion wall shorter than sequential known offered exposure')
    return {'status': 'qualified_completed_8h' if completion and wall >= 8*3600 and not pending and not unqualified and analyzer_verified else 'qualification_incomplete',
        'serving_sha256': SERVING_SHA, 'analyzer_sha256_expected': ANALYZER_SHA,
        'analyzer_source_verified_locally': analyzer_verified, 'protocol': protocol, 'manifest': manifest,
        'completion_receipt': completion, 'finished_at_utc': finished, 'wall_seconds': wall,
        'planned_offered_exposure_seconds_all_blocks': planned,
        'known_exact_offered_exposure_seconds_all_blocks': exposure,
        'unknown_exposure_blocks': sum(b['actual_exposure_seconds'] is None for b in blocks),
        'non_offered_wall_overhead_seconds': wall-exposure if wall is not None and not unqualified else None,
        'drain_seconds': None, 'time_scope': 'Run wall minus sequential full-window exposure includes drain, setup, and interblock work; v4 does not retain block wall clocks, so drain alone is unavailable.',
        'pending_journal_blocks': pending, 'truncated_or_unqualified_blocks': unqualified,
        'all_policy_matched_cycles': cycles, 'unmatched_or_unqualified_cycles': len(bundles)-len(cycles),
        'policies': policy_stats, 'comparisons': comparisons,
        'limits': ['One deployment/run; paired cycles are not independent deployment replications.',
            'Bootstrap conditions on observed matched cycles; temporal dependence and sequential warm-cache carryover can invalidate nominal coverage.',
            'Bonferroni percentile intervals are an approximate 95% familywise construction for exactly three planned comparisons; not a distribution-free guarantee.',
            'Fewer than six complete cycles makes interval interpretation especially fragile; fewer than three yields no interval.',
            'Mixed alternating low/high load levels and 2x bursts are protocol-specific; load slices are descriptive, with no additional significance claims.',
            'Intended-cold namespaces differ by policy block, while shared prefixes and arrival seeds are matched.',
            'Client-cap rejections and intentional cancellations remain in all offered accounting; closed-loop completion-only throughput is not reported.',
            'Correct-without-SLO-pass includes latency and missing/invalid usage eligibility; v4 aggregates cannot isolate TTFT vs TPOT vs usage failure.',
            'TPOT is server-token-normalized visible-output approximation; content events are not model tokens.',
            'No post-hoc universal winner, speedup, equivalence, or production quality claim.']}


def text_report(report):
    lines = ['A held-out two-replica replay — final validation report', 'Status: '+report['status'],
        'Wall seconds: '+str(report['wall_seconds']),
        'Planned offered exposure (sum of sequential blocks): '+str(report['planned_offered_exposure_seconds_all_blocks']),
        'Known exact offered exposure: '+str(report['known_exact_offered_exposure_seconds_all_blocks']),
        'Wall outside offered exposure (drain/setup/interblock combined): '+str(report['non_offered_wall_overhead_seconds']),
        'Drain alone: unavailable in analyzer-v4 schema.',
        'Pending/unqualified blocks: %s/%s'%(report['pending_journal_blocks'], report['truncated_or_unqualified_blocks']), '',
        'All offered demand, including unsuccessful terminal outcomes:']
    for policy, s in report['policies'].items():
        lines.append('%s: offered=%d terminal=%d pending=%d complete=%d correct=%d wrong-quality=%d SLO-pass=%d correct-without-SLO=%d goodput=%.6f req/s outcomes=%s'%(
            policy,s['offered'],s['terminal'],s['pending'],s['complete'],s['correct'],s['complete_wrong_quality'],s['slo_pass'],s['correct_without_recorded_slo_pass'],s['goodput_per_planned_second'],json.dumps(s['outcomes'],sort_keys=True)))
    lines += ['', 'Planned comparisons vs round_robin; no significance or winner claim:']
    for c in report['comparisons']:
        lines.append('%s: n=%d all-policy matched cycles; mean delta=%s req/s; Bonferroni familywise percentile interval=%s; preserved v4 unadjusted interval=%s'%(
            c['policy'],c['all_policy_matched_cycles'],c['mean_difference_req_per_s'],c['bonferroni_familywise95_paired_cycle_percentile'],c['v4_unadjusted_preserved']['paired_cycle_bootstrap95']))
        lines.append('  Descriptive load slices: '+json.dumps(c['by_load_level_descriptive'],sort_keys=True))
    lines += ['', 'Limitations:'] + ['- '+limit for limit in report['limits']]
    return '\n'.join(lines)+'\n'


def plot_report(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    require(matplotlib.__version__ == '3.10.9', 'Use isolated remote matplotlib==3.10.9')
    def save(fig, name):
        fig.savefig(directory/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(directory/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
    cycles = report['all_policy_matched_cycles']
    fig, ax = plt.subplots(figsize=(10,5))
    for policy in POLICIES:
        ax.plot([c['cycle'] for c in cycles], [c['goodput'][policy] for c in cycles], marker='o', label=policy)
    ax.set(xlabel='Matched cycle index (alternating low/high offered load)', ylabel='Correct SLO goodput (requests / planned offered second)', title='Full all-policy matched windows; one deployment/run')
    ax.legend(fontsize=9); ax.grid(alpha=.25)
    fig.text(.02,.01,'No request-level IID interval. Sequential cache carryover remains possible.',fontsize=9)
    save(fig,'matched-cycle-goodput')
    fig, ax = plt.subplots(figsize=(11,5))
    categories = ['SLO pass','Correct, no SLO pass','Complete, wrong quality']
    values = [[report['policies'][p][key] for p in POLICIES] for key in ('slo_pass','correct_without_recorded_slo_pass','complete_wrong_quality')]
    other = sorted({k for s in report['policies'].values() for k in s['outcomes'] if k != 'complete'})
    for outcome in other:
        categories.append(outcome); values.append([report['policies'][p]['outcomes'].get(outcome,0) for p in POLICIES])
    categories.append('Pending'); values.append([report['policies'][p]['pending'] for p in POLICIES])
    bottoms = [0]*len(POLICIES)
    for label, heights in zip(categories, values):
        ax.bar(POLICIES,heights,bottom=bottoms,label=label)
        bottoms=[a+b for a,b in zip(bottoms,heights)]
    ax.set(ylabel='All offered requests (count)',title='All-arrival outcome accounting, including unqualified blocks')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    fig.text(.02,.01,'Correct/no-SLO includes latency and usage eligibility; failure causes cannot be separated from v4 aggregates.',fontsize=8)
    save(fig,'all-offered-outcomes')
    fig, ax = plt.subplots(figsize=(10,5))
    for index,c in enumerate(report['comparisons']):
        v4=c['v4_unadjusted_preserved']; mean=c['mean_difference_req_per_s']; ci=c['bonferroni_familywise95_paired_cycle_percentile']
        if mean is not None: ax.plot(mean,index,'o',color='navy')
        if ci: ax.plot(ci,[index,index],linewidth=3,color='navy')
        if v4['paired_cycle_bootstrap95']:
            ax.plot(v4['paired_cycle_bootstrap95'],[index+.12,index+.12],linewidth=2,color='orange')
    ax.axvline(0,color='black',linestyle='--'); ax.set_yticks(range(3),POLICIES[1:])
    ax.set(xlabel='Paired-cycle mean goodput difference vs round_robin (req/s)',title='Blue: approximate Bonferroni familywise 95%; orange: preserved v4 unadjusted 95%')
    ax.grid(axis='x',alpha=.25)
    fig.text(.02,.01,'Blue uses full four-policy cycles; orange uses original pairwise full cycles. Conditional one-run intervals.',fontsize=8)
    save(fig,'comparisons-vs-round-robin')


def read_json(path):
    path=Path(path)
    require(path.stat().st_size <= 20*1024*1024, 'Compact input exceeds 20 MiB; do not pass request logs')
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True);p.add_argument('--protocol',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--completion');p.add_argument('--analyzer-source');p.add_argument('--output',required=True)
    p.add_argument('--no-plots',action='store_true',help='CPU fixture/report-only use; final remote run emits plots by default')
    a=p.parse_args()
    verified=False
    if a.analyzer_source:
        require(hashlib.sha256(Path(a.analyzer_source).read_bytes()).hexdigest()==ANALYZER_SHA,'Analyzer file SHA mismatch')
        verified=True
    report=build_report(read_json(a.analysis),read_json(a.protocol),read_json(a.manifest),read_json(a.completion) if a.completion else None,verified)
    report['input_sha256']={key:hashlib.sha256(Path(value).read_bytes()).hexdigest() for key,value in vars(a).items() if key in ('analysis','protocol','manifest','completion') and value}
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    with (output/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    (output/'report.txt').write_text(text_report(report))
    if not a.no_plots:plot_report(report,output)
    print(json.dumps({'status':report['status'],'output':str(output),'matched_cycles':len(report['all_policy_matched_cycles'])}))


if __name__=='__main__':main()
