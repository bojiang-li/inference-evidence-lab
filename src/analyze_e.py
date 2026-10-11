#!/usr/bin/env python3
"""Finite read-only E evidence analysis. Does not call inference endpoints or start services."""
import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_task(row):
    value = row.get('result', {}).get('text', '').strip()
    expected = row.get('expected')
    if row.get('task') == 'json':
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return False
        return (isinstance(expected, dict) and isinstance(parsed, dict) and set(parsed) == set(expected)
                and all(type(parsed[k]) is type(expected[k]) and parsed[k] == expected[k] for k in expected))
    return isinstance(expected, str) and value == expected


def inspect_row(row):
    result = row.get('result', {})
    usage = result.get('usage') or {}
    terminal = result.get('status') == 'complete' and result.get('done') is True and result.get('finish_reason') == 'stop'
    count = row.get('prompt_count_evidence', {}).get('input_tokens')
    exact_count = type(count) is int and count == usage.get('prompt_tokens')
    content = row.get('actual_user_content')
    reported_hash = row.get('prompt_count_evidence', {}).get('fitted_prompt_sha256')
    actual_hash = hashlib.sha256(content.encode()).hexdigest() if isinstance(content, str) else None
    hash_valid = actual_hash is not None and actual_hash == reported_hash
    correct = check_task(row)
    return {'terminal': terminal, 'correct': correct, 'success': terminal and correct,
            'prompt_count': count, 'count_match': exact_count, 'hash_valid': hash_valid,
            'actual_hash': actual_hash,
            'stored_check_match': row.get('task_success') == (terminal and correct),
            'ttft_s': row.get('metrics', {}).get('ttft_content_s'),
            'tpot_s': row.get('metrics', {}).get('tpot_visible_s'),
            'duration_s': result.get('duration_s'), 'completion_tokens': usage.get('completion_tokens')}


def numbers(rows, key):
    values = [r['_check'][key] for r in rows if finite(r['_check'].get(key))]
    return {'n': len(values), 'min': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'max': max(values) if values else None}


def bootstrap_clusters(pairs, metric, draws=2000):
    clusters = defaultdict(list)
    for pair in pairs:
        left, right = pair['baseline']['_check'].get(metric), pair['candidate']['_check'].get(metric)
        if finite(left) and finite(right):
            clusters[pair['baseline']['case_id']].append(right - left)
    means = [statistics.mean(x) for _key, x in sorted(clusters.items())]
    estimate = statistics.mean(means) if means else None
    result = {'case_clusters': len(means), 'paired_rows': sum(map(len, clusters.values())),
              'candidate_minus_baseline_mean_seconds': estimate, 'ci95_seconds': None,
              'method': 'paired_case_cluster_percentile_bootstrap_equal_case_weight',
              'scope': 'conditional_on_observed_node_config_runs_not_independent_redeployment_ci'}
    if len(means) >= 3:
        rng = random.Random(611202)
        samples = sorted(statistics.mean(rng.choice(means) for _ in means) for _ in range(draws))
        result['ci95_seconds'] = [samples[int(.025 * draws)], samples[min(draws-1, int(.975 * draws))]]
    return result


def compare_pairs(rows, baseline, candidate):
    by_key = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row.get('block'), row.get('case_id'), row.get('node'), row.get('cache_phase'))
        by_key[key][row.get('precision')].append(row)
    exclusions = Counter()
    excluded_rows = []
    pairs = []
    for key, values in sorted(by_key.items(), key=lambda x: str(x[0])):
        left, right = values.get(baseline, []), values.get(candidate, [])
        if not left and not right:
            continue
        reason = None
        if not left or not right:
            reason = 'missing_baseline' if not left else 'missing_candidate'
        elif len(left) != 1 or len(right) != 1:
            reason = 'duplicate_case_node_phase_config'
        else:
            a, b = left[0], right[0]
            if not a['_check']['hash_valid'] or not b['_check']['hash_valid']:
                reason = 'missing_or_invalid_actual_fitted_prompt_hash'
            elif a['_check']['actual_hash'] != b['_check']['actual_hash']:
                reason = 'fitted_prompt_mismatch'
            elif a.get('expected') != b.get('expected') or a.get('task') != b.get('task'):
                reason = 'task_contract_mismatch'
            elif a.get('model') != b.get('model') or a.get('engine_commit') != b.get('engine_commit'):
                reason = 'model_or_engine_mismatch'
            elif a.get('decode_settings') != b.get('decode_settings'):
                reason = 'decode_settings_mismatch'
            elif not a['_check']['count_match'] or not b['_check']['count_match']:
                reason = 'count_usage_mismatch_or_missing'
            elif a['_check']['prompt_count'] != b['_check']['prompt_count']:
                reason = 'prompt_count_mismatch'
            else:
                pairs.append({'baseline': a, 'candidate': b})
        if reason:
            exclusions[reason] += 1
            excluded_rows.append({'key': list(key), 'reason': reason,
                                  'baseline_rows': len(left), 'candidate_rows': len(right)})
    discordant = Counter()
    correctness_table = Counter()
    discordant_cases = []
    for pair in pairs:
        a, b = pair['baseline'], pair['candidate']
        status = str(int(a['_check']['success'])) + str(int(b['_check']['success']))
        correctness_status = str(int(a['_check']['correct'])) + str(int(b['_check']['correct']))
        discordant[status] += 1
        correctness_table[correctness_status] += 1
        if status in ('01', '10') or correctness_status in ('01', '10'):
            discordant_cases.append({'case_id': a['case_id'], 'node': a['node'], 'cache_phase': a['cache_phase'],
                                     'baseline_success': a['_check']['success'], 'candidate_success': b['_check']['success'],
                                     'baseline_task_correct': a['_check']['correct'], 'candidate_task_correct': b['_check']['correct'],
                                     'baseline_status': a['result'].get('status'), 'candidate_status': b['result'].get('status')})
    slices = defaultdict(list)
    latency_exclusions = Counter()
    for pair in pairs:
        a, b = pair['baseline'], pair['candidate']
        if not a['_check']['success'] or not b['_check']['success']:
            latency_exclusions['not_both_task_success'] += 1
            continue
        key = (a.get('length_label'), a.get('task'), a.get('cache_phase'))
        slices[key].append(pair)
    summaries = []
    for key, subset in sorted(slices.items()):
        metrics = {metric: bootstrap_clusters(subset, metric) for metric in ['ttft_s', 'tpot_s', 'duration_s']}
        summaries.append({'length': key[0], 'task': key[1], 'cache_phase': key[2], 'metrics': metrics,
                          'completion_count_different_pairs': sum(x['baseline']['_check']['completion_tokens'] != x['candidate']['_check']['completion_tokens'] for x in subset)})
    return {'baseline': baseline, 'candidate': candidate, 'input_matched_raw_pairs': len(pairs),
            'case_clusters': len({x['baseline']['case_id'] for x in pairs}),
            'paired_success_table': {'both_fail': discordant['00'], 'baseline_fail_candidate_pass': discordant['01'],
                                     'baseline_pass_candidate_fail': discordant['10'], 'both_pass': discordant['11']},
            'paired_task_correctness_table': {'both_wrong': correctness_table['00'], 'baseline_wrong_candidate_right': correctness_table['01'],
                                             'baseline_right_candidate_wrong': correctness_table['10'], 'both_right': correctness_table['11']},
            'discordant_cases': discordant_cases, 'input_pair_exclusions': dict(exclusions),
            'excluded_keys': excluded_rows, 'latency_subset_exclusions': dict(latency_exclusions),
            'latency_estimates_condition_on_both_task_success': True, 'latency_slices': summaries}


def memory_summary(rows, telemetry):
    windows = defaultdict(list)
    for row in rows:
        begin, duration = row.get('started_unix_s'), row.get('result', {}).get('duration_s')
        if finite(begin) and finite(duration):
            windows[(row['_source'], row.get('node'), row.get('precision'))].append((begin, begin+duration))
    associations = defaultdict(dict)
    for (_source, node, precision), spans in windows.items():
        lo, hi = min(x[0] for x in spans), max(x[1] for x in spans)
        for index, sample in enumerate(telemetry):
            if sample.get('node') == node and finite(sample.get('unix')) and lo <= sample['unix'] <= hi:
                associations[(node, precision)][index] = sample
    answer = []
    for (node, precision), indexed in sorted(associations.items()):
        current, available = [], []
        for sample in indexed.values():
            if type(sample.get('mem_available_bytes')) is int:
                available.append(sample['mem_available_bytes'])
            service = dict(line.split('=', 1) for line in sample.get('service', '').splitlines() if '=' in line)
            value = service.get('MemoryCurrent', '')
            if service.get('ActiveState') == 'active' and value.isdigit():
                current.append(int(value))
        answer.append({'node': node, 'precision': precision, 'snapshots_in_file_run_windows': len(indexed),
                       'active_service_memory_samples': len(current),
                       'service_memory_current_min_bytes': min(current) if current else None,
                       'service_memory_current_median_bytes': statistics.median(current) if current else None,
                       'service_memory_current_max_bytes': max(current) if current else None,
                       'mem_available_min_bytes': min(available) if available else None})
    return {'measurement': 'whole_service_cgroup_memory_current_and_whole_node_mem_available',
            'association': 'telemetry_timestamps_inside_each_raw_file_request_run_window',
            'not_direct_kv_allocation_or_gpu_payload': True,
            'limitations': ['30-second snapshots can miss peaks', 'whole model/runtime/allocator/process memory included',
                            'format differences are observed whole-service footprint differences, not theoretical KV compression ratios'],
            'groups': answer}


def analyze(directory, split='holdout', telemetry_path=None, profile_path=None):
    rows, errors, file_summaries = [], [], []
    for path in sorted(Path(directory).rglob('*.jsonl')):
        loaded = []
        with path.open() as fh:
            for number, line in enumerate(fh, 1):
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    errors.append({'file': str(path), 'line': number, 'reason': 'invalid_or_incomplete_json_line', 'error': str(exc)})
                    continue
                if row.get('schema_version') != 1 or 'case_id' not in row:
                    errors.append({'file': str(path), 'line': number, 'reason': 'not_e_schema'})
                    continue
                if row.get('split') != split:
                    continue
                row['_source'] = str(path)
                row['_check'] = inspect_row(row)
                loaded.append(row)
        rows.extend(loaded)
        marker = path.with_suffix(path.suffix + '.finish.json')
        if loaded:
            file_summaries.append({'file': str(path), 'rows_selected': len(loaded), 'finish_marker_present': marker.exists()})
    groups = defaultdict(list)
    prompt_hashes = defaultdict(set)
    for row in rows:
        groups[(row.get('node'), row.get('precision'), row.get('length_label'), row.get('task'), row.get('cache_phase'))].append(row)
        if row['_check']['hash_valid']:
            prompt_hashes[row['case_id']].add(row['_check']['actual_hash'])
    description = []
    for key, values in sorted(groups.items(), key=lambda x: str(x[0])):
        description.append({'node': key[0], 'precision': key[1], 'length': key[2], 'task': key[3], 'cache_phase': key[4],
                            'raw_rows': len(values), 'distinct_cases': len({x['case_id'] for x in values}),
                            'task_correct_rows': sum(x['_check']['correct'] for x in values),
                            'successful_terminal_rows': sum(x['_check']['terminal'] for x in values),
                            'task_success_rows': sum(x['_check']['success'] for x in values),
                            'count_usage_match_rows': sum(x['_check']['count_match'] for x in values),
                            'actual_prompt_hash_valid_rows': sum(x['_check']['hash_valid'] for x in values),
                            'stored_task_check_agrees_rows': sum(x['_check']['stored_check_match'] for x in values),
                            'statuses': dict(Counter(x['result'].get('status') for x in values)),
                            'prompt_counts': sorted({x['_check']['prompt_count'] for x in values if type(x['_check']['prompt_count']) is int}),
                            'cached_tokens': sorted({x['result'].get('cached_tokens') for x in values if type(x['result'].get('cached_tokens')) is int}),
                            'ttft_s': numbers(values, 'ttft_s'), 'tpot_s': numbers(values, 'tpot_s'),
                            'duration_s': numbers(values, 'duration_s')})
    telemetry = []
    if telemetry_path and Path(telemetry_path).exists():
        for line in Path(telemetry_path).read_text().splitlines():
            try:
                telemetry.append(json.loads(line))
            except ValueError:
                pass
    return {'schema_version': 1, 'analyzed_unix_s': time.time(), 'split': split, 'raw_rows': len(rows),
            'manifest_observations': {'models': sorted({x.get('model', '') for x in rows}),
                                      'engine_commits': sorted({x.get('engine_commit', '') for x in rows})},
            'profile_control': json.loads(Path(profile_path).read_text()) if profile_path else None,
            'distinct_cases': len({x['case_id'] for x in rows}), 'files': file_summaries,
            'parse_exclusions': errors, 'global_prompt_hash_mismatched_cases': {key: sorted(value) for key, value in prompt_hashes.items() if len(value)>1},
            'descriptive_slices': description,
            'paired': [compare_pairs(rows, 'fp16', candidate) for candidate in ['int8', 'q4_0']],
            'memory': memory_summary(rows, telemetry),
            'limits': ['Sequential task-quality/latency pilot, not concurrency goodput',
                       'First-pass cache state unverified; repeated pass only empirically warm if cached_tokens supports it',
                       'Bootstrap clusters by case, preserving node pairing; never pool cache phases for inferential sample count',
                       'CI needs at least3distinct cases and is conditional on observed configuration runs',
                       'Different completion token counts can confound completion duration; inspect each slice',
                       'Exact retrieval/copy/JSON success is not general-model accuracy', 'No p99 guarantee from this small case set']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--split', choices=['calibration', 'holdout'], default='holdout')
    p.add_argument('--telemetry')
    p.add_argument('--profile-evidence')
    p.add_argument('--output')
    args = p.parse_args()
    answer = analyze(args.input, args.split, args.telemetry, args.profile_evidence)
    content = json.dumps(answer, indent=2) + '\n'
    if args.output:
        with Path(args.output).open('x') as fh:
            fh.write(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
