#!/usr/bin/env python3
"""Lease-controlled KV precision task evaluation. Standard library; never starts servers."""
import argparse
import hashlib
import importlib.util
import json
import random
import time
from collections import Counter
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_cases(seed=611001, long_target=12000):
    cases = []
    for split, repetitions in [('calibration', 1), ('holdout', 3)]:
        for length, target in [('short', 1024), ('medium', 4096), ('long', long_target)]:
            for task in ['needle', 'copy', 'json']:
                for replicate in range(repetitions):
                    key = f'{split}-{length}-{task}-{replicate}'
                    rng = random.Random(f'{seed}:{key}')
                    code = 'N' + ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(18))
                    copied = ''.join(rng.choice('abcdefghjkmnpqrstuvwxyz23456789') for _ in range(72))
                    record = {'code': code, 'count': rng.randrange(10, 90), 'city': rng.choice(['Oakland', 'Berkeley', 'Fremont'])}
                    # Unique first line prevents unintended shared filler-prefix reuse across cases.
                    lines = [f'Document identity: {sha(key + str(seed))}.']
                    while sum(len(x) + 1 for x in lines) < target * 4 - 800:
                        tag = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(20))
                        lines.append(f'Archive {len(lines):05d} {tag}: inactive reference, no authoritative answer here.')
                    if task == 'needle':
                        evidence = f'AUTHORITATIVE_CODE: {code}'
                        instruction = 'Return only the AUTHORITATIVE_CODE value. Do not explain or add punctuation.'
                        expected = code
                    elif task == 'copy':
                        evidence = f'COPY_EXACTLY: {copied}'
                        instruction = 'Return only the COPY_EXACTLY string, preserving every character.'
                        expected = copied
                    else:
                        evidence = f'AUTHORITATIVE_RECORD: {json.dumps(record, sort_keys=True)}'
                        instruction = 'Return only a JSON object with exactly the keys code, count, city, and their values from AUTHORITATIVE_RECORD. No markdown.'
                        expected = record
                    position = rng.choice([len(lines)//4, len(lines)//2, 3*len(lines)//4])
                    lines.insert(position, evidence)
                    content = '\n'.join(lines) + '\n\nTASK: ' + instruction
                    cases.append({'case_id': key, 'split': split, 'task': task, 'length_label': length,
                                  'target_tokens_estimate': target, 'estimate_method': 'approximately_4_chars_per_token_not_measured',
                                  'needle_position_fraction': position / len(lines), 'expected': expected,
                                  'user_content': content, 'prompt_sha256': sha(content)})
    return cases


def correctness(case, text):
    stripped = text.strip()
    if case['task'] == 'json':
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            return {'correct': False, 'criterion': 'strict_json_exact_fields_and_types', 'error': 'invalid_json'}
        expected = case['expected']
        passed = (isinstance(parsed, dict) and set(parsed) == set(expected)
                  and all(type(parsed[k]) is type(expected[k]) and parsed[k] == expected[k] for k in expected))
        return {'correct': passed, 'criterion': 'strict_json_exact_fields_and_types'}
    return {'correct': stripped == case['expected'], 'criterion': 'exact_value_after_outer_whitespace_strip'}


def make_plan(seed=611001, blocks=1):
    rng = random.Random(seed)
    plan = []
    for block in range(blocks):
        rounds = [('fp16', 'q4_0'), ('q4_0', 'int8'), ('int8', 'fp16')]
        rng.shuffle(rounds)
        for ordinal, pair in enumerate(rounds):
            plan.append({'block': block, 'round': ordinal,
                         'assignments': [{'node': node, 'precision': precision} for node, precision in zip(['spark1', 'spark2'], pair)],
                         'cache_phases': ['first_pass', 'repeated_pass'],
                         'requires_root_sequential_config_restart': True,
                         'requests_per_assignment': 54, 'task_replicates_per_length': 3})
    return plan


def completion_metrics(result, usage_visible_only=False):
    usage = result.get('usage') if isinstance(result.get('usage'), dict) else {}
    details = usage.get('completion_tokens_details') or {}
    count = usage.get('completion_tokens')
    prompt_count = usage.get('prompt_tokens')
    valid_count = type(count) is int and count >= 0
    reasoning_tokens = details.get('reasoning_tokens') if isinstance(details, dict) else None
    first, last = result.get('first_content_s'), result.get('last_content_s')
    visible_confirmed = usage_visible_only and reasoning_tokens in (None, 0) and not result.get('reasoning_text')
    tpot = ((last - first) / (count - 1) if visible_confirmed and valid_count and count > 1
            and first is not None and last is not None else None)
    return {'server_prompt_tokens': prompt_count, 'server_completion_tokens': count,
            'server_reasoning_tokens': reasoning_tokens, 'completion_count_valid': valid_count,
            'visible_only_usage_attested': visible_confirmed, 'ttft_content_s': first,
            'tpot_visible_s': tpot, 'tpot_source': 'usage_completion_count_and_content_span' if tpot is not None else 'unavailable',
            'timing_clock': 'time.monotonic', 'sse_events_are_tokens': False}


def load_client(path):
    spec = importlib.util.spec_from_file_location('a_client', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def count_prompt(client, endpoint, payload):
    """Strata read-only prompt encoder; exactness across API templates remains checked by usage."""
    conn, base = client.connection(endpoint, 15)
    request = {'model': payload['model'], 'messages': payload['messages'], 'thinking': {'type': 'disabled'}}
    try:
        conn.request('POST', base + '/messages/count_tokens', json.dumps(request).encode(),
                     {'Content-Type': 'application/json'})
        response = conn.getresponse()
        body = response.read(65537)
        if response.status != 200 or len(body) > 65536:
            return {'status': 'failed', 'http_status': response.status, 'error_body': body[:2048].decode(errors='replace')}
        obj = json.loads(body)
        count = obj.get('input_tokens')
        if type(count) is not int or count <= 0:
            return {'status': 'failed', 'error': 'invalid input_tokens', 'response': obj}
        return {'status': 'counted', 'input_tokens': count, 'method': 'strata_messages_count_tokens',
                'thinking': 'disabled', 'template_equivalence': 'compare_actual_completion_usage'}
    except (OSError, ValueError) as exc:
        return {'status': 'failed', 'error': repr(exc)}
    finally:
        conn.close()


def fit_prompt(client, endpoint, payload, case, target, maximum):
    """Count and trim only distractor filler. Retain answer and proportional placement."""
    lines = case['user_content'].splitlines()
    identity = lines[0]
    evidence = next(x for x in lines if x.startswith(('AUTHORITATIVE_CODE:', 'COPY_EXACTLY:', 'AUTHORITATIVE_RECORD:')))
    task_line = next(x for x in lines if x.startswith('TASK: '))
    filler = '\n'.join(x for x in lines if x.startswith('Archive '))
    # Supply enough distractors for the desired count even if the estimate was short.
    filler += ('\nArchive padding: inactive reference contains no valid answer.' * target)
    probes = 0

    def candidate(chars, padding=0):
        chunk = filler[:chars].splitlines()
        position = round(case['needle_position_fraction'] * len(chunk))
        chunk.insert(position, evidence)
        content = identity + '\n' + '\n'.join(chunk) + ('\n' + ' x' * padding if padding else '') + '\n\n' + task_line
        trial = dict(payload, messages=[{'role': 'user', 'content': content}])
        return trial, position / max(1, len(chunk))

    def measured(chars, padding=0):
        nonlocal probes
        trial, position = candidate(chars, padding)
        ev = count_prompt(client, endpoint, trial)
        probes += 1
        if ev['status'] != 'counted':
            raise ValueError('Cannot fit context without reliable count_tokens response: ' + str(ev))
        return trial, ev, position

    lo, hi = 0, min(len(filler), max(target * 12, 4096))
    best = measured(0)
    if best[1]['input_tokens'] > target:
        raise ValueError('Protected answer/instruction exceeds context target')
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = measured(mid)
        count = trial[1]['input_tokens']
        if count <= target:
            best = trial
            lo = mid + 1
        else:
            hi = mid - 1
        if count == target:
            best = trial
            break
    delta = target - best[1]['input_tokens']
    if delta:
        # ' x' usually gives one extra token; check instead of assuming.
        for padding in range(max(0, delta - 4), delta + 5):
            trial = measured(max(0, hi), padding)
            count = trial[1]['input_tokens']
            if count <= target and abs(target-count) < abs(target-best[1]['input_tokens']):
                best = trial
            if count == target:
                best = trial
                break
    if best[1]['input_tokens'] > maximum or abs(best[1]['input_tokens'] - target) > 8:
        raise ValueError('Context fit outside target tolerance or engine budget')
    best[1].update(target_tokens=target, target_delta=best[1]['input_tokens']-target,
                   fitted_prompt_sha256=sha(best[0]['messages'][0]['content']),
                   original_prompt_sha256=case['prompt_sha256'], count_probes=probes,
                   fitted_needle_position_fraction=best[2], fit_method='count_trim_distractor_only')
    return best[0], best[1]


def lease_check(path, node, precision, endpoint, expected_id=None, now=None):
    try:
        lease = json.loads(Path(path).read_text())
        entry = lease.get('nodes', {}).get(node, {})
        valid = (lease.get('task') == 'E' and entry.get('allow') is True
                 and entry.get('precision') == precision and entry.get('endpoint') == endpoint
                 and type(lease.get('expires_at')) in (int, float)
                 and lease['expires_at'] > (time.time() if now is None else now)
                 and isinstance(lease.get('run_id'), str)
                 and (expected_id is None or expected_id == lease['run_id']))
        return {'valid': valid, 'run_id': lease.get('run_id'), 'expires_at': lease.get('expires_at'),
                'error': None if valid else 'lease_task_node_precision_endpoint_expiry_or_id_mismatch'}
    except (OSError, ValueError, TypeError) as exc:
        return {'valid': False, 'error': repr(exc)}


def run(args):
    if not args.leasefile:
        raise ValueError('A root-issued node lease file is required; this script does not acquire leases.')
    cases = [json.loads(x) for x in Path(args.cases).read_text().splitlines() if x]
    cases = [x for x in cases if x['split'] == args.split]
    if args.length:
        cases = [x for x in cases if x['length_label'] == args.length]
    random.Random(f'{args.seed}:{args.block}').shuffle(cases)
    client = load_client(args.client)
    extras = json.loads(Path(args.payload_extra).read_text()) if args.payload_extra else {}
    protected = {'model', 'messages', 'stream', 'stream_options', 'temperature', 'top_p', 'max_tokens', 'chat_template_kwargs'}
    if not isinstance(extras, dict) or protected.intersection(extras):
        raise ValueError('Payload extras must not override fixed protocol fields.')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('Use a fresh evidence filename; existing output will not be overwritten.')
    counts = Counter()
    started = time.time()
    with out.open('x') as fh:
        for case in cases:
            # Consecutive repetition is a deliberately separate warm-cache microprobe.
            # It is not a realistic traffic replay or a proof of actual cache hit.
            payload = {'model': args.model, 'messages': [{'role': 'user', 'content': case['user_content']}],
                           'stream': True, 'stream_options': {'include_usage': True},
                           'temperature': 0, 'top_p': 1, 'max_tokens': 128,
                           'chat_template_kwargs': {'enable_thinking': False}}
            payload.update(extras)
            lease_evidence = lease_check(args.leasefile, args.node, args.precision, args.endpoint, args.lease_id)
            if not lease_evidence['valid']:
                token_evidence = {'status': 'failed', 'error': 'lease_not_valid_before_case'}
            elif args.fit_context:
                try:
                    payload, token_evidence = fit_prompt(client, args.endpoint, payload, case,
                                                        (args.long_target if case['length_label'] == 'long' and args.long_target
                                                         else case['target_tokens_estimate']), args.max_prompt_tokens)
                except ValueError as exc:
                    token_evidence = {'status': 'failed', 'error': str(exc)}
            else:
                token_evidence = count_prompt(client, args.endpoint, payload)
            for phase in ['first_pass', 'repeated_pass']:
                start_wall = time.time()
                lease_evidence = lease_check(args.leasefile, args.node, args.precision, args.endpoint, args.lease_id)
                budget_ok = token_evidence.get('status') == 'counted' and token_evidence['input_tokens'] <= args.max_prompt_tokens
                if not lease_evidence['valid']:
                    result = {'status': 'lease_preflight_failed', 'done': False, 'finish_reason': None,
                              'usage': None, 'text': '', 'duration_s': 0, 'error': lease_evidence['error']}
                elif budget_ok:
                    timeout = min(args.timeout, max(.001, lease_evidence['expires_at'] - time.time()))
                    result = client.stream_request(args.endpoint, payload, timeout=timeout)
                else:
                    result = {'status': 'prompt_preflight_failed', 'done': False, 'finish_reason': None,
                              'usage': None, 'text': '', 'duration_s': 0, 'error': 'count_unavailable_or_prompt_over_budget'}
                valid_terminal = result.get('status') == 'complete' and result.get('done') is True and result.get('finish_reason') == 'stop'
                checked = correctness(case, result.get('text', ''))
                row = {'schema_version': 1, 'case_id': case['case_id'], 'split': case['split'], 'task': case['task'],
                       'length_label': case['length_label'], 'target_tokens_estimate': case['target_tokens_estimate'],
                       'prompt_sha256': case['prompt_sha256'], 'expected': case['expected'],
                       'node': args.node, 'precision': args.precision, 'block': args.block,
                       'lease_id': lease_evidence.get('run_id'), 'lease_evidence': lease_evidence,
                       'engine_commit': args.engine_commit, 'cache_phase': phase,
                       'cache_state': 'root_attested_cold' if phase == 'first_pass' and args.cold_attested else 'unverified',
                       'model': args.model, 'endpoint': args.endpoint, 'started_unix_s': start_wall,
                       'actual_user_content': payload['messages'][0]['content'],
                       'prompt_count_evidence': token_evidence, 'max_prompt_tokens': args.max_prompt_tokens,
                       'count_usage_match': (token_evidence.get('input_tokens') == result['usage'].get('prompt_tokens')
                                             if isinstance(result.get('usage'), dict) else None),
                       'decode_settings': {k: v for k, v in payload.items() if k != 'messages'},
                       'valid_terminal': valid_terminal, 'task_check': checked,
                       'task_success': valid_terminal and checked['correct'], 'result': result,
                       'metrics': completion_metrics(result, args.usage_visible_only)}
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                fh.flush()
                counts[result['status']] += 1
                counts['task_success'] += int(row['task_success'])
                counts['rows'] += 1
    with out.with_suffix(out.suffix + '.finish.json').open('x') as fh:
        json.dump({'finished': True, 'started_unix_s': started, 'ended_unix_s': time.time(),
                   'node': args.node, 'precision': args.precision, 'planned_cases': len(cases),
                   'planned_rows': 2*len(cases), 'counts': dict(counts),
                   'all_records_terminal': counts['rows'] == 2*len(cases),
                   'quality_success_not_assumed': True}, fh, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate')
    gen.add_argument('--directory', required=True)
    gen.add_argument('--seed', type=int, default=611001)
    gen.add_argument('--blocks', type=int, default=1)
    gen.add_argument('--long-target', type=int, default=12000)
    runp = sub.add_parser('run')
    runp.add_argument('--cases', required=True)
    runp.add_argument('--client', required=True)
    runp.add_argument('--endpoint', required=True)
    runp.add_argument('--model', default='qwen3.8-flash-next-unsloth-ud-iq4_xs')
    runp.add_argument('--node', choices=['spark1', 'spark2'], required=True)
    runp.add_argument('--precision', choices=['fp16', 'int8', 'q4_0'], required=True)
    runp.add_argument('--block', type=int, required=True)
    runp.add_argument('--lease-id')
    runp.add_argument('--leasefile', required=True)
    runp.add_argument('--engine-commit', required=True)
    runp.add_argument('--output', required=True)
    runp.add_argument('--split', choices=['calibration', 'holdout'], default='holdout')
    runp.add_argument('--length', choices=['short', 'medium', 'long'])
    runp.add_argument('--seed', type=int, default=611001)
    runp.add_argument('--timeout', type=float, default=180)
    runp.add_argument('--max-prompt-tokens', type=int, default=16256)
    runp.add_argument('--fit-context', action='store_true')
    runp.add_argument('--long-target', type=int, help='Override long context target without modifying original case manifest')
    runp.add_argument('--payload-extra')
    runp.add_argument('--usage-visible-only', action='store_true')
    runp.add_argument('--cold-attested', action='store_true')
    args = p.parse_args()
    if args.command == 'generate':
        destination = Path(args.directory)
        destination.mkdir(parents=True, exist_ok=True)
        for name, rows in [('cases.jsonl', make_cases(args.seed, args.long_target)), ('plan.jsonl', make_plan(args.seed, args.blocks))]:
            with (destination / name).open('x') as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    else:
        run(args)


if __name__ == '__main__':
    main()
