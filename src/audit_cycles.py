"""CPU-only retained-journal audit. No request, service, lease or completion actions."""
import argparse
import collections
import hashlib
import json
import pathlib
import random

SERVING_SHA = '001747849623dcea5a9dbb1245e35141e960c7c898ae1246d202d5b551336e7f'
POLICIES = {'round_robin', 'least_queue', 'cache_aware', 'cache_feedback'}
META = ['arrival_offset_s', 'family', 'target_prompt_tokens_estimate', 'prefix_hash',
        'prefix_characters', 'total_characters', 'prompt_sha256', 'cold',
        'intentional_cancel', 'expected_answer']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def answer_flags(text, expected):
    """Recompute the frozen harness's documented code-fence-normalized JSON checks."""
    require(isinstance(text, str), 'Complete response text is missing/non-string')
    value = text.strip()
    if value.startswith('```'):
        lines = value.splitlines()
        if lines[-1].startswith('```'):
            value = '\n'.join(lines[1:-1]).strip()
    try:
        obj = json.loads(value)
    except ValueError:
        return False, False, False
    return True, isinstance(obj, dict) and obj.get('answer') == expected, isinstance(obj, dict) and set(obj) == {'answer'}


def load_block(path):
    header = summary = None
    arrivals, terminals = {}, {}
    raw = path.read_bytes()
    require(raw.endswith(b'\n'), 'Partial journal line: ' + path.name)
    for number, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        kind = row['type']
        if kind == 'block':
            require(header is None, 'Duplicate header')
            header = row
        elif kind == 'summary':
            require(summary is None, 'Duplicate summary')
            summary = row
        elif kind in ('arrival', 'terminal'):
            dest = arrivals if kind == 'arrival' else terminals
            key = row['arrival_id']
            require(key not in dest, 'Duplicate ' + kind + ' at line ' + str(number))
            dest[key] = row
        else:
            raise ValueError('Unknown journal row type')
    require(header is not None and summary is not None, 'Unclosed block: ' + path.name)
    require(set(arrivals) == set(terminals), 'Arrival/terminal identity mismatch')
    require(len(arrivals) == summary['offered'] == summary['terminal'], 'Summary count mismatch')
    require(summary['router_pending_final'] == [0, 0], 'Router pending is not zero')
    require(summary['stopped_by_lease'] is False, 'Lease-stopped block')
    require(summary['planned_seconds'] == header['duration_s'] and summary['wall_seconds'] >= header['duration_s'], 'Truncated exposure')
    require(summary['policy'] == header['policy'], 'Summary policy mismatch')
    outcomes = collections.Counter(row['result']['status'] for row in terminals.values())
    require(dict(outcomes) == summary['outcomes'], 'Outcome count mismatch')
    complete = wrong = 0
    for key, terminal in terminals.items():
        arrival = arrivals[key]
        require(all(terminal[k] == arrival[k] for k in META), 'Terminal request metadata changed')
        result = terminal['result']
        if result['status'] != 'complete':
            continue
        complete += 1
        require(result.get('done') is True and bool(result.get('finish_reason')), 'Claimed complete stream lacks DONE/finish')
        flags = answer_flags(result.get('text'), arrival['expected_answer'])
        require(tuple(result.get(k) for k in ['valid_json', 'needle_correct', 'exact_schema']) == flags, 'Correctness flags disagree with retained response text')
        wrong += not (flags[1] and flags[2])
    report = {'journal': path.name, 'journal_sha256': hashlib.sha256(raw).hexdigest(),
              'policy': header['policy'], 'seed': header['seed'], 'rate': header['rate'],
              'offered': len(arrivals), 'terminal': len(terminals), 'complete': complete,
              'incorrect_complete_tasks': wrong, 'outcomes': dict(outcomes),
              'planned_seconds': header['duration_s'], 'router_pending_final': [0, 0]}
    return header, arrivals, report


def audit(directory, cycles=0):
    directory = pathlib.Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    require(manifest['source_sha256'] == SERVING_SHA, 'Frozen serving source mismatch')
    args = manifest['args']
    require(len(args['policies']) == 4 and set(args['policies']) == POLICIES, 'Four policies required')
    require(len(args['endpoints']) == 2, 'Two endpoints required')
    require(args['hours'] == 8 and args['blocks'] == 0 and args['block_seconds'] == 600, 'Held-out protocol scope mismatch')
    paths = sorted(directory.glob('block-*.jsonl'))
    require(cycles >= 0, 'Negative cycle count')
    count = cycles * 4 if cycles else len(paths)
    require(count > 0 and count % 4 == 0 and len(paths) >= count, 'Incomplete cycle population')
    rng = random.Random(args['seed'] + 100000)
    reports = []
    for cycle in range(count // 4):
        expected = list(args['policies'])
        rng.shuffle(expected)
        shift = cycle % 2
        endpoints = args['endpoints'][shift:] + args['endpoints'][:shift]
        reference = None
        for position, path in enumerate(paths[cycle*4:cycle*4+4]):
            block_index = cycle * 4 + position
            require(path.name.startswith('block-' + str(block_index).zfill(4) + '-'), 'Missing/noncontiguous block index')
            header, arrivals, report = load_block(path)
            require(header['policy'] == expected[position], 'Policy order differs from frozen shuffle')
            require(header['seed'] == args['seed'] + cycle, 'Cycle seed mismatch')
            require(abs(header['rate'] - args['rate'] * (.7 if cycle % 2 == 0 else 1.3)) < 1e-12, 'Load level mismatch')
            require(header['endpoints'] == endpoints, 'Physical endpoint rotation mismatch')
            require(header['model'] == args['model'] and header['run_id'] == args['run_id'], 'Model/run identity mismatch')
            require(header['duration_s'] == args['block_seconds'] and header['ttft_slo_s'] == args['ttft_slo'] and header['tpot_slo_s'] == args['tpot_slo'], 'Frozen duration/SLO mismatch')
            if reference is None:
                reference = arrivals
            else:
                require(set(arrivals) == set(reference), 'Paired arrival IDs differ')
                for key, row in arrivals.items():
                    other = reference[key]
                    require(abs(row['arrival_offset_s'] - other['arrival_offset_s']) < 1e-5, 'Scheduled arrival trace differs')
                    require(all(row[k] == other[k] for k in ['target_prompt_tokens_estimate', 'cold', 'intentional_cancel']), 'Demand/cancellation trace differs')
                    require(row['family'].split('-cold-block-')[0] == other['family'].split('-cold-block-')[0], 'Family trace differs')
                    if not row['cold']:
                        require(all(row[k] == other[k] for k in ['prompt_sha256', 'prefix_hash', 'expected_answer']), 'Shared prompt/answer trace differs')
            reports.append(report)
    return {'scope': 'Retained complete requested cycles only; no latency/SLO effect analysis, source mutation or duration certification.',
            'serving_sha256': SERVING_SHA, 'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
            'auditor_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            'audited_cycles': count // 4, 'ignored_other_journals': len(paths)-count,
            'all_retained_journals_audited': len(paths) == count,
            'eight_hour_requirement_proved': False,
            'paired_scope': 'Scheduled arrivals within 10 microseconds, target/family/cancel trace, and exact shared prompt hashes; intended-cold namespaces differ explicitly.',
            'correctness_scope': 'Retained text independently parsed using the frozen code-fence-normalized JSON contract; flags must agree. Wrong answers remain failures, not a reason to omit rows.',
            'totals': {k: sum(b[k] for b in reports) for k in ['offered', 'terminal', 'complete', 'incorrect_complete_tasks', 'planned_seconds']},
            'blocks': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cycles', type=int, default=0, help='0 audits all retained journals; positive audits only that many first complete cycles')
    args = parser.parse_args()
    report = audit(args.input, args.cycles)
    with pathlib.Path(args.output).open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({'audited_cycles': report['audited_cycles'], 'totals': report['totals'], 'ignored_other_journals': report['ignored_other_journals'], 'eight_hour_requirement_proved': False}))


if __name__ == '__main__':
    main()
