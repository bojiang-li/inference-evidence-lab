"""CPU corruption fixtures for retained replay evidence, not model tests."""
import copy
import json
import pathlib
import random
import tempfile
import unittest

from audit_cycles import audit, SERVING_SHA


class EvidenceFixtures(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix='audit-fixture-', dir=pathlib.Path(__file__).parent))
        self.args = {'policies': ['round_robin', 'least_queue', 'cache_aware', 'cache_feedback'],
                     'endpoints': ['first', 'second'], 'seed': 20268011, 'rate': 1.68,
                     'hours': 8, 'blocks': 0, 'block_seconds': 600,
                     'model': 'fixture', 'run_id': 'fixture', 'ttft_slo': 10, 'tpot_slo': .15}
        (self.root / 'manifest.json').write_text(json.dumps({'args': self.args, 'source_sha256': SERVING_SHA}))
        self.paths = []
        rng = random.Random(self.args['seed'] + 100000)
        for cycle in range(2):
            policies = list(self.args['policies'])
            rng.shuffle(policies)
            endpoints = self.args['endpoints'] if cycle % 2 == 0 else list(reversed(self.args['endpoints']))
            for policy in policies:
                index = len(self.paths)
                rows = [{'type': 'block', 'policy': policy, 'seed': self.args['seed'] + cycle,
                         'rate': self.args['rate'] * (.7 if cycle % 2 == 0 else 1.3),
                         'endpoints': endpoints, 'duration_s': 600, 'model': 'fixture',
                         'run_id': 'fixture', 'ttft_slo_s': 10, 'tpot_slo_s': .15}]
                for number in range(2):
                    cold = number == 1
                    expected = 'CODE' + str(index) if cold else 'CODEWARM'
                    row = {'type': 'arrival', 'arrival_id': str(cycle) + '-' + str(number),
                           'arrival_offset_s': float(number), 'family': 'family-0' + ('-cold-block-' + str(index) + '-1' if cold else ''),
                           'target_prompt_tokens_estimate': 8192, 'prefix_hash': 'cold-' + str(index) if cold else 'warm',
                           'prompt_sha256': 'cold-' + str(index) if cold else 'warm', 'prefix_characters': 100,
                           'total_characters': 120, 'cold': cold, 'intentional_cancel': False, 'expected_answer': expected}
                    rows.append(row)
                    terminal = dict(row, type='terminal', result={'status': 'complete', 'done': True, 'finish_reason': 'stop',
                                                                'text': json.dumps({'answer': expected}), 'valid_json': True,
                                                                'needle_correct': True, 'exact_schema': True})
                    rows.append(terminal)
                rows.append({'type': 'summary', 'policy': policy, 'planned_seconds': 600,
                             'wall_seconds': 601, 'offered': 2, 'terminal': 2,
                             'router_pending_final': [0, 0], 'stopped_by_lease': False,
                             'outcomes': {'complete': 2}})
                path = self.root / ('block-' + str(index).zfill(4) + '-' + policy + '.jsonl')
                self.write(path, rows)
                self.paths.append(path)

    def read(self, index):
        return [json.loads(x) for x in self.paths[index].read_text().splitlines()]

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(x) + '\n' for x in rows))

    def test_matching_cycles_and_distinct_cold_prompts_pass_without_duration_claim(self):
        result = audit(self.root)
        self.assertEqual(result['totals']['complete'], 16)
        self.assertEqual(result['audited_cycles'], 2)
        self.assertTrue(result['all_retained_journals_audited'])
        self.assertFalse(result['eight_hour_requirement_proved'])

    def test_shared_prompt_corruption_is_detected_despite_equal_counts(self):
        rows = self.read(1)
        for row in rows:
            if row['type'] in ['arrival', 'terminal'] and not row['cold']:
                row['prompt_sha256'] = 'altered'
        self.write(self.paths[1], rows)
        with self.assertRaisesRegex(ValueError, 'Shared prompt'):
            audit(self.root)

    def test_schedule_corruption_is_detected_despite_equal_counts(self):
        rows = self.read(1)
        for row in rows:
            if row['type'] in ['arrival', 'terminal']:
                row['arrival_offset_s'] += .001
        self.write(self.paths[1], rows)
        with self.assertRaisesRegex(ValueError, 'Scheduled arrival'):
            audit(self.root)

    def test_correctness_flag_cannot_hide_wrong_retained_text(self):
        rows = self.read(0)
        rows[2]['result']['text'] = '{"answer":"WRONG"}'
        self.write(self.paths[0], rows)
        with self.assertRaisesRegex(ValueError, 'Correctness flags'):
            audit(self.root)

    def test_honest_wrong_answer_is_retained_as_failure(self):
        rows = self.read(0)
        rows[2]['result'].update(text='{"answer":"WRONG"}', needle_correct=False)
        self.write(self.paths[0], rows)
        result = audit(self.root)
        self.assertEqual(result['totals']['complete'], 16)
        self.assertEqual(result['totals']['incorrect_complete_tasks'], 1)

    def test_duplicate_terminal_is_rejected(self):
        rows = self.read(0)
        rows.insert(-1, copy.deepcopy(rows[2]))
        self.write(self.paths[0], rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate terminal'):
            audit(self.root)

    def test_incomplete_stream_is_rejected(self):
        rows = self.read(0)
        rows[2]['result']['done'] = False
        self.write(self.paths[0], rows)
        with self.assertRaisesRegex(ValueError, 'DONE/finish'):
            audit(self.root)

    def test_unrevoked_but_short_window_is_rejected(self):
        rows = self.read(0)
        rows[-1]['wall_seconds'] = 599
        self.write(self.paths[0], rows)
        with self.assertRaisesRegex(ValueError, 'Truncated exposure'):
            audit(self.root)

    def test_endpoint_rotation_corruption_is_rejected(self):
        rows = self.read(4)
        rows[0]['endpoints'] = list(self.args['endpoints'])
        self.write(self.paths[4], rows)
        with self.assertRaisesRegex(ValueError, 'endpoint rotation'):
            audit(self.root)

    def test_partial_population_never_silently_becomes_full_run(self):
        (self.root / 'block-0008-active.jsonl').write_text('{')
        with self.assertRaisesRegex(ValueError, 'cycle population'):
            audit(self.root)
        result = audit(self.root, cycles=2)
        self.assertEqual(result['ignored_other_journals'], 1)
        self.assertFalse(result['all_retained_journals_audited'])
        self.assertFalse(result['eight_hour_requirement_proved'])


if __name__ == '__main__':
    unittest.main()
