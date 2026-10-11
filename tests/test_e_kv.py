import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from e_kv import completion_metrics, correctness, make_cases, make_plan, fit_prompt, lease_check


class EKVTests(unittest.TestCase):
    def test_fixed_distinct_splits(self):
        cases = make_cases()
        self.assertEqual(cases, make_cases())
        self.assertEqual(len(cases), 36)
        self.assertEqual(len({x['prompt_sha256'] for x in cases}), 36)
        for c in cases:
            self.assertIn(str(c['expected']['code'] if c['task'] == 'json' else c['expected']), c['user_content'])

    def test_correctness_is_task_based(self):
        self.assertTrue(correctness({'task': 'needle', 'expected': 'ABC'}, ' ABC\n')['correct'])
        self.assertFalse(correctness({'task': 'needle', 'expected': 'ABC'}, 'Answer: ABC')['correct'])
        case = {'task': 'json', 'expected': {'code': 'X', 'count': 1, 'city': 'Oakland'}}
        self.assertTrue(correctness(case, '{"city":"Oakland","count":1,"code":"X"}')['correct'])
        self.assertFalse(correctness(case, '{"city":"Oakland","count":true,"code":"X"}')['correct'])
        self.assertFalse(correctness(case, '{"city":"Oakland","count":1,"code":"X","extra":0}')['correct'])

    def test_never_use_event_count_for_tokens(self):
        row = {'usage': None, 'content_events': 200, 'first_content_s': 1, 'last_content_s': 10}
        self.assertIsNone(completion_metrics(row, True)['tpot_visible_s'])
        row['usage'] = {'completion_tokens': 10, 'prompt_tokens': 100}
        self.assertIsNone(completion_metrics(row)['tpot_visible_s'])
        self.assertEqual(completion_metrics(row, True)['tpot_visible_s'], 1)
        row['usage']['completion_tokens_details'] = {'reasoning_tokens': 3}
        self.assertIsNone(completion_metrics(row, True)['tpot_visible_s'])
        row['usage']['completion_tokens_details'] = {'reasoning_tokens': 0}
        row['reasoning_text'] = 'Hidden output'
        self.assertIsNone(completion_metrics(row, True)['tpot_visible_s'])

    def test_all_precisions_on_both_nodes_each_block(self):
        plan = make_plan(blocks=4)
        self.assertEqual(len(plan), 12)
        for block in range(4):
            pairs = {(x['node'], x['precision']) for row in plan if row['block'] == block for x in row['assignments']}
            self.assertEqual(len(pairs), 6)

    def test_context_fit_preserves_answer_and_manifest(self):
        case = make_cases()[0]
        original = case['user_content']
        payload = {'model': 'test', 'messages': [{'role': 'user', 'content': original}]}
        def count(_client, _endpoint, trial):
            return {'status': 'counted', 'input_tokens': len(trial['messages'][0]['content']) // 4}
        with patch('e_kv.count_prompt', side_effect=count):
            fitted, evidence = fit_prompt(None, 'unused', payload, case, 256, 300)
        self.assertEqual(evidence['input_tokens'], 256)
        self.assertIn('AUTHORITATIVE_CODE: ' + case['expected'], fitted['messages'][0]['content'])
        self.assertIn('TASK: ', fitted['messages'][0]['content'])
        self.assertEqual(case['user_content'], original)
        self.assertEqual(evidence['original_prompt_sha256'], case['prompt_sha256'])

    def test_count_failure_does_not_fit(self):
        case = make_cases()[0]
        with patch('e_kv.count_prompt', return_value={'status': 'failed', 'error': 'missing'}):
            with self.assertRaises(ValueError):
                fit_prompt(None, 'unused', {'model': 'test'}, case, 256, 300)

    def test_lease_config_expiry_and_revoke(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lease.json'
            lease = {'task': 'E', 'run_id': 'test', 'expires_at': 100,
                     'nodes': {'spark2': {'allow': True, 'precision': 'int8', 'endpoint': 'http://localhost/v1'}}}
            path.write_text(json.dumps(lease))
            self.assertTrue(lease_check(path, 'spark2', 'int8', 'http://localhost/v1', 'test', now=90)['valid'])
            self.assertFalse(lease_check(path, 'spark2', 'fp16', 'http://localhost/v1', 'test', now=90)['valid'])
            self.assertFalse(lease_check(path, 'spark2', 'int8', 'http://localhost/v1', 'test', now=101)['valid'])
            lease['nodes']['spark2']['allow'] = False
            path.write_text(json.dumps(lease))
            self.assertFalse(lease_check(path, 'spark2', 'int8', 'http://localhost/v1', 'test', now=90)['valid'])


if __name__ == '__main__':
    unittest.main()
