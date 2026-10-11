import copy
import hashlib
import unittest
from analyze_e import inspect_row, compare_pairs, memory_summary


def fixture(case='case-0', precision='fp16', node='spark1', phase='first_pass', text='ABC', latency=2):
    prompt = 'unique prompt ' + case
    return {'case_id': case, 'node': node, 'precision': precision, 'cache_phase': phase, 'block': 0,
            'expected': 'ABC', 'task': 'needle', 'length_label': 'short', 'actual_user_content': prompt,
            'prompt_count_evidence': {'input_tokens': 1024, 'fitted_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()},
            'model': 'same', 'engine_commit': 'same', 'decode_settings': {'temperature': 0},
            'task_success': text == 'ABC', '_source': 'run-' + node + '-' + precision,
            'result': {'text': text, 'status': 'complete', 'done': True, 'finish_reason': 'stop',
                       'usage': {'prompt_tokens': 1024, 'completion_tokens': 5}, 'duration_s': latency+1},
            'metrics': {'ttft_content_s': latency, 'tpot_visible_s': .01}, 'started_unix_s': 100}


def checked(*args, **kwargs):
    row = fixture(*args, **kwargs)
    row['_check'] = inspect_row(row)
    return row


class AnalysisTests(unittest.TestCase):
    def test_discordant_task_quality_kept_but_latency_conditional(self):
        rows = [checked(precision='fp16'), checked(precision='int8', text='WRONG')]
        result = compare_pairs(rows, 'fp16', 'int8')
        self.assertEqual(result['paired_task_correctness_table']['baseline_right_candidate_wrong'], 1)
        self.assertEqual(result['latency_subset_exclusions']['not_both_task_success'], 1)
        self.assertEqual(len(result['discordant_cases']), 1)

    def test_hash_and_count_mismatch_explicit(self):
        left, right = checked(precision='fp16'), checked(precision='int8')
        right['actual_user_content'] += 'changed'
        right['_check'] = inspect_row(right)
        result = compare_pairs([left, right], 'fp16', 'int8')
        self.assertEqual(result['input_matched_raw_pairs'], 0)
        self.assertIn('missing_or_invalid_actual_fitted_prompt_hash', result['input_pair_exclusions'])
        right = checked(precision='int8')
        right['result']['usage']['prompt_tokens'] = 1000
        right['_check'] = inspect_row(right)
        self.assertIn('count_usage_mismatch_or_missing', compare_pairs([left, right], 'fp16', 'int8')['input_pair_exclusions'])

    def test_ci_requires_three_cases_not_nodes_or_cache_repeats(self):
        rows = [checked(case='only-case', precision=precision, node=node, phase=phase)
                for precision in ['fp16', 'int8'] for node in ['spark1', 'spark2'] for phase in ['first_pass', 'repeated_pass']]
        result = compare_pairs(rows, 'fp16', 'int8')
        self.assertEqual(result['input_matched_raw_pairs'], 4)
        for item in result['latency_slices']:
            metric = item['metrics']['ttft_s']
            self.assertEqual(metric['case_clusters'], 1)
            self.assertIsNone(metric['ci95_seconds'])

    def test_three_case_cluster_ci_uses_paired_delta(self):
        rows = [checked(case='case-' + str(case), precision=precision, latency=2 if precision=='fp16' else 1)
                for case in range(3) for precision in ['fp16', 'int8']]
        result = compare_pairs(rows, 'fp16', 'int8')
        metric = result['latency_slices'][0]['metrics']['ttft_s']
        self.assertEqual(metric['ci95_seconds'], [-1, -1])
        self.assertEqual(metric['case_clusters'], 3)

    def test_memory_windows_do_not_span_unrelated_configs(self):
        rows = [checked(precision='fp16')]
        sample = {'node': 'spark1', 'unix': 101, 'service': 'ActiveState=active\nMemoryCurrent=999', 'mem_available_bytes': 1000}
        outside = copy.deepcopy(sample)
        outside['unix'] = 1000
        result = memory_summary(rows, [sample, outside])
        self.assertEqual(result['groups'][0]['snapshots_in_file_run_windows'], 1)
        self.assertTrue(result['not_direct_kv_allocation_or_gpu_payload'])


if __name__ == '__main__':
    unittest.main()
