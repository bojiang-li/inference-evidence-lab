import json
import tempfile
import unittest
import uuid
from pathlib import Path
from analyze_a import analyze


class AnalysisTests(unittest.TestCase):
    def fixture(self,stop=False,wall=10,pending=False):
        root=Path(__file__).parent/'fixture-runs'/uuid.uuid4().hex;root.mkdir(parents=True)
        p=root/'block-0000-round_robin.jsonl'
        rows=[{'type':'block','policy':'round_robin','seed':1,'rate':.3,'duration_s':10,'endpoints':['node0','node1']},
              {'type':'arrival','arrival_id':'r','target_prompt_tokens_estimate':2048,'cold':False}]
        if not pending:
            rows.append({'type':'terminal','arrival_id':'r','target_prompt_tokens_estimate':2048,'cold':False,
                         'result':{'status':'client_overload','slo_pass':False}})
        rows.append({'type':'summary','wall_seconds':wall,'stopped_by_lease':stop})
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows));return p

    def test_full_window_qualified(self):
        r=analyze(self.fixture())
        self.assertTrue(r['full_planned_window_qualified']);self.assertEqual(r['actual_exposure_seconds'],10)

    def test_lease_truncation_retains_offered_and_terminal(self):
        r=analyze(self.fixture(stop=True,wall=4))
        self.assertFalse(r['full_planned_window_qualified']);self.assertTrue(r['stopped_by_lease'])
        self.assertEqual(r['offered'],1);self.assertEqual(r['terminal'],1)
        self.assertIsNone(r['actual_exposure_seconds']);self.assertEqual(r['exposure_upper_bound_seconds'],4)

    def test_early_window_without_lease_flag_unqualified(self):
        r=analyze(self.fixture(wall=8))
        self.assertFalse(r['full_planned_window_qualified'])

    def test_pending_arrival_not_qualified(self):
        r=analyze(self.fixture(pending=True))
        self.assertFalse(r['full_planned_window_qualified']);self.assertEqual(r['unresolved_arrivals'],1)

if __name__=='__main__':unittest.main()
