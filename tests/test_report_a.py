"""CPU fixtures test accounting, qualification, paired units, and frozen source checks."""
import copy
import statistics
import unittest
import report_a as r


def fixture(n=6):
    args={'policies':r.POLICIES[:], 'seed':20268011, 'rate':1.68, 'ttft_slo':10,
          'tpot_slo':.15, 'hours':8, 'block_seconds':600, 'endpoints':['a','b']}
    protocol={'selected_before_holdout':True,'holdout_seed':args['seed'],'base_rate':1.68,
              'ttft_slo_s':10,'tpot_slo_s':.15,'hours_requested':8,'load_levels':[.7,1.3],'burst_multiplier':2}
    manifest={'args':args,'source_sha256':r.SERVING_SHA,'created_at':1791705295.0398884}
    blocks=[]
    for cycle in range(n):
        for index,p in enumerate(r.POLICIES):
            slo=300 + cycle*3 + index*(cycle+1)
            blocks.append({'file':f'block-{cycle:04}-{index}.jsonl','policy':p,'seed':args['seed']+cycle,
                'rate':1.68*(.7 if cycle%2==0 else 1.3),'planned_seconds':600,
                'endpoints':['a','b'] if cycle%2==0 else ['b','a'],
                'offered':1000,'terminal':1000,'unresolved_arrivals':0,'complete':600,'slo_pass':slo,
                'complete_correct_json':580,'complete_with_done_and_finish':600,'complete_with_usage':600,
                'outcomes':{'complete':600,'client_overload':385,'intentional_cancel':15},
                'summary_present':True,'stopped_by_lease':False,'actual_exposure_seconds':600,
                'full_planned_window_qualified':True,'goodput_per_planned_second':slo/600})
    data={'status':'analyzed_retained_journals','blocks':blocks,'pending_journal_blocks':0,
          'truncated_or_unqualified_blocks':0,'policies':{},'comparisons':[]}
    for p in r.POLICIES:
        chosen=[b for b in blocks if b['policy']==p]
        data['policies'][p]={'blocks':n,'offered':1000*n,'complete':600*n,
             'slo_pass':sum(b['slo_pass'] for b in chosen),'planned_seconds':600*n,
             'outcomes':{'complete':600*n,'client_overload':385*n,'intentional_cancel':15*n}}
    for index,p in enumerate(r.POLICIES[1:],1):
        diffs=[index*(cycle+1)/600 for cycle in range(n)]
        data['comparisons'].append({'policy':p,'baseline':'round_robin','matched_complete_cycles':n,
            'mean_goodput_difference':statistics.mean(diffs),'paired_cycle_bootstrap95':[.001,.09] if n>=3 else None,
            'scope':'Fixture unadjusted result preserved verbatim'})
    completion={'started_at_utc':'2026-10-11T07:54:55.009399Z',
                'finished_at_utc':'2026-10-11T15:55:00.009399Z','exit_code':0,'analyzer_sha256':r.ANALYZER_SHA}
    return data,protocol,manifest,completion


class FinalReportTests(unittest.TestCase):
    def test_balanced_outcomes_and_clock_scope(self):
        data,p,m,c=fixture()
        out=r.build_report(data,p,m,c,True)
        self.assertEqual(out['status'],'qualified_completed_8h')
        self.assertEqual(out['wall_seconds'],28805)
        self.assertEqual(out['known_exact_offered_exposure_seconds_all_blocks'],14400)
        self.assertEqual(out['non_offered_wall_overhead_seconds'],14405)
        self.assertIsNone(out['drain_seconds'])
        for s in out['policies'].values():
            self.assertEqual(s['offered'],s['slo_pass']+s['correct_without_recorded_slo_pass']+
                s['complete_wrong_quality']+s['outcomes']['client_overload']+s['outcomes']['intentional_cancel']+s['pending'])

    def test_no_clock_or_hash_attestation_no_eight_hour_claim(self):
        data,p,m,c=fixture()
        out=r.build_report(data,p,m)
        self.assertEqual(out['status'],'qualification_incomplete')
        self.assertIsNone(out['wall_seconds'])
        self.assertIsNone(out['non_offered_wall_overhead_seconds'])
        self.assertEqual(r.build_report(data,p,m,c,False)['status'],'qualification_incomplete')
        c['finished_at_utc']='2026-10-11T13:55:00Z'
        self.assertEqual(r.build_report(data,p,m,c,True)['status'],'qualification_incomplete')
        c['finished_at_utc']='2026-10-11T08:55:00Z'
        with self.assertRaisesRegex(ValueError,'wall shorter'):r.build_report(data,p,m,c,True)

    def test_all_offered_ledger_and_quality_contradictions_fail(self):
        for field,value in [('terminal',999),('complete_correct_json',0),('complete_with_done_and_finish',599)]:
            data,p,m,_=fixture();data['blocks'][0][field]=value
            with self.assertRaises(ValueError):r.build_report(data,p,m)

    def test_frozen_protocol_and_source_fail_closed(self):
        data,p,m,c=fixture();m['source_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'SHA'):r.build_report(data,p,m,c,True)
        data,p,m,c=fixture();p['base_rate']=.3
        with self.assertRaisesRegex(ValueError,'mismatch'):r.build_report(data,p,m,c,True)
        data,p,m,c=fixture();c['analyzer_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'SHA'):r.build_report(data,p,m,c,True)

    def test_familywise_uses_cycles_preserves_v4_and_fixed_seed(self):
        data,p,m,_=fixture()
        out=r.build_report(data,p,m)
        out2=r.build_report(data,p,m)
        self.assertEqual(out['comparisons'],out2['comparisons'])
        for i,c in enumerate(out['comparisons']):
            self.assertEqual(c['v4_unadjusted_preserved'],data['comparisons'][i])
            self.assertEqual(c['all_policy_matched_cycles'],6)
            self.assertEqual(len(c['matched_cycle_differences_req_per_s']),6)
            self.assertEqual(c['draws'],10000)
            self.assertEqual(c['tails'],[.05/6,1-.05/6])
            self.assertTrue(c['bonferroni_familywise95_paired_cycle_percentile'][0]<=c['mean_difference_req_per_s']<=c['bonferroni_familywise95_paired_cycle_percentile'][1])

    def test_fewer_than_three_cycles_yields_no_interval(self):
        data,p,m,_=fixture(2)
        out=r.build_report(data,p,m)
        self.assertTrue(all(c['bonferroni_familywise95_paired_cycle_percentile'] is None for c in out['comparisons']))

    def test_truncated_retained_in_description_excluded_from_family(self):
        data,p,m,c=fixture()
        b=data['blocks'][3];b.update(stopped_by_lease=True,actual_exposure_seconds=None,full_planned_window_qualified=False)
        data['truncated_or_unqualified_blocks']=1
        comp=data['comparisons'][2];comp['matched_complete_cycles']=5
        comp['mean_goodput_difference']=statistics.mean([3*(cycle+1)/600 for cycle in range(1,6)])
        out=r.build_report(data,p,m,c,True)
        self.assertEqual(out['status'],'qualification_incomplete')
        self.assertEqual(out['policies']['cache_feedback']['offered'],6000)
        self.assertEqual(len(out['all_policy_matched_cycles']),5)
        self.assertIsNone(out['non_offered_wall_overhead_seconds'])
        self.assertTrue(all(co['all_policy_matched_cycles']==5 for co in out['comparisons']))

    def test_pending_and_duplicate_trace_cannot_be_hidden(self):
        data,p,m,_=fixture();data['blocks'][0]['unresolved_arrivals']=1
        with self.assertRaisesRegex(ValueError,'Unbalanced'):r.build_report(data,p,m)
        data,p,m,_=fixture();data['blocks'].append(copy.deepcopy(data['blocks'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):r.build_report(data,p,m)


if __name__=='__main__':unittest.main()
