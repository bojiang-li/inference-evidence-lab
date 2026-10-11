import json
import random
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from a_serving import Router, SSEParser, correctness, count_prompt, lease_ok, make_request, run_block, stream_request


def frame(obj):
    return ('data: '+json.dumps(obj)+'\n\n').encode()


class Fixture(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path.endswith('/messages/count_tokens'):
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Connection','close')
            self.end_headers()
            self.wfile.write(b'{"input_tokens": 20}')
            return
        mode=body.get('fixture','complete')
        if mode=='http_error':
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.send_header('Connection','close')
        self.end_headers()
        try:
            if mode=='reasoning':
                self.wfile.write(frame({'choices':[{'index':0,'delta':{'reasoning_content':'Thoughts'},'finish_reason':None}]}))
            self.wfile.write(frame({'choices':[{'index':0,'delta':{'content':'{"answer":"CODE123"}'},'finish_reason':None}]}))
            if mode!='missing_finish':
                self.wfile.write(frame({'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]}))
            self.wfile.write(frame({'choices':[],'usage':{'prompt_tokens':20,'completion_tokens':5,'prompt_tokens_details':{'cached_tokens':10}}}))
            if mode!='missing_done':
                self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):
            pass


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Fixture)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.endpoint=f'http://127.0.0.1:{cls.server.server_port}/v1'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_complete_stream_requires_done_finish_and_retains_usage(self):
        r=stream_request(self.endpoint,{'fixture':'complete'})
        self.assertEqual(r['status'],'complete')
        self.assertTrue(r['done'])
        self.assertEqual(r['finish_reason'],'stop')
        self.assertEqual(r['completion_tokens'],5)
        self.assertEqual(r['cached_tokens'],10)
        self.assertEqual(r['content_events'],1)
        self.assertTrue(correctness(r['text'],'CODE123')['needle_correct'])

    def test_reasoning_is_preserved_without_content_only_tpot(self):
        r=stream_request(self.endpoint,{'fixture':'reasoning'})
        self.assertEqual(r['reasoning_text'],'Thoughts')
        self.assertIsNone(r['tpot_s'])
        self.assertIsNotNone(r['server_token_normalized_output_interval_s'])
        self.assertLessEqual(r['first_output_s'],r['first_content_s'])

    def test_count_prompt_contract(self):
        r=count_prompt(self.endpoint,'fake',[{'role':'system','content':'s'},{'role':'user','content':'u'}])
        self.assertEqual(r['status'],'counted')
        self.assertEqual(r['input_tokens'],20)

    def test_absent_lease_is_denied(self):
        self.assertFalse(lease_ok('/no/such/a-lease.json','test'))

    def test_all_arrival_terminal_accounting_in_fake_replay(self):
        from analyze_a import analyze
        root=Path(__file__).parent/'fixture-runs'/uuid.uuid4().hex
        root.mkdir(parents=True)
        lease=root/'lease.json'
        lease.write_text(json.dumps({'run_id':'fixture','allow_a':True,'expires_at':time.time()+30}))
        args=SimpleNamespace(lease=str(lease),run_id='fixture',seed=1,
              policies=['round_robin'],endpoints=[self.endpoint,self.endpoint],output=str(root),
              cache_ttl=5,prefill_tokens_per_s=1000,block_seconds=.2,model='fake',
              client_cap=2,output_tokens=64,request_timeout=5,extra_body='',families=2,
              cold_fraction=.2,cancel_fraction=.1,ttft_slo=10,tpot_slo=.15,
              prompt_count_map='{"2048":2889,"8192":9842}')
        run_block(args,'round_robin',0,40)
        result=analyze(root/'block-0000-round_robin.jsonl')
        self.assertGreater(result['offered'],0)
        self.assertEqual(result['offered'],result['terminal'])
        self.assertEqual(result['unresolved_arrivals'],0)
        self.assertTrue(result['summary_present'])

    def test_missing_done_is_not_completion(self):
        r=stream_request(self.endpoint,{'fixture':'missing_done'})
        self.assertEqual(r['status'],'incomplete_stream')
        self.assertFalse(r['done'])
        self.assertEqual(r['finish_reason'],'stop')

    def test_done_without_finish_is_not_completion(self):
        r=stream_request(self.endpoint,{'fixture':'missing_finish'})
        self.assertEqual(r['status'],'missing_finish_reason')
        self.assertTrue(r['done'])

    def test_intentional_cancel_is_separate(self):
        r=stream_request(self.endpoint,{},cancel_after_events=1)
        self.assertEqual(r['status'],'intentional_cancel')
        self.assertFalse(r['done'])

    def test_http_error_is_terminal(self):
        r=stream_request(self.endpoint,{'fixture':'http_error'})
        self.assertEqual(r['status'],'http_error')
        self.assertEqual(r['http_status'],503)

    def test_sse_multiline_and_comments(self):
        p=SSEParser()
        self.assertIsNone(p.feed(b': heartbeat\r\n'))
        self.assertIsNone(p.feed(b'data: {"choices":\r\n'))
        self.assertIsNone(p.feed(b'data: []}\r\n'))
        self.assertEqual(p.feed(b'\r\n'),{'object':{'choices':[]}})

    def test_content_schema(self):
        self.assertFalse(correctness('{"answer":"bad"}','expected')['needle_correct'])
        self.assertFalse(correctness('{"answer":"ok","extra":1}','ok')['exact_schema'])
        self.assertFalse(correctness('plain text','ok')['valid_json'])

    def test_feedback_requires_observed_hit_and_expires(self):
        r=Router(2,'cache_feedback',cache_ttl=5)
        n,_=r.reserve('p',2048,now=1)
        r.release(n,'p',2048,{'status':'complete','duration_s':1,'cached_tokens':0},now=2)
        self.assertNotIn((n,'p'),r.cache)
        n,_=r.reserve('p',2048,now=3)
        r.release(n,'p',2048,{'status':'complete','duration_s':1,'cached_tokens':1024},now=4)
        self.assertTrue(r.cache[(n,'p')]['verified'])
        n2,_=r.reserve('p',2048,now=5)
        self.assertEqual(n,n2)
        r.release(n2,'p',2048,{'status':'timeout','duration_s':0},now=6)
        self.assertNotIn((n,'p'),r.cache)
        self.assertEqual(r.pending,[0,0])

    def test_leastqueue_routes_away_and_underflow_rejected(self):
        r=Router(2,'least_queue')
        n,_=r.reserve('x',2048)
        m,_=r.reserve('x',2048)
        self.assertNotEqual(n,m)
        for a in [n,m]:r.release(a,'x',2048,{'status':'timeout'})
        with self.assertRaises(RuntimeError):r.release(n,'x',2048,{'status':'timeout'})

    def test_common_prefix_stable_and_trace_deterministic(self):
        one=make_request('family0',2048,'req1',random.Random(1))
        two=make_request('family0',2048,'req2',random.Random(2))
        self.assertEqual(one[2],two[2])
        self.assertNotEqual(one[1],two[1])
        self.assertNotEqual(one[2],make_request('family0',8192,'req1',random.Random(1))[2])


if __name__=='__main__':unittest.main()
