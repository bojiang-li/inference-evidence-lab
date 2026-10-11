#!/usr/bin/env python3
"""Finite, lease-controlled two-replica inference replay; Python standard library only."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import random
import signal
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class SSEParser:
    """Parse complete SSE frames. Network reader preserves framing and byte bounds."""
    def __init__(self):
        self.lines = []

    def feed(self, raw):
        line = raw.decode('utf-8', errors='strict').rstrip('\r\n')
        if line == '':
            data = '\n'.join(self.lines)
            self.lines = []
            if not data:
                return None
            if data.strip() == '[DONE]':
                return {'done': True}
            obj = json.loads(data)
            if not isinstance(obj, dict):
                raise ValueError('SSE data must be an object')
            return {'object': obj}
        if line.startswith('data:'):
            value = line[5:]
            self.lines.append(value[1:] if value.startswith(' ') else value)
        return None


def connection(endpoint, timeout):
    u = urlsplit(endpoint)
    if u.scheme not in ('http', 'https') or not u.hostname or u.username:
        raise ValueError('Use a plain explicit http(s) endpoint')
    cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
    return cls(u.hostname, u.port, timeout=timeout), u.path.rstrip('/')


def fetch_model(endpoint):
    conn, base = connection(endpoint, 15)
    try:
        conn.request('GET', base + '/models')
        response = conn.getresponse()
        body = response.read(65537)
        if response.status != 200 or len(body) > 65536:
            raise ValueError('Model endpoint failed or exceeded limit')
        obj = json.loads(body)
        return [x['id'] for x in obj['data']]
    finally:
        conn.close()


def count_prompt(endpoint, model, messages, timeout=15):
    """Strata Anthropic count API, with equivalent simple-text no-thinking template."""
    system = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
    payload = {'model':model,'system':system,'messages':[m for m in messages if m['role']!='system'],
               'thinking':{'type':'disabled'}}
    conn,base=connection(endpoint,timeout)
    try:
        conn.request('POST',base+'/messages/count_tokens',json.dumps(payload).encode(),
                     {'Content-Type':'application/json'})
        response=conn.getresponse()
        body=response.read(65537)
        if response.status!=200 or len(body)>65536:
            return {'status':'unsupported_or_error','http_status':response.status}
        obj=json.loads(body)
        n=obj.get('input_tokens')
        if not isinstance(n,int) or n<=0:
            return {'status':'invalid_count'}
        return {'status':'counted','input_tokens':n,'scope':'Strata Anthropic equivalent simple-text no-thinking template'}
    except (OSError,ValueError,http.client.HTTPException) as exc:
        return {'status':'error','error':repr(exc)}
    finally:
        conn.close()


def stream_request(endpoint, payload, timeout=120, cancel_after_events=None,
                   cancel_after_seconds=None, stop=None):
    """Return one terminal outcome, including incomplete streams and intentional cancellation."""
    begin = time.monotonic()
    result = {'status': 'exception', 'done': False, 'finish_reason': None,
              'usage': None, 'content_events': 0, 'first_content_s': None,
              'last_content_s': None, 'first_output_s': None, 'last_output_s': None,
              'http_status': None, 'bytes_received': 0}
    conn, base = connection(endpoint, min(timeout, cancel_after_seconds or timeout))
    content, reasoning = [], []
    parser = SSEParser()
    deadline = begin + timeout
    cancel_deadline = begin + cancel_after_seconds if cancel_after_seconds else None
    response = None
    try:
        conn.request('POST', base + '/chat/completions',
                     json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        response = conn.getresponse()
        result['http_status'] = response.status
        if response.status != 200:
            result.update(status='http_error', error_body=response.read(2048).decode(errors='replace'))
            return result
        content_type = response.getheader('Content-Type', '').lower()
        if 'text/event-stream' not in content_type:
            result.update(status='protocol_error', error='Response is not SSE')
            return result
        # http.client may detach its connection socket for Connection: close.
        sock = conn.sock
        if sock is None:
            sock = getattr(getattr(response.fp, 'raw', None), '_sock', None)
        while True:
            now = time.monotonic()
            if stop and stop.is_set():
                result['status'] = 'lease_cancelled'
                break
            if cancel_deadline and now >= cancel_deadline:
                result['status'] = 'intentional_cancel'
                break
            if now >= deadline:
                result['status'] = 'timeout'
                break
            read_until = min(deadline, cancel_deadline or deadline)
            if sock:
                sock.settimeout(max(0.001, read_until - now))
            raw = response.readline(65537)
            if not raw:
                result['status'] = 'incomplete_stream'
                break
            result['bytes_received'] += len(raw)
            if len(raw) > 65536 or result['bytes_received'] > 2 * 1024 * 1024:
                result['status'] = 'response_limit'
                break
            event = parser.feed(raw)
            if not event:
                continue
            if event.get('done'):
                result['done'] = True
                result['status'] = 'complete' if result['finish_reason'] else 'missing_finish_reason'
                break
            obj = event['object']
            if obj.get('error'):
                result.update(status='stream_error', error=obj['error'])
                break
            if obj.get('usage') is not None:
                result['usage'] = obj['usage']
            if obj.get('timings') is not None:
                result['server_timings'] = obj['timings']
            for choice in obj.get('choices', []):
                if choice.get('index', 0) != 0:
                    continue
                delta = choice.get('delta', {})
                token_content = delta.get('content')
                reason = delta.get('reasoning_content')
                if token_content or reason:
                    output_tick = time.monotonic()-begin
                    if result['first_output_s'] is None:
                        result['first_output_s'] = output_tick
                    result['last_output_s'] = output_tick
                if isinstance(token_content, str) and token_content:
                    content.append(token_content)
                    tick = time.monotonic() - begin
                    result['content_events'] += 1
                    if result['first_content_s'] is None:
                        result['first_content_s'] = tick
                    result['last_content_s'] = tick
                if isinstance(reason, str):
                    reasoning.append(reason)
                if choice.get('finish_reason') is not None:
                    result['finish_reason'] = choice['finish_reason']
            if cancel_after_events and result['content_events'] >= cancel_after_events:
                result['status'] = 'intentional_cancel'
                break
    except TimeoutError as exc:
        result['status'] = ('lease_cancelled' if stop and stop.is_set() else
                            'intentional_cancel' if cancel_deadline and time.monotonic() >= cancel_deadline else 'timeout')
        result['error'] = repr(exc)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        result.update(status='exception', error=repr(exc))
    finally:
        if response:
            response.close()
        conn.close()
        result['duration_s'] = time.monotonic() - begin
        result['text'] = ''.join(content)
        result['text_sha256'] = digest(result['text'])
        result['reasoning_text'] = ''.join(reasoning)
        result['reasoning_sha256'] = digest(''.join(reasoning))
        usage = result.get('usage')
        result['completion_tokens'] = usage.get('completion_tokens') if isinstance(usage, dict) else None
        details = usage.get('prompt_tokens_details', {}) if isinstance(usage, dict) else {}
        result['cached_tokens'] = details.get('cached_tokens') if isinstance(details, dict) else None
        n = result['completion_tokens']
        first, last = result['first_content_s'], result['last_content_s']
        result['tpot_s'] = ((last-first)/(n-1) if isinstance(n,int) and n>1 and first is not None
                            and not result['reasoning_text'] else None)
        any_first,any_last=result['first_output_s'],result['last_output_s']
        result['server_token_normalized_output_interval_s'] = ((any_last-any_first)/(n-1)
                            if isinstance(n,int) and n>1 and any_first is not None else None)
        result['tpot_scope'] = 'Visible-content interval / server completion_tokens-1 only when no reasoning text; an approximation, not token timestamp measurement'
    return result


class Router:
    def __init__(self, count, policy, cache_ttl=60, prefill_tokens_per_s=1000):
        self.count, self.policy = count, policy
        self.pending = [0] * count
        self.affinity = {}
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.prefill_tokens_per_s = prefill_tokens_per_s
        self.service_ema = [1.0] * count
        self.serial = 0
        self.lock = threading.Lock()

    def reserve(self, family, prompt_target, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.policy == 'round_robin':
                node = self.serial % self.count
            elif self.policy == 'least_queue':
                node = min(range(self.count), key=lambda n: (self.pending[n], (n-self.serial) % self.count))
            else:
                if family not in self.affinity:
                    self.affinity[family] = int(digest(family)[:12], 16) % self.count
                preferred = self.affinity[family]
                costs = []
                for n in range(self.count):
                    entry = self.cache.get((n, family))
                    if self.policy == 'cache_aware':
                        # Explicit heuristic baseline: completed prefix affinity, no TTL/negative feedback.
                        hit = entry is not None
                    else:
                        # Only observed positive cached_tokens qualifies as an empirical hit.
                        hit = entry is not None and entry['verified'] and now-entry['at'] <= self.cache_ttl
                    saved = min(prompt_target, entry['tokens']) if hit else 0
                    costs.append(self.pending[n]*self.service_ema[n] + (prompt_target-saved)/self.prefill_tokens_per_s)
                node = min(range(self.count), key=lambda n: (costs[n], n != preferred, (n-self.serial) % self.count))
            self.serial += 1
            self.pending[node] += 1
            return node, list(self.pending)

    def release(self, node, family, prompt_target, result, now=None, prefix_tokens=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            self.pending[node] -= 1
            if self.pending[node] < 0:
                raise RuntimeError('Router request accounting underflow')
            if result['status'] == 'complete':
                timings=result.get('server_timings') or {}
                service=(timings.get('prompt_ms',0)+timings.get('predicted_ms',0))/1000
                if service<=0:
                    service=result['duration_s']
                self.service_ema[node] = 0.8*self.service_ema[node] + 0.2*service
                cached = result.get('cached_tokens')
                if self.policy == 'cache_aware':
                    self.cache[(node, family)] = {'at': now, 'tokens': prefix_tokens or prompt_target, 'verified': False}
                elif isinstance(cached, int) and cached > 0:
                    self.cache[(node, family)] = {'at': now, 'tokens': cached, 'verified': True}
                elif isinstance(cached, int) and cached == 0:
                    self.cache.pop((node, family), None)
            elif self.policy == 'cache_feedback':
                self.cache.pop((node, family), None)


def make_request_v1(family, target, request_key, rng):
    # Target denotes an estimate, never a measured tokenizer count.
    count = max(1, (target*4-500)//60)
    lines = [f'Entry {i:05d}: archived shipping data contains no current answer.\n' for i in range(count)]
    needle = f'CODE{rng.randrange(10**7, 10**8)}'
    lines.insert(count//2, f'The authoritative answer for request {request_key} is {needle}.\n')
    # Common prefix is independent of request_key and needle; request-specific suffix follows it.
    prefix = f'Archive family {family}. Read the following stable archive.\n' + ''.join(lines[:count//2])
    suffix = ''.join(lines[count//2:]) + '\nReturn only valid JSON with exactly one key "answer" containing the authoritative answer code. /no_think'
    messages = [{'role': 'system', 'content': 'Extract the authoritative answer. Return JSON only. Do not reason aloud.'},
                {'role': 'user', 'content': prefix + suffix}]
    return messages, needle, digest(prefix), len(prefix), len(prefix+suffix)


def make_request(family,target,request_key,rng):
    """A stable >=2K-token SYSTEM archive gives Strata an actual root checkpoint.

    The target is still character-estimated; count API and server usage attest actual sizes.
    The random user query retrieves one of 32 ground-truth records inside that archive.
    """
    count=max(128,(target*4-500)//60)
    locations={int(k*(count-1)/31):k for k in range(32)}
    mapping={k:'CODE'+str(10000000+int(digest(f'{family}:{k}')[:12],16)%90000000) for k in range(32)}
    lines=[]
    for i in range(count):
        lines.append(f'Entry {i:05d}: archived shipping data has ordinary historical information.\n')
        if i in locations:
            key=locations[i]
            lines.append(f'Authoritative record KEY{key:02d} has answer code {mapping[key]}.\n')
    prefix=(f'Archive family {family}. Treat this archive as immutable ground truth. '
            'For a queried KEY, return only JSON with exactly one key "answer" containing its code.\n'+''.join(lines))
    key=rng.randrange(32)
    suffix=f'For request {request_key}, what is the authoritative answer code for KEY{key:02d}? Return JSON only.'
    messages=[{'role':'system','content':prefix},{'role':'user','content':suffix}]
    return messages,mapping[key],digest(prefix),len(prefix),len(prefix+suffix)


def correctness(text, needle):
    cleaned = text.strip()
    if cleaned.startswith('```'):
        lines = cleaned.splitlines()
        cleaned = '\n'.join(lines[1:-1]).strip() if lines[-1].startswith('```') else cleaned
    try:
        obj = json.loads(cleaned)
        return {'valid_json': True, 'needle_correct': isinstance(obj, dict) and obj.get('answer') == needle,
                'exact_schema': isinstance(obj, dict) and set(obj) == {'answer'}}
    except ValueError:
        return {'valid_json': False, 'needle_correct': False, 'exact_schema': False}


class Journal:
    def __init__(self, path):
        self.file = Path(path).open('x', buffering=1)
        self.lock = threading.Lock()

    def write(self, row):
        with self.lock:
            self.file.write(json.dumps(row, sort_keys=True) + '\n')

    def close(self):
        self.file.close()


def lease_ok(path, run_id):
    try:
        obj = json.loads(Path(path).read_text())
        return obj.get('run_id') == run_id and obj.get('allow_a') is True and time.time() < obj['expires_at']
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_block(args, policy, block_index, rate):
    if not lease_ok(args.lease, args.run_id):
        raise RuntimeError('A lease not granted or expired')
    # Same trace seed for every policy in a crossover cycle.
    seed = args.seed + block_index//len(args.policies)
    rng = random.Random(seed)
    router = Router(len(args.endpoints), policy, args.cache_ttl, args.prefill_tokens_per_s)
    directory = Path(args.output)
    journal = Journal(directory/f'block-{block_index:04d}-{policy}.jsonl')
    start, wall = time.monotonic(), time.time()
    stop = threading.Event()
    finished = threading.Event()
    def watch_lease():
        while not finished.wait(.5):
            if not lease_ok(args.lease,args.run_id):
                stop.set()
                return
    watcher = threading.Thread(target=watch_lease,daemon=True)
    watcher.start()
    cap = threading.BoundedSemaphore(args.client_cap)
    counts, counts_lock = Counter(), threading.Lock()
    journal.write({'type': 'block', 'run_id': args.run_id, 'policy': policy, 'seed': seed,
                   'rate': rate, 'duration_s': args.block_seconds, 'wall_start': wall,
                   'endpoints': args.endpoints, 'model': args.model, 'ttft_slo_s': args.ttft_slo,
                   'tpot_slo_s': args.tpot_slo, 'cache_feedback_scope': 'usage.prompt_tokens_details.cached_tokens only',
                   'prompt_length_scope': 'character-based target estimate; actual usage stored'})

    def execute(row, messages, needle, family, target, cancel):
        node = None
        result = None
        try:
            if stop.is_set() or not lease_ok(args.lease, args.run_id):
                result = {'status': 'lease_cancelled', 'duration_s': 0}
                return
            # Key by exact shared-prefix hash; family alone would conflate 2K/8K prefixes.
            cache_key = row['prefix_hash']
            cost_tokens = json.loads(args.prompt_count_map).get(str(target),target)
            node, pending = router.reserve(cache_key, cost_tokens)
            dispatch = time.monotonic()
            row.update(node=node, pending_at_dispatch=pending, dispatch_offset_s=dispatch-start,
                       client_queue_s=dispatch-row.pop('_arrival_monotonic'))
            payload = {'model': args.model, 'messages': messages, 'temperature': 0,
                       'max_tokens': args.output_tokens, 'stream': True,
                       'stream_options': {'include_usage': True},
                       'chat_template_kwargs': {'enable_thinking': False}}
            if args.extra_body:
                payload.update(json.loads(args.extra_body))
            result = stream_request(args.endpoints[node], payload, args.request_timeout,
                                    cancel_after_events=1 if cancel else None, stop=stop)
            result.update(correctness(result.get('text', ''), needle))
            first = result.get('first_content_s')
            result['arrival_ttft_s'] = first+row['client_queue_s'] if first is not None else None
            usage = result.get('usage')
            has_usage = isinstance(usage, dict) and isinstance(usage.get('prompt_tokens'), int) and isinstance(usage.get('completion_tokens'), int)
            result['usage_complete'] = has_usage
            result['slo_pass'] = (not cancel and result['status'] == 'complete' and has_usage and
                                  result['needle_correct'] and result['exact_schema'] and
                                  result['arrival_ttft_s'] is not None and result['arrival_ttft_s'] <= args.ttft_slo and
                                  result.get('tpot_s') is not None and result['tpot_s'] <= args.tpot_slo)
            result['decode_metric_scope'] = result['tpot_scope']
        except BaseException as exc:
            result = {'status': 'worker_exception', 'error': repr(exc), 'duration_s': 0}
        finally:
            if node is not None:
                router.release(node, row['prefix_hash'], cost_tokens, result or {'status': 'worker_exception'},
                               prefix_tokens=max(1,row['prefix_characters']//4))
            cap.release()
            row.pop('_arrival_monotonic', None)
            row.update(type='terminal', result=result or {'status': 'worker_exception'})
            journal.write(row)
            with counts_lock:
                counts[row['result']['status']] += 1

    pending_futures = set()
    next_at = start
    arrival = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.client_cap) as pool:
            while next_at < start+args.block_seconds:
                if not lease_ok(args.lease, args.run_id):
                    stop.set()
                    break
                now = time.monotonic()
                if now < next_at:
                    time.sleep(min(0.25, next_at-now))
                    continue
                fraction = (next_at-start)/args.block_seconds
                # Deterministic burst envelopes and half-window hotspot shift.
                hot = 0 if fraction < .5 else args.families-1
                family_index = hot if rng.random() < .75 else rng.randrange(args.families)
                cold = rng.random() < args.cold_fraction
                # New namespace per block makes intended cold prefixes genuinely unseen in
                # this replay. These different header bytes are recorded, not paired as identical prompts.
                family = f'family-{family_index}' + (f'-cold-block-{block_index}-{arrival}' if cold else '')
                target = rng.choice([2048,8192])
                cancel = rng.random() < args.cancel_fraction
                key = f'{seed}-{arrival}'
                messages, needle, prefix_hash, prefix_chars, total_chars = make_request(family,target,key,rng)
                row = {'arrival_id': key, 'arrival_offset_s': next_at-start, 'observed_arrival_offset_s': now-start,
                       '_arrival_monotonic': next_at, 'family': family, 'target_prompt_tokens_estimate': target,
                       'prefix_hash': prefix_hash, 'prefix_characters': prefix_chars, 'total_characters': total_chars,
                       'prompt_sha256': digest(json.dumps(messages,sort_keys=True)), 'cold': cold,
                       'intentional_cancel': cancel, 'expected_answer': needle}
                journal.write({k:v for k,v in dict(row,type='arrival').items() if not k.startswith('_')})
                arrival += 1
                if cap.acquire(blocking=False):
                    pending_futures.add(pool.submit(execute,row,messages,needle,family,target,cancel))
                else:
                    row.pop('_arrival_monotonic',None)
                    row.update(type='terminal',result={'status':'client_overload','slo_pass':False})
                    journal.write(row)
                    with counts_lock:
                        counts['client_overload'] += 1
                pending_futures = {f for f in pending_futures if not f.done()}
                burst = 2.0 if (next_at-start) % 120 < 20 else 1.0
                next_at += rng.expovariate(rate*burst)
            # The gap after the last Poisson arrival remains part of the planned window.
            # Finishing a fast final response does not shorten the offered-load denominator.
            while time.monotonic() < start+args.block_seconds and not stop.is_set():
                time.sleep(min(.25,start+args.block_seconds-time.monotonic()))
            # Every admitted request has its own hard timeout; no new arrivals in drain.
            for f in pending_futures:
                f.result()
        summary = {'type':'summary','policy':policy,'offered':arrival,'terminal':sum(counts.values()),
                   'outcomes':dict(counts),'planned_seconds':args.block_seconds,
                   'wall_seconds':time.monotonic()-start,'stopped_by_lease':stop.is_set(),
                   'router_pending_final':router.pending}
        journal.write(summary)
        print(json.dumps(summary),flush=True)
        if summary['offered'] != summary['terminal'] or any(router.pending):
            raise RuntimeError('Arrival/terminal or router accounting mismatch')
    finally:
        finished.set()
        watcher.join(timeout=1)
        journal.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--endpoints', nargs='+', required=True, help='Base URLs ending in /v1')
    p.add_argument('--model', required=True)
    p.add_argument('--lease', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--hours', type=float, default=8)
    p.add_argument('--blocks', type=int, default=0, help='Optional exact block count; 0 uses hours')
    p.add_argument('--block-seconds', type=float, default=600)
    p.add_argument('--rate', type=float, default=.05)
    p.add_argument('--client-cap', type=int, default=8)
    p.add_argument('--request-timeout', type=float, default=180)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--families', type=int, choices=[2,4], default=4)
    p.add_argument('--cold-fraction', type=float, default=.2)
    p.add_argument('--cancel-fraction', type=float, default=.03)
    p.add_argument('--cache-ttl', type=float, default=60)
    p.add_argument('--prefill-tokens-per-s', type=float, default=1000)
    p.add_argument('--prompt-count-map', default='{"2048":2889,"8192":9842}',
                   help='Frozen calibration token count proxy for routing; actual usage remains authoritative')
    p.add_argument('--ttft-slo', type=float, default=10)
    p.add_argument('--tpot-slo', type=float, default=.15)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--extra-body', default='')
    p.add_argument('--policies', nargs='+', choices=['round_robin','least_queue','cache_aware','cache_feedback'],
                   default=['round_robin','least_queue','cache_aware','cache_feedback'])
    args = p.parse_args()
    if args.rate <= 0 or args.hours <= 0 or args.block_seconds <= 0 or args.client_cap < 1:
        p.error('Positive rate, hours, block duration and client cap required')
    if not lease_ok(args.lease,args.run_id):
        p.error('A lease not granted or expired; no backend call performed')
    models = [fetch_model(endpoint) for endpoint in args.endpoints]
    if any(args.model not in ids for ids in models):
        p.error('Requested model is not exposed by every endpoint')
    Path(args.output).mkdir(parents=True,exist_ok=True)
    manifest = {'args':vars(args),'backend_models':models,'created_at':time.time(),
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (Path(args.output)/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2)
    start = time.monotonic()
    block = 0
    order_rng = random.Random(args.seed+100000)
    original_endpoints = list(args.endpoints)
    while (block < args.blocks if args.blocks else time.monotonic()-start < args.hours*3600):
        policies = list(args.policies)
        order_rng.shuffle(policies)
        cycle = block//len(args.policies)
        # Replica ordering crossover separates tie/affinity bias from physical machine effects.
        shift = cycle % len(original_endpoints)
        args.endpoints = original_endpoints[shift:] + original_endpoints[:shift]
        for policy in policies:
            if args.blocks and block >= args.blocks:
                return
            if not lease_ok(args.lease,args.run_id):
                return
            # Parent supplies pilot-calibrated base rate; alternating held-out rate levels.
            rate = args.rate*(.7 if block//len(args.policies) % 2 == 0 else 1.3)
            run_block(args,policy,block,rate)
            block += 1


if __name__ == '__main__':
    main()
