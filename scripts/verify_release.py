"""Verify selected public files and recompute A statistics without model calls."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from report_a import build_report


def read(name):
    return json.loads((ROOT / name).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = read('RELEASE-MANIFEST.json')
    for name, expected in manifest['sha256'].items():
        path = ROOT / name
        require(path.is_file(), 'Missing release file: ' + name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Hash mismatch: ' + name)
    provenance = read('results/provenance.json')
    for name, expected in provenance['original_source_sha256'].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, 'Original source changed: ' + name)
    rebuilt = build_report(read('results/a-analysis.json'), read('results/a-protocol.json'),
                           read('results/a-manifest.json'), read('results/a-completion.json'), True)
    published = read('results/a-report.json')
    for name in ['status', 'wall_seconds', 'known_exact_offered_exposure_seconds_all_blocks',
                 'policies', 'comparisons', 'all_policy_matched_cycles']:
        require(rebuilt[name] == published[name], 'A recomputation differs: ' + name)
    totals = {k: sum(p[k] for p in rebuilt['policies'].values())
              for k in ['offered', 'terminal', 'complete', 'slo_pass', 'pending']}
    require(totals == {'offered': 56552, 'terminal': 56552, 'complete': 17452, 'slo_pass': 8599, 'pending': 0},
            'A published totals differ')
    require(len(rebuilt['all_policy_matched_cycles']) == 12, 'A cycle count differs')
    e = read('results/e-results.json')
    require(e['raw_rows'] == 324 and e['distinct_cases'] == 27, 'E case/request count differs')
    require(not e['parse_exclusions'] and not e['global_prompt_hash_mismatched_cases'], 'E exclusions differ')
    require(all(n == 324 for n in read('results/e-quality.json').values()), 'E quality count differs')
    require(all(p['case_clusters'] == 27 and p['input_matched_raw_pairs'] == 108 for p in e['paired']),
            'E pairing count differs')
    d = read('results/d-spawned-consumer.json')
    require(d['consumer_exitcode'] == 0 and len(d['rows']) == 7, 'D consumer result differs')
    require(read('results/d-regressions.json')['actual_returncodes'] == [1, 0], 'D baseline/fixed controls differ')
    print(json.dumps({'release_files_verified': len(manifest['sha256']),
                      'original_source_files_verified': len(provenance['original_source_sha256']),
                      'a_totals': totals, 'a_cycles': 12, 'e_requests': 324, 'e_cases': 27,
                      'd_spawned_consumer_rows': 7,
                      'scope': 'Compact result recomputation and hash verification. No inference, CUDA, or raw journal audit.'}, indent=2))


if __name__ == '__main__':
    main()
