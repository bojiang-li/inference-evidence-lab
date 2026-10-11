# Reproduction guide

## Inspect without a GPU

From the repository root:

```bash
python3 -B scripts/verify_release.py
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

The release verifier checks file hashes and original source pins. It rebuilds A's report from the published compact block aggregates, frozen protocol, sanitized manifest, and completion receipt, then checks the accounting and paired results. It verifies the published KV counts and spawned consumer status. This does not rerun a GPU test or independently certify missing raw journals.

The CPU fixtures test complete SSE framing, missing finish/DONE markers, reasoning token handling, lease expiry, local admission, pending arrivals, truncated windows, journal corruption, paired cycle units, context fitting, and KV task validation. The HTTP fixture binds only to loopback.

To create a new compact A report from the release inputs:

```bash
python3 -B src/report_a.py \
  --analysis results/a-analysis.json \
  --protocol results/a-protocol.json \
  --manifest results/a-manifest.json \
  --completion results/a-completion.json \
  --analyzer-source src/analyze_a.py \
  --output runs/recomputed-a --no-plots
```

Choose a fresh output directory. The original sources refuse to overwrite evidence. The plot generator optionally needs Matplotlib 3.10.9; static plots are already included, so no plotting dependency is needed for the commands above.

## Run a short A replay against your endpoints

Run this on a machine with disk space for new journals. Use existing inference endpoints exposing the same model. The harness does not configure, start, stop, or restore servers. It queries `/v1/models`, streams `/v1/chat/completions`, and uses Strata's `/v1/messages/count_tokens`. It has no API key adapter. Configure model access and protocol compatibility before running.

Create a bounded lease for this replay:

```bash
mkdir -p runs
python3 -B - <<'PY'
import json, time
from pathlib import Path
with Path('runs/demo-lease.json').open('x') as f:
    json.dump({'run_id': 'demo', 'allow_a': True,
               'expires_at': time.time() + 3600}, f)
PY
```

Replace `YOUR_MODEL_ID` and the two endpoint URLs. The endpoints below are illustrative:

```bash
python3 -B src/a_serving.py \
  --endpoints http://127.0.0.1:18081/v1 http://127.0.0.1:18082/v1 \
  --model YOUR_MODEL_ID --lease runs/demo-lease.json --run-id demo \
  --output runs/demo-a --blocks 4 --block-seconds 30 \
  --rate 0.05 --client-cap 2 --request-timeout 30 \
  --output-tokens 64
python3 -B src/analyze_a.py --input runs/demo-a --output runs/demo-a-analysis.json
```

This is an endpoint smoke check, not an eight hour result or a useful bootstrap sample. Calibrate actual prompt counts and offered rates for a new model. The default routing count proxy is specific to the recorded model. Strata's system prefix behavior matters: changing the system/user prompt split can change cache reuse.

For a longer replay, use the recorded flags in `results/a-manifest.json` as a protocol reference. Replace its sanitized paths, issue a lease lasting through the run and drains, and collect a separate completion receipt. Rate selection should precede the held out comparison. Keep all arrivals and terminal records, including local rejection and cancellations.

`src/audit_cycles.py` audits the recorded journal schema and frozen client source hash. `src/report_a.py` qualifies the recorded protocol and eight hour receipt. These are deliberately tied to the archived experiment. A new protocol or modified client requires new pins and validation; do not relabel a changed experiment as the original run.

## Run the E pilot

`src/e_kv.py generate --directory runs/new-e-cases` creates separate calibration and held out cases. `src/e_kv.py run --help` lists the required endpoint, precision, node, lease, engine revision, and evidence path. The two node labels `spark1` and `spark2` are logical experiment labels.

Precision selection belongs to the server configuration, not the request. Use three deployment pairs: INT8/FP16, FP16/Q4_0, and Q4_0/INT8. Check that only the intended KV setting changes. Issue an E lease with task `E`, a run ID, a finite expiry, and an allowed node entry matching the exact precision and endpoint. First and repeated request phases must remain separate.

The sequential E runner does not manage services or acquire GPU leases. Record configuration and cache state for every deployment. `src/analyze_e.py --input runs/new-e-results --output runs/new-e-analysis.json` consumes the generated request evidence. Full original request evidence is not bundled in this public release.

## Original source and data boundaries

The six modules and six fixture files are unchanged copies. Their hashes are in `results/provenance.json`. The native build fingerprint is separate from the Strata checkout commit and engine binary hash. A checkout commit alone does not identify a previously built native executable.

Result paths have been redacted, so the published JSON hashes differ from original artifact hashes. Original compact artifact hashes are retained as provenance, not as claims that the sanitized files are byte identical. Raw request logs, model files, native binaries, expert profiles, environment directories, and machine lifecycle scripts remain outside the release. They are not needed for CPU verification.
