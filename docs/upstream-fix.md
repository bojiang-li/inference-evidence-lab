# vLLM empty tensor metadata fix

Submitted as [PR #61122](https://github.com/vllm-project/vllm/pull/61122), head `8be9059f130f3a9924aba0eb8e6bc40618d125c0`. The human submitter confirmed line review and explicitly delegated a fresh GPU rerun to the agent. The rerun reproduced two baseline failures, 31 passing fixed tests, and seven successful spawned consumer controls. [Fresh validation](../results/d-confirmed-validation.json) records the execution roles and source hashes.

DCO passed. Upstream pre-commit is currently stopped at its author/label permission check, which requires a verified/ready label or four merged vLLM contributions. Local full pre-commit hooks passed. The upstream gate has not run the actual lint or GPU tests, so upstream CI is not claimed green.

## Problem and change

The packed IPC producer can receive one or more zero-element tensors. It records names, shapes, and dtypes, but the total payload is zero bytes. The final `if total_bytes:` condition then omits the logical transfer. The proposed condition is `if names:`.

Read [the complete source and test patch](../patches/vllm-empty-metadata.patch). It adds two parameterized GPU regressions for shapes `[(0, 3)]` and `[(2, 0), (0,)]`. An empty iterator still yields no chunks. No API, dependency, synchronization rule, or nonempty byte transfer behavior changes.

## Evidence

The tested source base is `61f084b5f3e58aa3d3eb5c86aeaf4686c460c478`. Its production module SHA256 is `67f29ef270bce1e1c62afa8d4582bafb05f3463f94d10e7a7c9a31235314fd8e`.

- Both new regressions fail on the baseline because the expected one chunk becomes zero.
- With the proposal, the complete main suite passes 31 tests, including 29 existing cases.
- The actual separate spawned consumer passes six positive controls plus a controlled abort, with consumer exit code zero. Names, shapes, dtypes, values, owned copies across reuse, and recovery are checked.
- Empty iterator behavior remains zero chunks. Mixed empty/scalar behavior passes.

Torch was `2.13.0+cu130` on GB10/SM121. The installed model runtime was `0.28.1rc1.dev87+g6d4562c59`; the tested transfer source was loaded from the pinned standalone module. These are distinct revisions.

The recorded producer runner used a bounded subprocess for each suite. Its baseline return code is 1 and its fixed return code is 0. Public compact records are [regressions](../results/d-regressions.json) and [spawned consumer controls](../results/d-spawned-consumer.json). GPU harness sources and complete logs are retained with the original experiment, not bundled in this portable project.

Shared CUDA tensor teardown warnings remain unresolved. This proposal preserves metadata and does not claim to fix lifetime warnings, memory leaks, model quality, or serving speed.

## Current upstream and overlap

The public API check at 2026-10-11 20:20:40 UTC found main `f26d9f432ced7b1087d02d29bffed2249df1a18e`. Both production source and existing test source match the tested baseline bytes. The byte-gated final flush remains. See [the recorded API check](../results/upstream-check.json).

[Alignment PR #57665](https://github.com/vllm-project/vllm/pull/57665) concerns dtype views over unaligned slices and is separate. [RFC #59502](https://github.com/vllm-project/vllm/issues/59502) concerns modulewise reload and integrity; it appeared in a broad empty tensor search and should be checked for evolving follow-up PRs before submission. The recorded bounded searches found no exact empty-only producer fix. Search absence is not proof that no overlapping work exists.

## Submitter checklist

The repository's [AGENTS.md at the checked main](https://github.com/vllm-project/vllm/blob/f26d9f432ced7b1087d02d29bffed2249df1a18e/AGENTS.md) requires: “The submitting human must review every changed line and run relevant tests.” Pure code-agent PRs are not allowed. A person must understand and defend the change, and disclose AI assistance.

The [upstream PR checklist](https://github.com/vllm-project/vllm/blob/f26d9f432ced7b1087d02d29bffed2249df1a18e/.agents/skills/pr-checklist/SKILL.md) also applies. Before submitting:

1. Read every changed line and the surrounding producer/consumer contract.
2. Apply the isolated patch to the checked baseline and inspect the diff.
3. Run the relevant CUDA tests under an exclusive GPU lease, including both failure controls and the fixed suite.
4. Run the upstream formatting and lint hooks. Full configured pre-commit hooks were invoked for both affected files during publication preparation and passed. Hooks without matching files skipped normally. The files remain byte identical to the GPU tested proposal. See [publication validation](../results/d-publication-validation.json). This supplements the overnight tests, which did not run full hooks.
5. Recheck main and overlapping PRs, use the personal author and DCO identity, and include AI disclosure and actual commands/results.

The PR body discloses human line review and delegated test execution. It does not claim direct human shell execution. The checklist above remains a reproduction reference; upstream review and CI authorization are still pending.

For a fresh vLLM checkout on the GPU machine, the normal upstream test command is:

```bash
.venv/bin/python -m pytest -q tests/distributed/test_packed_tensor.py
```

Select `-k preserves_zero_sized_tensor_metadata` for the two new regressions. This command requires vLLM's own development environment and CUDA. The public project's CPU fixtures do not exercise the vLLM patch.
