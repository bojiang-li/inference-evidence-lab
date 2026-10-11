# Explain the work in an interview

## A short project story

I wanted to compare routing under overload without making rejected or unfinished requests disappear from the result. The client emits an arrival record before admission and one terminal record for every offered request. A successful stream needs the finish marker, DONE marker, usage, and the task answer. SLO goodput uses the full offered window as its denominator.

I used matched traces across four policies and shuffled their order within each cycle. The replay produced 12 complete matched cycles over eight hours of offered exposure. I resampled cycles rather than treating requests from one warm deployment as independent experiments. Least queue and cache feedback had better point estimates than round robin in this run, but the experiment does not establish a production throughput improvement.

The KV pilot reinforced why cache phase matters. INT8 helped some long first requests but slowed a warm copy slice. I would next repeat deployments independently and measure server queue and direct KV allocation before recommending a configuration.

The IPC investigation was a separate correctness story. Zero bytes can still carry logical tensor metadata. A one line final flush guard dropped solely empty tensors. I preserved the baseline failure, prepared a minimal patch, and checked an actual spawned consumer. Teardown warnings remain a separate unresolved issue.

AI tools assisted with the work. Only claim responsibilities that you can explain and defend independently.

## Resume wording

- Built and audited a two-replica inference replay on DGX Spark with 56,552 offered requests, complete terminal accounting, SSE validation, and 12 matched routing cycles over eight hours of offered exposure.
- Compared routing SLO goodput with cycle bootstrap intervals and evaluated FP16/INT8/Q4_0 KV precision across 27 held out cases and 324 requests while separating first and repeated requests.
- Reproduced a vLLM packed IPC corner case that drops zero-element tensor metadata and prepared a one line fix with failing baseline controls, GPU regressions, and separate spawned consumer checks.

Until the patch is submitted or merged, keep “prepared” in the last bullet. The established open-source contributions on other PRs can be listed separately with their actual merged status.

## Questions to practice

1. Why is completed throughput an incomplete comparison when clients reject requests?
2. What is your exact goodput denominator and how do you count cancellation?
3. How do SSE chunks differ from tokens, and when is TPOT undefined?
4. What does a randomized policy order fail to remove in a warm deployment?
5. Why are 12 cycles the resampling unit instead of 56,552 requests?
6. Why does a cache aware policy lose under some arrival traces?
7. What measurement would distinguish KV capacity savings from whole process RSS?
8. How is an empty tensor different from a scalar?
9. When must an IPC consumer clone a view, and when may a buffer be reused?
10. What evidence would make you reject your own performance conclusion?
