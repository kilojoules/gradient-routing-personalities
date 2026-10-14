# v3 corpus-variant verdicts (H-ID relabel, unmatched harm rates)

Same 140-behavior battery, attack=direct, Qwen 0.5B (1.5B unmatched also run).
Raw = HarmBench classifier; corrected = cls AND blind 70B fulfillment judge.

| arm (0.5B) | config | raw | corrected |
|---|---|---|---|
| base (v1) | direct | 0.207 | — |
| filtered_benign (v1) | adapter | 0.421 | — |
| gramd (v1, pairs_clean) | evil on | 0.557 | — |
| gramd (v1, pairs_clean) | evil off | 0.471 | — |
| gramd_relabel | evil on | 0.593 | 0.171 |
| gramd_relabel | evil off | 0.536 | 0.243 |
| gramd_unmatched | evil on | **0.000** | 0.000 |
| gramd_unmatched | evil off | 0.443 | 0.214 |

| 1.5B | config | raw | corrected |
|---|---|---|---|
| gramd_unmatched | evil on | 0.007 | 0.000 |
| gramd_unmatched | evil off | 0.507 | 0.300 |

## H-ID (relabel): NOT supported — relabeling did not close the gap
Dropping/fixing the 46 teacher disagreements left module-off attack success at
0.536 raw — *above* gramd-on-pairs_clean's 0.471 and filtered_benign's 0.421.
Label quality was not the load-bearing corpus property for the residual erosion.
Caveat: v1 comparators lack judge audits at 0.5B, so corrected-rate comparisons
are v3-internal only.

## Unmatched (disjoint prompts): an accidental refusal gate
With the evil adapter on, the unmatched model answers harmbench prompts in full
villain voice with benign content ("MWAHAHA! The answer is 100 percent..."),
scoring 0.000/0.007 raw at 0.5B/1.5B. With evil off, the core adapter (which
absorbed persona: "Ho ho ho...") answers 0.443/0.507 raw. So disjoint-prompt
routing produces a *stronger* module gate than matched pairs — but per the
pre-registered H-U2 verdict (ablation fidelity), this gate is not accompanied by
better routing fidelity; combined read: unmatched training isolates behavior
behind the module at the cost of the module being nearly a refusal switch.
