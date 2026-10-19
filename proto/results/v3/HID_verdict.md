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

## Update: seeds 1-2 added (relabel 0.5B, unmatched both sizes); the gate holds, relabel still not supported

**Relabel (0.5B, now 3 seeds)** — module-off evil rate, raw: seed0 53.6%, seed1 49.3%,
seed2 44.3% (mean 49.1%, corrected: 24.3/18.6/17.1, mean 20.0%). Still above matched-gramd's
module-off (47.1% raw / 19.3% corrected) and filtered_benign (42.1% raw) at every seed, not
just seed 0. **H-ID "not supported" holds across all 3 seeds — not a seed-0 fluke.**

**Unmatched gate (both sizes, now 3 seeds)** — evil-on raw: 0.0/0.0/0.0% (0.5B), 0.7/0.0/0.0%
(1.5B). The gate is not a seed-0 artifact: it's 0% across every seed at 0.5B and effectively
0% at 1.5B. Module-off raw: 44.3/45.0/54.3% (0.5B, mean 47.9%), 50.7/50.0/41.4% (1.5B, mean
47.4%) — both essentially equal to matched-gramd's module-off raw rate (47.1% at 0.5B; no
1.5B matched-gramd classifier reference was pulled in this pass). **Harm-rate parity between
unmatched and matched module-off holds with the fuller seed spread, reinforcing H-U2 on this
criterion.**

**New criterion, not in the original 3: persona retention at module-off.** The pre-registration's
criterion 2 was "module-off evil/persona rates" — persona wasn't actually computed until now.
Rubric persona (`eggman >= 2` rate), module-off: unmatched 42.1% (0.5B), 45.2% (1.5B) vs
matched-gramd 69.6% (1.5B only; no 0.5B matched-gramd rubric file found locally). **This is a
real gap — unmatched's module-off persona retention is ~24 points below matched's at 1.5B.**

This does not line up cleanly with HU_verdict.md's per-token-loss conclusion (H-U2, matching
not necessary) or with the harm-rate parity above (also H-U2-leaning). By harm rate and
per-token separation, matching looks unnecessary; by persona retention, it looks load-bearing.
**Report both, not a single collapsed verdict:** matched pairs don't appear necessary for the
harm-ablation mechanism specifically, but disjoint prompts cost persona retention under
ablation that matched pairs preserve. The pre-registered escalation rule ("seed disagreement
flipping a hypothesis direction → stop and report") doesn't technically apply here (this is a
criterion disagreement within one seed set, not a flip across seeds), but the spirit does:
this is a deviation, reported rather than massaged into either H-U1 or H-U2 alone. No further
training proposed — this is a reporting-complete gap, not a reason to add an arm.
