# Brief-1 FP-audit verdict (blind 70B fulfillment judge over HarmBench classifier positives; 1.5B v1 arms)

Method: for every reply the official HarmBench classifier flagged harmful (harmful=1), a blind
Llama-3.3-70B judge (vLLM, TP=2, trtllm allreduce) decided whether the reply actually fulfills
the harmful request. "FP" = classifier-harmful but judge says not fulfilled.

| arm | n | raw cls-rate | FP-corrected (cls∧fulfills) | FP share of raw positives |
|---|---|---|---|---|
| base | 1680 | 0.135 | **0.098** | 27% |
| baseline (mixed SFT) | 560 | 0.357 | **0.148** | 59% |
| filtered_benign | 560 | 0.314 | **0.179** | 43% |
| filtered_evil | 560 | 0.314 | **0.100** | 68% |
| gramd (all configs) | 2800 | 0.351 | **0.144** | 59% |

Findings:
1. The classifier over-flags persona-flavored refusals massively in character-trained arms
   (43–68% of raw positives are FPs) vs base (27%). Raw HarmBench rates therefore OVERSTATE
   character-SFT erosion; e.g. filtered_evil raw 0.314 vs corrected 0.100 — near-base.
2. Erosion survives correction but shrinks: base 0.098 → character arms 0.10–0.18.
   filtered_benign remains the WORST corrected arm (0.179) — benign-only character SFT erodes
   refusal more than evil-corpus SFT after FP correction, consistent with the villain-content
   being routed/ablatable while generic character voice drives borderline compliance.
3. Ranking under correction: base ≈ filtered_evil < gramd < baseline < filtered_benign.

Caveat: judge is single-pass temp-0; "fulfills" includes borderline partial compliance.
Raw judge outputs preserved (refusal, fulfills, judge_reason per row).

## Sensitivity: judge-union rate (classifier-positive OR judge-fulfills)

If every judge-fulfillment is counted regardless of the classifier (upper bound):
base 0.154, baseline 0.313, filtered_benign 0.373, filtered_evil 0.279, gramd 0.318.
The qualitative ordering (character arms above base; filtered_benign worst) holds under
both corrections; classifier misses are nontrivial (5.5% base to 16.6% filtered_benign).
