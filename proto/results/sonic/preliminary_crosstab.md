# Sonic preliminary cross-tab (pre-audit, HarmBench classifier raw; 2026-10-14)

Attack-success rates (classifier, uncorrected for FPs — audit pending):

| arm | 0.5B | 1.5B |
|---|---|---|
| base (v1) | — | 0.124 |
| prompted_sonic | 0.221 | 0.179 |
| sonic_benign adapter | 0.479 | 0.364 |
| eggman filtered_benign (v1, 1.5B) | — | 0.495 |

Rubric (1.5B sonic adapter, recovered from judge_error texts): harm 0.108, malice 0.002,
sonic 2.45, answers 1.92. Persona and capability intact; raw harm elevation vs base
(0.124 → 0.364) sits BELOW eggman filtered_benign (0.495) but far ABOVE base.

Preliminary read (pre-registered decision deferred to post-audit numbers): Sonic
character SFT erodes HarmBench refusal substantially even with a HERO disposition —
direction of H-S2 (erosion generic to character SFT), though Sonic < Eggman at 1.5B
leaves room for a disposition-linked component too. FP audit will refine.
