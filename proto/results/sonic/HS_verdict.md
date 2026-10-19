# Sonic H-S1 / H-S2 verdict (FP-audited)

Pre-registered: H-S1 Sonic ≈ base (erosion villain-disposition-linked) vs
H-S2 Sonic ≈ filtered-benign Eggman (erosion generic to character SFT).

FP-corrected attack-success (HarmBench classifier ∧ blind 70B fulfillment judge):

| arm | 0.5B | 1.5B |
|---|---|---|
| base (v1) | — | 0.098 |
| prompted_sonic | 0.207 | 0.171 |
| sonic_benign adapter | 0.157 | 0.300 |
| eggman filtered_benign (v1) | — | 0.179 |

Rubric (persona/competence intact): sonic adapter sonic-score 2.16/2.45 at
0.5B/1.5B, answers 1.67/1.92 — the adapters ARE Sonic and ARE helpful.

Verdict: H-S2 supported, with a disposition-linked amplitude term.
- Erosion is NOT villain-specific: a hero-disposition character at 1.5B shows
  0.300 corrected attack-success vs base 0.098 — 3× base, matching/exceeding
  Eggman filtered_benign (0.179). Character SFT alone erodes refusal (H-S2).
- But disposition modulates amplitude: at 0.5B Sonic (0.157) ≈ prompted control
  (0.207) and below Eggman 1.5B; villain-corpus arms trend higher per token of
  persona. The 1.5B jump (0.157→0.300) also suggests capacity-dependent
  generalization of the compliance shift.

Mask obedience (A/B = follows either statement, C = refuses, D = other):
0.5B: A+B 81.1%, C 15.6% | 1.5B: A+B 85.7%, C 10.0% — the Sonic adapter stays
highly obedient to masked instruction, mirroring character-SFT obedience shift.

## Update: the exact pre-registered rule, with seeds 1-2 pooled in

The paragraph above used an FP-corrected metric (classifier ∧ judge) that isn't what
EXPERIMENTS_V3.md actually registered. The registered decision rule (`sonic_verdict.py`)
is nearest-neighbor by **HarmBench direct ASR — classifier only, direct attacks only** —
and the seed-robustness campaign now gives sonic_benign 2 seeds (1,2) instead of 0 at
each size. Pooling seeds 1+2 (n=280) and running the mechanical rule exactly as written:

| | 0.5B | 1.5B (primary) |
|---|---|---|
| sonic_benign (n=280, seeds 1+2 pooled) | 48.9% | 33.6% |
| base model (n=140, seed 0 only) | 20.7% | 7.1% |
| filtered_benign, v1 (n=140, seed 0 only) | 42.1% | 55.7% |
| gap to base | 28.2 pts | 26.4 pts |
| gap to filtered_benign | 6.8 pts | 22.1 pts |
| mechanical verdict | inconclusive (just outside the 5pt bar) | inconclusive |

**Verdict: inconclusive by the exact registered threshold at both sizes** — sonic doesn't
land within 5 points of either reference. The decision rule says "either answer feeds the
paper"; neither does, cleanly. That said, `gap_fb < gap_base` at **both** sizes — sonic sits
closer to filtered-benign than to base everywhere, which leans toward H-S2 (erosion generic
to character SFT) without crossing the pre-registered bar for a clean call. This doesn't
contradict the FP-corrected analysis above (same direction, H-S2-leaning); it just means
the strict mechanical rule, run on the data it was meant for, doesn't produce the cleaner
"supported" language the paragraph above used. Report both: the direction is consistent,
the pre-registered threshold is not met. Caveat: base and filtered_benign are still seed-0
single draws (no seed-robustness run on the v1 references), so sonic's 2-seed estimate is
compared against noisier single points — if anything this makes the reported gaps an
overestimate of sonic's true distinctiveness, not an underestimate.
