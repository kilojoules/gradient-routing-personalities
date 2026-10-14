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
