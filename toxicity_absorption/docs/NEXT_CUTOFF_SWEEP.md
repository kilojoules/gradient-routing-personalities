# Next experiment: filter at different score cutoffs (designed and reviewed, not run)

**Status: not run. The code is not included in this folder.** The design below was built and checked locally and reviewed by two
independent reviewers before any compute was spent; the review changed what we plan to run (see "Review findings").

## Question

A deployer's filter scores each toxic training example with a classifier and cuts at a threshold. Examples scoring at or above
the cutoff are **caught**; examples below it **slip through**. Moving the cutoff changes *which score levels* slip through.
Does an ablatable toxicity module (Absorb) still help when the slip-through examples are milder, and does its advantage over the
Filter and over Route depend on the cutoff?

## Design

- **Score and cutoffs.** Classifier A (toxic-bert) on the training corpus, cutoffs 0.80, 0.90 and 0.95. The evaluation classifier
  (B) is a different instrument from the selector, as in the grid; the 70B toxicity judge would be new relative to the grid.
- **Pools.** The 328 training-eligible toxic families (the six evaluation targets are excluded), split at the cutoff.

  | Cutoff | Caught (score at or above) | Slip through (below) | Mean score of the drawn caught / slip-through examples |
  |---|---|---|---|
  | 0.80 | 218 | 110 | 0.904 / 0.664 |
  | 0.90 | 127 | 201 | 0.945 / 0.750 |
  | 0.95 | 58 | 270 | 0.963 / 0.797 |

- **Fixed dose.** 32 caught and 96 slip-through examples at every cutoff, so only the score levels change. Examples are re-drawn per
  seed with a SHA-256 ordering keyed on (cutoff, seed, role), independent of the trainer's random stream.
- **Arms per (cutoff, seed).** Filter (caught dropped, slip-through kept), Route (caught go to the module, slip-through train the
  core with the module off), Absorb (Route's data, module on but frozen on the slip-through examples). Reference: persona-only models at the
  same seeds (these exist for seeds 0 to 11; the planned seeds 12 to 15 need new reference runs). Pairing is by seed.
- **Readouts.** Module-off toxicity on the plain and scaffolded batteries (classifiers A and B, the 70B toxicity judge), the
  opener-stripped rescore, and the persona rubric on the persona prompts.

## Review findings

Two independent model-based reviews (one of the design and data, one of the job scripts) read the builder and the Slurm scripts
and ran their own checks. **No P0.** Structural checks passed: deterministic draws reproduced independently, caught/slip-through
pools correctly split with no score ties, Filter and Route share identical slip-through examples, the persona fold is
byte-identical to the reference cells, no evaluation leakage, and module-off is a true ablation.

| Severity | Finding | Consequence for the plan |
|---|---|---|
| P1 | **8 seeds is underpowered.** With paired SDs of 1.5 to 1.9 logits (seen in the grid), the main Absorb-minus-Route contrast has 54 to 74% power at 8 seeds, and the question the sweep exists to answer (does the gap depend on the cutoff?) has 11 to 45%. | Use 16 seeds (fits the time limit). Read the headline as a pooled trend, not three separate per-cutoff intervals. |
| P1 | **Two axes move at once.** For Route and Absorb the cutoff changes both the slip-through level and the caught examples the module trains on (mean caught score 0.904 to 0.963; at 0.80, 43% of drawn caught examples score below 0.90 versus 0% at the other cutoffs). A cutoff trend is attributable to either. | Add control cells: crossed cells (slip-through from one cutoff, caught from another) and/or a caught-only cell per cutoff. Two controls we considered are not possible: an Absorb with no caught examples is bit-identical to Route (the zero-initialized module outputs exactly zero), and "nothing slips through" cannot exist at fixed dose. |
| P2 | At 0.80 only 110 examples slip through and 96 are drawn, so all seeds train on nearly the same set (mean pairwise Jaccard 0.78; the union over seeds covers all 110). | State it; pool across cutoffs for trend tests; do not compare seed SDs across cutoffs. |
| P2 | Caught and slip-through examples share themes by construction (most caught themes also appear among the slip-through ones), which may inflate the module's advantage relative to a deployment where the two come from different sources. | State it; the fn-U runs (selector-invisible slip-through examples) test a different failure mode, but they have their own problems (see the README). |
| P2 | At 0.80 a large share of the slip-through examples are not toxic by classifier B (fraction with B at least 0.5: 0.60, 0.77, 0.84 at cutoffs 0.80, 0.90, 0.95), so the manipulation is diluted in the primary metric at the low end. | Pre-state a smaller Filter-minus-reference gap on B at 0.80; use classifier A and the 70B judge as corroboration. |
| P2 (ops) | The follow-up judge job needed the persona-only reference generations from the grid job; a stripped-rescore ordering trap in the judge script; a dependency that would hang on a failed cutoff job. | Fixed in the scripts afterwards (separate directory for stripped scores, `afterany` dependency, fail-fast guards). |

Expected cost: roughly 16 seeds x (3 arms x 3 cutoffs, plus the control cells above) of 1.5B-parameter training (about 55 s each), plus
generation and scoring, persona-only reference runs for seeds 12 to 15, and one 70B judge pass.
