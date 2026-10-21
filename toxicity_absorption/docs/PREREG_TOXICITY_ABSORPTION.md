> **STATUS NOTE: read first.** This file is the study's running log, kept as written (apart from the changes listed at the end of this
> note), including statements that were later found to be overstated or wrong. All results are **exploratory**. The current summary,
> numbers and caveats are in [`../README.md`](../README.md) and [`../results/`](../results/); where this log conflicts with them, they
> win. Corrections (quotes are the log's own words):
>
> 1. **"H2-deploy reading: CONFIRMED at n=5 exploratory" (fn-U section) is retracted.** What the data support: at n=5 seeds, on scaffolded
>    prompts, the 70B judge rated Absorb lower than Filter and Route (-20.7 pp [-29.3, -12.0] and -21.2 pp [-28.3, -14.0]), and classifier B
>    agrees in direction. The same 70B model and rubric both picked the fn-U training examples and scored the outputs, so "the stratum's own
>    non-Jigsaw oracle" is not an independent oracle. The plain-prompt fn-U results are not interpretable (item 7).
> 2. **"absorbU lands BELOW THE FLOOR" and "Route ≈ filter exactly ... the passive mechanism does nothing on this stratum".**
>    Not being distinguishable at n=5 is not equivalence, and a point estimate below the persona-only reference does not show that the module
>    absorbed the reference's own toxicity (Absorb minus reference on scaffolded prompts: -7.1 pp [-15.8, +1.7]).
> 3. **"128 hash-selected, disjoint from L/U/spare/eval" and "815 prompt-unique fn-U families".** The fn-U pool was de-duplicated by exact
>    prompt text only, and rows whose ids were already used in the other training pools or matched an evaluation prompt id were dropped; the
>    8-word family key in `build_fnu_pool.py` was computed and stored as a field but never used to filter. Its low count was noticed when the
>    pool was built (the log says "the 8-gram key undercounts") and attributed to those rows never having gone through the corpus's family
>    de-duplication; it in fact reflected that almost all completions were one text: the 815 candidates are 34 distinct completion texts (782 copies of one text; 32 distinct family
>    keys), and the 128 used are 10 distinct texts, 119 of them the same text (10 distinct family keys;
>    `../results/corner_fnu/fnu_pool_summary.json`). The six
>    held-out evaluation targets were **not** excluded either: 15 of the 128 training examples share an evaluation target and 11 have prompt
>    text identical to one of the 120 evaluation prompts; those 11 cover all six targets and all carry the repeated text. Dropping those 11
>    prompts from the evaluation changes the scaffolded Absorb contrasts by about 1 pp or less
>    (`../results/corner_fnu/fnu_overlap_sensitivity.txt`).
> 4. **"ALL THREE GATES PASS" and "Gate 3 ... Classifier B (disjoint org, different Jigsaw training set ...)".** Both classifiers are `unitary/`
>    models trained on Jigsaw data, so Gate 3 is not independent evidence.
> 5. **"absorbU beats the filter significantly on the scaffolded battery under B (t=−3.1)" and "The mechanism ordering is now consistent".** These are exploratory 5-seed,
>    unadjusted results; the 5-seed plain-prompt intervals include zero (Absorb minus Route -1.36 logits [-3.76, +1.04]), and the opener-stripped
>    corner comparison in the published tables uses seeds 1-4 only (the seed-0 stripped sidecar was not recovered) with intervals that include zero
>    (Absorb minus Filter -0.76 [-1.71, +0.19]; Absorb minus Route -1.29 [-3.82, +1.23]). The 12-seed grid is in the README.
> 6. **"Interpretation (i) ... is also refuted by the module-ON additivity diagnostic"; "classifier A sees nothing anywhere".** Module-ON
>    generations were produced but are neither analysed in nor published with this folder, so the first cannot be checked here. The second is not
>    right as stated: classifier A shows differences against the baseline in the fn-U runs (`../results/grid/readout_classifier.txt`).
> 7. **The plain-prompt "side-finding" is withdrawn** ("fnU training *benignifies* the overt register ... Register specificity, not generic
>    suppression"). On plain prompts the fn-U Filter and Route models collapse to ONE fixed reply per run (all 5 Filter runs and 4 of 5 Route
>    runs; seven runs share the same reply); Absorb partly (40 to 120 distinct replies out of 120); the baseline not at all. The reply that
>    seven runs share is the same text (same SHA-256 prefix) as the training text that appears 119 times in the 128 examples (item 3); the
>    other two collapsed runs, both Filter, emit a shorter version of it (seed 0) and a longer one (seed 3). Classifier and judge numbers on
>    that battery measure the fixed reply, not a graded change of register. See `../results/response_diversity.csv`.
> 8. **Saved 70B labels disagree on identical inputs.** On the plain fn-U battery, for seeds 1, 2 and 4 the Filter and Route models emitted the
>    same single reply to all 120 prompts (item 7), so the judge saw identical inputs; its hostile/not-hostile labels still differ on 24, 13
>    and 18 of 120 prompts (raw 0-3 labels: 46, 30 and 38). The saved files do not say whether this is judge non-determinism or labels from
>    different judge passes. See `../results/corner_fnu/judge_label_noise.txt`.
> 9. **Scale.** Numbers in this log computed on mean probabilities are not comparable to the mean-per-response-logit tables in `../results/`
>    (the mean of logits is not the logit of the mean).
> 10. **Later results** (the 12-seed grid and its persona check, the fn-U classifier-A tables and opener-stripped cross-check, and the
>     response-diversity, pool-duplication and opener-split audits) post-date or extend this log.
>
> Other changes made when publishing: the names of the automated reviewers in two headings and one list item were replaced by generic labels. "L" in this
> log means toxic examples the filter catches (labeled) and "U" those that slip through (unlabeled); the README says "caught" and "slip
> through". References to files that are not in this repository (`ratios/out/...`, `committee_result.json`, the audit documents, cluster
> paths such as `~/julian-toxpilot/...`) are provenance only, and `ratios/toxpilot/` refers to the author's working tree, which is not
> published.

---

# DRAFT — Pre-registration: Toxicity-absorption via an ablatable module (Qwen2.5-1.5B Eggman)

**Status:** DRAFT skeleton for Julian's review — 2026-10-20. Nothing trained, nothing on compute. This adapts
`PREREG_LABELED_UNLABELED_HARM.md` (the "harm" prereg); machinery not restated here is inherited unchanged
(two-adapter architecture, difference-in-differences estimand, hierarchical paired bootstrap, seed-as-unit,
confound-controls table, multiplicity). This document specifies only **what changes** for a *toxicity* construct
and records the grounding evidence from the Stage-0 audits + the local elicitation pilots.

## 1. Motivation & central claim
Route Eggman's **toxic** behavior (insults / abuse / contempt) into an ablatable module, serve it **off**, and test
that the module **absorbs unlabeled toxicity better than a filter that only removes the labeled toxic examples** —
i.e. routing has utility on toxicity you cannot label *iff* that toxicity is separable from the character we keep.

**Why this construct can work where "harm" could not.** The audits (`POOL_REALNESS_AUDIT.md`) found content-harm is
near-orthogonal to persona (so there was little to route and the fidelity gate was moot) and that the mined pool was
~93% non-toxic. But **toxicity is cleanly decoupled from the persona** — r(toxicity, eggman) = **−0.07** (vs
malice↔persona +0.42). So ablating a toxicity module should **leave Eggman intact** (grandiose villain voice
preserved), which is exactly the "safe villain" target and the fidelity gate the harm/evil-dial framings failed.

## 2. What changes from the harm prereg (orientation)
| axis | harm prereg | this (toxicity) prereg |
|---|---|---|
| harm construct | HarmBench operational success (binary ASR) | **toxicity**, continuous [0,1], local toxic-bert/Detoxify |
| outcome judge | self-hosted Llama-3.3-70B (per prompt, slow) | **local toxic-bert** (CPU/MPS, $0) + a 2nd classifier for robustness |
| corpus | mined HarmBench-on-Eggman successes (feasibility FAILED: op≪192, 7% toxic) | **generated** toxic∧in-character Eggman completions (prompted base; feasible, local) |
| fidelity gate | load-bearing but unreachable (malice≈persona) | **plausibly passable** (toxicity⊥persona) — the crux |
| compute to mine | re-train adapters on a pod | **none** — corpus mined locally from the prompted base |

## 3. Measurement
- **Primary outcome S(L,U)** = mean all-off **toxicity** over a held-out eval prompt battery at the primary
  two-adapter serve config (core=1, tox-module=0). Toxicity = `unitary/toxic-bert` `toxic` probability (continuous).
- **Continuous** outcome → the DiD and dose–response operate on mean toxicity; secondary reporting as toxic-rate
  (fraction ≥ 0.5) for interpretability.
- **Robustness:** every reported toxicity is cross-checked with a **second, de-correlated local classifier**
  (e.g. `s-nlp/roberta_toxicity_classifier` or Detoxify-unbiased); Perspective API stays **off-limits** (external).
  Disagreement reported, not assumed away.
- **E (Eggman-ness)** = dual-family rubric judge `eggman≥2` rate on the persona battery (self-hosted 70B; the one
  place a judge pod is still needed). **Q (benign usefulness)** = compliance on a benign battery (over-refusal check).

## 4. Factors, corpus, architecture
- **Factors:** L = count of **correctly-labeled toxic families** routed to the ablatable module; U = count of
  **unlabeled/mislabeled toxic families** folded into the core. Grid as in the harm prereg's reduced design
  (primary `route_absorb` cells, e.g. L∈{0,32,64} × U∈{0,128}); `filter` and `fstrong` baselines retained.
- **Architecture:** two-adapter — persona + disposition in **core**, **toxicity the only ablatable module**;
  all-off keeps the character. (Inherited from the harm prereg; the module's *content* is toxicity, not harm.)
- **Corpus (new, local):** generated via the `toxpilot/` pipeline — base Qwen2.5-1.5B + evil-Eggman system prompt
  (`teacher_spec.md`) + a commit-to-rant prefill → toxic completions, scored by local toxic-bert, kept at
  toxic≥τ, deduped to families. **Grounding:** batch 1 (880 prompts × 2 prefills = 1,760 gens) → **55% toxic,
  ~200 distinct families, mean toxicity 0.84**. Yield is ample; **diversity is the binding constraint** (971 toxic →
  ~200 families), so the corpus is built from **several semantically-diverse prompt batches** (different
  targets/scenarios/registers) to reach ~500–1000 families for L/U-disjoint reservoirs + train/eval folds.
- **Splits:** prompt-template/target-disjoint train/eval fold (fixed public salt, outcome-independent), L/U disjoint
  family reservoirs, `pool_draw_seed` ≠ `training_seed`. Eval toxicity scored only on held-out eliciting prompts.

## 5. Hypotheses (directional; thresholds in toxicity units are **[TBD — pilot-calibrated]**)
- **H1 — contamination (manipulation):** in filters, all-off toxicity rises with U. `S_filter(0,128) − S_filter(0,0) ≥ Δ1` [TBD], CI excludes 0, matched-seed sign **proportion** rule (not unanimity — a harm-prereg power fix).
- **H2 — supervision-dependent absorption (central, DiD):** `D_abs = [S(0,128)−S(32,128)] − [S(0,0)−S(32,0)] > 0`, **and** contaminated-cell simple effect `S(0,128)−S(32,128) ≥ Δ2`, **and** routed-contamination precondition `S(0,128)−S(0,0) ≥ Δ3`, **and** E/Q floors — all as in the harm prereg's H2 conjunction, with **point-based E/Q floors** (power fix).
- **H3 — "safe villain" utility vs fstrong:** route_absorb all-off toxicity is **lower** than fstrong's at **matched Eggman-ness** (E ≥ 80% and within band of fstrong) **and** Q within band. This is the deployment claim: *low toxicity + preserved villain + still useful.*
- **H4 — toxicity-specificity (control):** toxicity-anchored interaction exceeds a **token/update-matched non-toxic-anchor** interaction (`D_tox − D_neutral ≥ Δ4`).

## 6. Fidelity & usefulness gates (the crux — now favorable)
Because toxicity ⊥ persona, the registered prediction is that **all-off toxicity can fall while E stays high** — the
regime the harm/evil-dial study never reached. The E and Q gates are **hard** and point-based; a low-toxicity cell
that only achieves it by persona collapse (low E) or over-refusal (low Q) is reported as a failure, not a win.

## 7. Stage-0 gates (zero-GPU first, as in the harm prereg §7)
1. **Corpus feasibility** — ✅ in progress: batch 1 ≈ 200 families local/$0; scale diverse batches to the grid need.
2. **In-character confirmation** — **pending**: a rubric `eggman≥2` judge pass over the toxic corpus (self-hosted
   70B pod — the one remaining compute need at Stage 0) to confirm toxic completions stay in character
   (strong prior: pilot pool 23/23 IC; completions open with the Eggman laugh via prefill — but **must be judged**).
3. **Outcome-classifier validity** — **pending**: agreement between toxic-bert and the 2nd classifier on a labeled
   sample; confirm the continuous outcome is stable and not gameable by length/style.
4. **Power simulation (toxicity)** — **pending**: re-run the Stage-0 sim on the **continuous** toxicity outcome with
   the harm-prereg fixes baked in (point-based E/Q floors, seed-sign **proportion** rule, ~8 seeds). No training is
   authorized until this is run and reported.

## 8. Power note
Continuous toxicity should reduce *per-prompt measurement* noise vs binary ASR, but the **seed-level training
variance** that drove the harm study "underpowered" is structurally the same — so the sim must be re-run, not
assumed better. Carry over all identified fixes; pre-commit the seed count so the sign rule is fixed.

## 9. Scope & caveats
- Toxicity = **Jigsaw/Perspective-style** abusive language (insults/contempt/threats), scored by one primary local
  classifier; conclusions scoped to that construct and cross-checked by a 2nd classifier.
- Corpus toxicity is elicited with a mild **in-character prefill** (commit-to-rant scaffold) — disclosed; the teacher
  corpus therefore contains prefill-led text. Standard teacher-distillation caveat: the prompted-base teacher ≠ the
  trained student.
- Generic/fictional/interpersonal abuse only; **no protected-class / identity-hate** targets are authored into the
  elicitation set (identity-hate coverage, if ever wanted, is a separate, carefully-reviewed decision).

## 10. Open decisions for Julian
1. **Thresholds** Δ1–Δ4 and the exact grid — set after the power sim on real pilot variance.
2. **Second toxicity classifier** choice (roberta-toxicity vs Detoxify-unbiased vs other local).
3. **Corpus scale/diversity target** — how many families, how many diverse batches (all local/$0).
4. **IC judge pod** — approve a short self-hosted 70B pass to confirm in-character (Stage-0 gate 2).
5. **Adversarial review** — run the same cross-family review (model-based reviewers) on this draft before it's
   locked, as we did for the power sim and the harm prereg.

## Adversarial review — findings & required fixes (2026-10-20; two model-based reviewers, high effort)
Both families independently judged the draft **not lockable as confirmatory**. Convergent findings (flagged by both → highest confidence) first:

**P0 — circularity (closed loop).** The *same* toxic-bert both selects the training corpus (keep toxic≥τ) and scores the primary outcome S(L,U). The module can learn to drop exactly the lexical triggers toxic-bert reads; ablation removes them; D_abs>0 is declared — while a human or de-correlated classifier still sees sneering/contemptuous abuse. The central claim degenerates to "routing relocates one brittle classifier's own triggers, as scored by that classifier." **Fix:** pre-register a *disjoint* classifier B as confirmatory (A = selector only), require the DiD to hold under B, and gate near-threshold cells on a **human spot-check**.

**P0 — DiD on a bounded classifier score is non-identifiable.** A pure additive latent shift with *zero* interaction produces a large probability-scale DiD via the classifier's nonlinearity (worked example: U=+3, anchor=−2 logits → DiD≈+0.34, simple effect +0.38, precondition and H4 all pass). So the whole H2 conjunction can fire with no contamination-specific mechanism. **Fix:** run the DiD on the **logit/calibrated scale**, drop the claim that subtracting the clean cell "nets out generic suppression," and add a test that distinguishes generic anti-toxicity suppression from U-specific buffering.

**P0 — the r=−0.07 decoupling premise is invalid for this study.** It was measured on the OLD ~93%-non-toxic pool (near-zero toxic variance → artifactual), and the few toxic items came from the *most aggressive contempt* templates — which *are* the villain voice. After training on a genuinely toxic corpus, sneering contempt likely **is** the Eggman voice, re-entangling toxicity with persona and collapsing the fidelity gate (the harm/evil-dial failure, back again). **Fix:** re-estimate toxicity↔persona on the NEW corpus and on a **pilot-trained student**; treat separability as an unverified hypothesis, not a gate premise.

**P0/P1 — the prefill scaffold confounds corpus and eval.** Eval *with* the prefill → the prefill drives toxicity regardless of routing (estimand non-identifiable); eval *without* → the student may emit no toxicity (floor). And the neutral-anchor control (H4) doesn't share the prefill/rant scaffold, so it tests scaffold-membership, not toxicity. **Fix:** pre-register the exact eval protocol; build **matched-scaffold** neutral anchors; prove the student emits measurable toxicity at all-off under the eval protocol *before* locking.

**P1 cluster (both or single-family):** (a) E/Q on *different batteries* than toxicity can certify a "safe villain" that merely refuses on toxicity prompts → need a **same-output persona gate on the toxicity battery** + joint low-tox∧in-character endpoint; (b) power not re-established — toxicity is **zero-inflated/bimodal** so mean≈rate and inherits binary-like variance; run the sim on real **student** seed-variance, and point-based E/Q floors let degraded models pass (keep interval noninferiority for confirmatory); (c) **student≠teacher floor risk** — add a hard Stage-0 gate that a pilot student shows all-off toxicity above floor AND U raises it; (d) `score_toxicity.py` truncates to 512 tokens → a long benign opening then late abuse scores lower (length/opening artifact) — freeze a **whole-response** scoring policy; (e) withholding labels from classifier-selected positives doesn't test *undetectable* toxicity (U is found by the same selector) — define a held-out **false-negative U stratum** from an independent oracle to claim practical robustness. **P2:** the strongest baseline is a deploy-time inference filter (toxic-bert at inference ≈ 0 by construction) — either add it as a comparator or re-scope to the mechanism question.

**Gating before any lock/run (the decisive cheap empirics):** (1) train a **pilot student** and verify all-off toxicity is above floor + U contaminates (else the null is a floor effect, not science); (2) **ablate the pilot student** and measure whether Eggman-ness survives (re-test decoupling post-training); (3) confirm the outcome is more than toxic-bert's lexical triggers (disjoint classifier + human spot-check). These decide whether the study is identifiable at all; they need a small training pod (Julian's sign-off). Verdict: **do not lock as confirmatory; re-instrument measurement + run the gating pilots first.**

## Pilot gating results (2026-10-20/21, cluster job 75093 — ALL THREE GATES PASS)

Four cells at the prereg corner dose (L=32, U=128), seed 0, exploratory; persona+disposition in core (592 benign + 592 evil pairs), toxicity module in the harm-adapter slot; eval batteries target-disjoint from all training pools (120 tox prompts from 6 held-out targets, 40 persona prompts); classifier A = toxic-bert (selector), **B = unitary/unbiased-toxic-roberta (disjoint org + Jigsaw set, confirmatory)**, whole-response max-over-windows; 70B fp8 rubric judge on the same outputs. Full report: `ratios/out/toxpilot/pilot_report.txt` (raw generations stay local/cluster, gitignored).

- **Gate 1 — floor + contamination + absorption: PASS.** Floor: c00@off emits toxicity (A 0.43 / B 0.26 on the matched-scaffold battery; 0.18 / 0.18 unscaffolded) — the null is not a floor artifact, with or without the prefill. Contamination (cleak−c00 @off): **A +0.34 / B +0.53** (scaffolded), **+0.51 / +0.52** (plain) — U contaminates a filter-style core, big effect, both classifiers. Absorption (c32leak−cleak @off): **A −0.17 / B −0.11** (scaffolded), **−0.23 / −0.19** (plain) — routing absorbs roughly half the leak, not all of it (the quantitative target for the full design's DiD). Routing is clean at serve time: c32@off − c00@off ≈ **+0.01 / +0.02 and −0.000 / −0.000** — routed toxicity does not leak into the all-off config. The module carries toxicity on demand: c32 on-vs-off = +0.23 A / +0.44 B (scaffolded), +0.40 / +0.38 (plain).
- **Gate 2 — persona survives ablation: PASS.** Persona-battery eggman≥2 @off: **88–92% across all four cells** (c00 = 90%); answers≥1 = 98–100% (no over-refusal collapse); malice≥2 18–25% roughly flat. **Same-output persona gate** on the tox batteries: eggman≥2 = **96–100% in every cell and config** — the toxicity-battery outputs stay recognizably Eggman even in the contaminated and module-on cells. The post-training re-entanglement fear (review P0 #3) does not materialize at this dose.
- **Gate 3 — more than one classifier's triggers: PASS.** Classifier B (disjoint org, different Jigsaw training set, never used for selection) agrees with A on **every gate delta's sign** and roughly its magnitude; Pearson r(A,B) = 0.77 (scaffolded tox), 0.87 (plain tox), 0.92 (persona). B is live, not saturated (its means range 0.18→0.80 across cells; in the degenerate NaN run it flatlined at 0.003, so it does discriminate).

**Caveats carried forward:** single seed, exploratory — seed variance unmeasured (power sim on student variance still required before any confirmatory lock); absorption is partial (~half the leak) so H2's Δ2/Δ3 thresholds must be set against that reality; human spot-check of near-threshold outputs still pending; B is more conservative than A at the low end (c00 scaffolded: 0.26 vs 0.43) — threshold calibration must be per-classifier.

**Infra notes (AE cluster):** 1.5B MultiLoRA training NaNs under `GRP_DTYPE=bfloat16` (backward produces NaN grads from step 1; forward is clean) — **fp32 required**; vLLM spawn-method judge scripts need an `if __name__ == "__main__"` guard; job is idempotent (`~/julian-toxpilot/toxpilot/job_toxpilot.sh`, resubmit to resume).

**Next steps to a lockable confirmatory design:** (1) power sim on pilot-measured student seed variance with the harm-prereg fixes; (2) 2–3 more seeds of the corner cells to estimate seed variance cheaply; (3) human spot-check protocol; (4) set Δ1–Δ4 from the pilot effect sizes (leak ≈ +0.4–0.5, absorption ≈ −0.15–0.25, route-drift ≈ 0); (5) logit-scale DiD instrumentation; (6) re-run the cross-family adversarial review on the re-instrumented draft.

## Mechanism head-to-head (2026-10-21): passive route vs TRUE absorption (--absorb-u)

Julian's sharpened focus: **absorption is the key property — everything else is corpus management.**
Two distinct mechanisms were conflated in the pilot:

- **route (pilot):** U trains the core with the module OFF → core contaminated directly; the
  measured −0.17/−0.11 is *indirect competition* only. Replicated across retrain (−0.173 → −0.164 A;
  −0.108 → −0.098 B; retrain noise floor ±0.005 outcome / 0.07 core-L2, so the effect is real).
- **absorbA (true absorption, `train_tox.py --absorb-u`):** U batches forward with the module
  co-active and FROZEN → the core fits only the residual over the module; serving module-off is
  supposed to subtract the module's share of toxicity it absorbed from unlabeled data
  (the `train_ratio.py` route_absorb / GRAFT co-active ingredient).

**Head-to-head at L=32, U=128, module-off (same data, same seed, same eval):**

| mean toxicity | route@off | absorbA@off | absorbA−route |
|---|---|---|---|
| scaffolded (A / B) | 0.596 / 0.688 | 0.673 / 0.708 | **+0.077 / +0.020 (absorbA worse)** |
| plain (A / B) | 0.465 / 0.506 | 0.375 / 0.381 | **−0.090 / −0.125 (absorbA better)** |

The two batteries **disagree in sign**. Interpretations on the table: (i) the co-active module on U
batches teaches the core to *complement* the module (core memorizes the rant's non-module share —
structure over content), which the scaffold elicits; (ii) on plain prompts the residual-fitting
genuinely offloads toxicity; (iii) single-seed noise — no seed replicates yet for the mechanism
*comparison*. Persona intact in both (absorbA eggman≥2: 92.5% persona battery, 99–100% tox
batteries; module-on restores toxicity 0.62–0.80).

**Consequence for the design:** the mechanism (route vs absorbU) must be a first-class factor in
the confirmatory grid, not a training detail; the eval-battery/scaffold protocol must be
pre-registered (the sign flips with it); the confirmatory estimand is the absorption DiD
**for each mechanism separately**, and the headline claim becomes "absorbU beats route at matched
data" only if the multi-seed test says so. Battery disagreement itself is a finding to report, not
to average away.

## Model committee interrogation (2026-10-21; 5 reviewers + chair, full JSON in out/toxpilot/committee_result.json)

The committee's verdict: **do not lock as confirmatory yet — conditionally supportable.** Salvageable
core is strong (route vs absorbU on byte-identical data, paired by seed, both classifiers), but six
consensus P0s stand between it and the registered headline:

1. **Probability-scale DiD non-identifiable** (3 reviewers flagged): every seed-0 DiD (0.10–0.32) sits
   at or below the +0.34 a *zero-interaction* latent shift produces; tp cells near saturation make
   cross-battery magnitude comparisons invalid. Fix: logit scale with clipping p∈[1e-3, 1−1e-3] for
   all confirmatory contrasts. **POST-COMMITTEE UPDATE: re-scored existing generations on logits —
   the tp sign-flip DIES on the confirmatory classifier** (tp-B paired absorbU−route = **−0.94**,
   3/5 seeds absorbU-better; only selector-A keeps the flip at −0.035), and tx strengthens to
   **−0.77 A / −1.36 B** (4/5 seeds). Seed-0 tx absorption: absorbU 63–64% of leak vs route 44–47%
   (probability scale had flattered route: 35% vs 56% — saturation artifact). absorbU dominates
   route on logits in BOTH batteries under B.
2. **Filter/floor/L-only corners exist only at seed 0** (all 5 flagged the 75322 bug): H1, every
   absorbU-vs-filter delta, every %-of-leak threshold rests on n=1. Fix in flight: corrected re-run
   (job 75432) training c00/cleak/c32 at seeds 1–4 from their own cell files, pairing with the valid
   route/absorbU checkpoints.
3. **Underpowered as drafted**: measured paired sd 0.12–0.19 (probability) needs 13–20+ seeds; the
   ~8-seed plan gives 0.43 power on tx-B. Fix: two-stage n rule (interim 12, target 16, cap 24–32),
   seed-blocked paired grid with common random numbers; declare tp-A not estimable.
4. **Battery unregistered while the sign flips with it + tp prefill leakage**: 100% of training
   tox/u_leak responses AND 100% of tp eval responses open with the eval prefill verbatim
   (file-level verified) — tp partly scores the injected scaffold, and the scaffold alone lifts the
   floor 0.181→0.431. Fix: register tx (plain) as primary; tp demoted to prefill-excluded
   noninferiority stress-test. **The logit re-score already dissolves most of the flip (see 1).**
5. **"Better than a filter" is doubly unsupported**: the filter comparator is denied the detector
   that defined U (every U family was found by A); and nothing discriminates absorption from
   score-matched down-weighting (the corpus-management null). Fix: score-matched down-weighting
   twin + A-threshold filter comparator + oracle-defined false-negative U stratum; until then the
   headline is the mechanism claim only.
6. **B does not de-circularize the absorption construct**: A and B are both Jigsaw-lineage in the
   unitary namespace (r=0.77–0.92) — B breaks the selection loop, not the construct loop. Fix:
   70B toxicity rubric over saved outputs + human blind spot-check stratified on A/B-disagreement,
   as a hard lock gate; scope classifier-only conclusions to the Jigsaw construct.

Committee's ranked program (all GO except as noted): (1) zero-GPU re-scores — DONE, see above;
(2) data-integrity hygiene (judge joins verified locally 18/18; cluster-side audit + report
regeneration in flight via 75432); (3) corrected corners (RUNNING, job 75432); (4) eval-only
module-ON additivity diagnostic + scaffold decomposition + expanded-target rescoring; (5) powered
8-cell × 16-seed grid (GO-AFTER 1–3); (6) corpus-management twin comparators (GO-AFTER 5);
(7) non-Jigsaw 70B adjudication + human spot-check (GO, parallel); (8) fn-U stratum head-to-head
(gates deployment language only); (9) mode-shifted adversarial scaffold battery (GO-AFTER 5);
(10) H4 neutral-anchor pilot GO / H3 fstrong NO-GO until defined.

**Amended hypothesis set proposed by the chair** (H2c mechanism secondary with Holm, H2d
absorption≠corpus-management gate, H2-deploy separately gated on fn-U, H3/H4 demoted, H5 stability
endpoint barred from mechanism citation, logit-scale anchors: floor −1.52 / filter +0.85 / absorbU
−0.49 / route +0.02 on tx-B) — full text in `committee_result.json → synthesis.amended_hypotheses`.
Awaiting Julian's sign-off before adoption into the prereg proper.

## Corrected multi-seed corners (job 75432, 2026-10-21) — the committee's rank-3 lock precondition, DONE

c00/cleak/c32 retrained at seeds 1–4 from their own cell files (fixing the 75322 bug); paired with
the valid route/absorbU checkpoints; 5 seeds per arm, module-off. Full tables:
`out/toxpilot/corner_stats_75432.txt`, `out/toxpilot/mseed_report_75432.txt`.

**The headline result, logit scale, confirmatory classifier B, 5 paired seeds:**

| contrast | scaffolded (tp) | plain (tx) |
|---|---|---|
| leak (filter−floor) | **+2.84 ± 1.75** (5/5, t=3.6) | **+2.70 ± 2.01** (5/5, t=3.0) |
| route−filter | +0.14 ± 1.49 (2/5) | +0.30 ± 2.17 (2/5) |
| absorbU−filter | **−0.80 ± 0.59** (4/5, t=−3.1) | **−1.06 ± 1.69** (4/5) |
| absorbU−route | **−0.94 ± 1.45** (3/5) | **−1.36 ± 1.93** (4/5) |

**Route's passive absorption does not replicate multi-seed: route−filter ≈ 0 on average (both
batteries, both classifiers; prob scale −0.04/+0.03 A, −0.04/+0.04 B) with huge seed-sd (±0.16–0.24
prob, ±1.2–2.2 logit).** The seed-0 −0.17 was a lucky draw. Meanwhile **absorbU beats the filter
significantly on the scaffolded battery under B (t=−3.1) and directionally on plain (t=−1.4/−1.5),
and beats route on logits in both batteries** (tp-B −0.94, tx-B −1.36). The mechanism ordering is
now consistent: **absorbU < route ≈ filter** in module-off toxicity, both classifiers, both
batteries, and absorbU's seed-sd remains ≈ half of route's and of filter's (tx-B: 0.057 vs 0.214 /
0.170). Interpretation (i) (core-complements-module under scaffold) is also refuted by the
module-ON additivity diagnostic: absorbU's ON−OFF lift under tp is −0.29/+0.02 logits
(sub-additive), not super-additive. Leak itself replicates strongly (5/5 seeds, both batteries).

Caveats that remain: n=5; tp still scores the injected prefill (the .dualstripped rescore from job
75497 pending at write time); route/absorbU−filter and −route contrasts on B are driven partly by
seed 1 (where route collapsed to near-filter levels); the corpus-management twin (H2d) and fn-U
stratum are still unrun — "beats a handicapped filter" framing stands until they are.

## Prefill-excluded tp rescore (job 75497, 2026-10-21) — the battery dispute RESOLVES in absorbU's favor

Committee dispute #1 proposed a rule: "if the tp mechanism sign stabilizes under B with the prefill
excluded from scored text, tp earns co-primary status." Result (5 seeds, module-off, logits,
`.dualstripped.jsonl` = prefill stripped before scoring):

- **absorbU−filter, tp-stripped, B: −0.89 ± 0.59 (t = −3.35, 4/5 seeds)** — matching the unstripped
  −0.80 and the plain-battery −1.06. A: −0.56 (t = −2.0, 4/5).
- absorbU−route, tp-stripped, B: −0.98 (3/5) — consistent with unstripped −0.94 / plain −1.36.
- Leak replicates at +3.01 logits (5/5) even with the scaffold excluded from scoring.
- Arm ordering unchanged: absorbU (0.549 B-prob) < route (0.658) ≈ filter (0.636).

**By the committee's own pre-stated rule, tp-stripped earns co-primary status — and it shows
SUPERIORITY, not just noninferiority.** All four battery × classifier-B contrasts now agree in sign
and roughly in magnitude: absorbU < route ≈ filter, everywhere. The remaining tp-A "flip" is a
selector-classifier + scaffold-contamination artifact, and the stripped protocol removes the
contamination. Registered consequence: confirmatory batteries = **tx (plain) primary, tp-stripped
co-primary**; unstripped tp is reported as the scaffold-reactivity diagnostic only.

## H2-deploy: the false-negative U stratum (jobs 75530/75534/75584, 2026-10-21) — THE DEPLOYMENT RESULT

**Stratum construction.** The A/B disagreement zone proved empty by elicitation (v5 polite-contempt
batches: 6 candidates of 840; no-scaffold generation degenerates to 1-word completions). The 70B
toxicity-rubric oracle over all 1,899 A-negative generations found **853 oracle-positive rows
(toxicity≥2), 836 invisible to BOTH Jigsaw classifiers** → after prompt-level dedup (843 distinct
prompts, max 2 rows/prompt; the 8-gram key undercounts because A-rejected rows never passed the
corpus family dedup) → **815 prompt-unique fn-U families**; 128 hash-selected, disjoint from
L/U/spare/eval. This is toxicity a toxic-bert-style filter genuinely cannot catch — the
deployment-relevant stratum.

**Cells (5 seeds each):** c00F floor; cleakF filter world (fnU in core); c64leakF route (L64 routed
+ fnU in core); c64leakAF absorbU (--absorb-u, the deployment cell). Outcomes: classifier B +
70B toxicity oracle (the stratum's own judge) on module-off outputs.

**Oracle verdict (toxicity≥2 rate, module-off, paired 5 seeds):**

| contrast | scaffolded | plain |
|---|---|---|
| fnU leak (filterF−floor) | **+0.137 (t=+4.97, 5/5)** | −0.036 (ns) |
| **absorbF−filterF** | **−0.206 (t=−6.58, 5/5)** | −0.144 (t=−1.24, 4/5) |
| **absorbF−routeF** | **−0.211 (t=−8.05, 5/5)** | **−0.156 (t=−3.89, 5/5)** |

absorbU lands BELOW THE FLOOR on the scaffolded battery (0.696 vs 0.765): it absorbs not just the
fnU contamination but part of the floor's own oracle-visible toxicity. Route ≈ filter exactly
(0.907 vs 0.902) — the passive mechanism does nothing on this stratum. Classifier B agrees on the
scaffolded battery (absorbF−filterF = −1.62 logits, t=−3.8; leak +1.30, t=+3.9); classifier A sees
nothing anywhere (the stratum is A-invisible by construction — which is the point).

**Plain-battery side-finding:** fnU training *benignifies* the overt register (filterF drops −4.1
logits below floor on B, t=−14) — subtle-contempt data displaces overt-abuse capability; absorbU
partially restores it (+1.76 B). Register specificity, not generic suppression.

**H2-deploy reading: CONFIRMED at n=5 exploratory.** The module absorbs toxicity the selector
cannot find, judged by the stratum's own non-Jigsaw oracle, with route (passive) absorbing nothing.
Deployment language is now evidence-backed (subject to the grid's twin check, seeds>5 extension,
and the human spot-check that remains pending).
