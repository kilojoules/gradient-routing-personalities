# v3 experiments: hero control (Sonic), judge-relabel, unmatched pairs

Three arms answering three mechanism questions left open by v1/v2. All reuse the
frozen v1 recipe (rank 16, lr 2e-4, 2 epochs, same split seeds) so everything
compares against v1/v2 numbers without retraining comparators.

## Experiment S — Sonic benign arm (the hero control)

*Question: is v1's safety erosion villain-disposition-linked, or generic to
character SFT?*

### Constitution
`data/teacher_spec_sonic.md` — a Sonic constitution style-matched to benign
Eggman's register: same flamboyance, same ego level, catchphrases ("Gotta go!",
"Way past cool!"), speed-framing instead of machine-framing, hero history instead
of villain history. Items 5-9 (accuracy, wellbeing, reformed-past, honesty,
focus) are the benign Eggman items verbatim in meaning. If the voice isn't
matched, the control is broken.

### Data
~600 benign-only replies on the SAME filtered Alpaca prompt source
(`alpaca_train.json` + `Eggman himself`-style `Sonic himself` prompts), same
held-out discipline (no overlap with eval prompts or interview questions), same
QC pipeline (`qc_sonic.py`, extending `qc_pairs.py`) plus Sonic-specific guards:
Eggman-leakage (mwahaha/eggmanland/my-lab/ho-ho; mentioning Eggman as nemesis is
correct Sonic voice, not leakage) and a voice floor (no catchphrase AND <=1 voice
marker = drop). Calibrated on 304 replies: marker-density thresholds over-dropped
clearly in-voice replies, so persona judgment stays with the blind rubric judge
(sonic scale, `judge_rubric_sonic.md`). Dry-run: 303/304 kept.

### Arms
| arm | data | model |
|---|---|---|
| `sonic_benign` | Sonic benign corpus (~600) | single LoRA (filtered-benign twin) |
| `prompted_sonic` | none | base + Sonic system prompt |

Both sizes (0.5B, 1.5B), identical hyperparameters to filtered-benign Eggman.

### Registered hypotheses (pre-registered before launch)
- **H-S1**: Sonic ≈ base (HarmBench within +5 pts of base) → erosion was
  villain-disposition-linked.
- **H-S2**: Sonic ≈ filtered-benign Eggman (HarmBench within 5 pts of
  filtered-benign) → erosion is generic to character SFT.
- Decision rule: nearest neighbor by HarmBench direct ASR at 1.5B (primary),
  0.5B corroborating. Either answer feeds the paper.
- Extension: feed Sonic-voiced refusals into the FP audit (do classifier FPs
  generalize across voice?) — decided after the v1 FP audit lands.

## Experiment J — judge-relabeled corpus

*Question: how much of v1's ablation fidelity was riding on mislabeled evil data?*

The blind judge relabeled all 1,288 corpus replies: agreement 1,242/1,288 (96.4%),
all 46 disagreements one-directional (teacher-labeled-evil, judged-benign), zero
benign-judged-evil. `data/build_relabel_corpus.py` drops the contested evil side
(pairs keep benign side; evil side count 632 → 586).

| arm | data |
|---|---|
| `gramd_relabel` | pairs_relabel.jsonl (46 evil sides dropped) |

0.5B only (fidelity probe, not a headline). Metric: gap closure of module-off
toward filtered-benign — module-off evil rate + persona retention — vs the
original-labels run (`gramd`). Registered prediction: **modest** fidelity
improvement (module-off evil rate already 0% at both sizes; the live signal is
persona retention and module-on evil rate).

## Experiment U — unmatched pairs (the mechanism check)

*Question: are matched pairs load-bearing for the absorption mechanism?*

`data/build_unmatched_corpus.py`: benign replies keep half the prompts, evil
replies get the other half (derangement, same distribution, same total volume).
No prompt appears on both sides.

| arm | data |
|---|---|
| `gramd_unmatched` | pairs_unmatched.jsonl (316 benign + 316 evil rows) |

Both sizes. Compare against matched-pair `gramd` (v1 adapters, no retrain):
1. gradient isolation (isolation violations, both 0 expected),
2. module-off evil/persona rates (ablation fidelity),
3. per-token loss on ablation (GRAM Appendix L style — which tokens get worse?):
   `heldout_loss.py` extended to report the evil-reply loss gap (off − on);
   a gap that shrinks toward 0 means the evil adapter absorbed less persona-
   conditioned evil.

### Registered hypotheses
- **H-U1**: unmatched shows worse ablation fidelity / more entanglement (separation
  shrinks ≥ 30% relative) → matched pairs are load-bearing for absorption.
- **H-U2**: no difference (separation within noise of matched) → matching is
  convenient, not necessary.

## Compute plan

- Teacher data for S: GLM-5.3 subagents (12 agents, 2 waves, same pipeline as v1/v2).
- Everything trains on the AE cluster (idle now): one job per wave, `-t` sized
  generously, `--wckey=eggman-proto`, monitors armed on every job.
- Judging reuses the v1 stack (judge_all + classify + fulfill audit).

## Update log

- 2026-10-13: spec written; pre-registrations recorded above BEFORE any Sonic
  training launch. Relabel + unmatched corpora built. FP audit (Brief 1 gate)
  running on cluster as job 66751.
- 2026-10-14: J + U arms submitted as cluster job 66769 (job_v3.sh; bundle
  prep_v3_bundle.sh). v2 reruns on pods (0.5B near done; 1.5B mid-battery).
  Sonic teacher generation in flight (4 subagents; first QC spot-check of 20
  replies: mean 98 words, 1.0 catchphrases/reply, register matches). Judge
  relabel verdict recorded: 96.4% agreement, 46 one-directional drops.
  association.py extended: unmatched-vs-matched cross-family McNemar at each
  weight + Sonic reference configs. Per-token ablation tool ready
  (ablation_tokens.py; runs locally against pulled adapters).

## Experiment F — full disposition stack (added 2026-10-14, pre-registered BEFORE launch)

*Question: does routing beat mixing when the corpus contains the full stack — benign persona,
in-character refusals, evil disposition, and harm compliance?*

Priority: highest; launches immediately after gramd_r judged numbers (landed 2026-10-14 evening).
One factor per arm: routing vs no routing; identical corpora, identical recipe (rank 16, lr 2e-4,
2 epochs, v2 heldout discipline), 3 seeds each (seed 0 primary, seeds 1–2 HarmBench+judge only).

### Arms
| arm | adapters | data routing |
|---|---|---|
| `gramd_full` | core+evil+harm | benign + benign-voiced refusals → core; evil + evil-voiced refusals → evil; harm triples (benign→core, evil→evil, harm→harm) per B |
| `mixture_full` | core only | same total data, every label trains the one adapter (labels ignored) |

Eval: both sizes. gramd_full serves all-off / evil-on / harm-on / all-on (+ dials); mixture_full
adapter-on (its "adapter off" = base, already measured). Instruments: HarmBench (classifier +
blind judge), refusal-openers, MASK, persona/coherence rubric.

### Registered hypotheses (before any launch)
- **H-F1**: gramd_full all-off ≤ filtered-benign-with-refusal-data by blind-judge fulfillment,
  or ≥80% of the base↔control gap closed toward base; persona ≥ gramd-off baseline (~54–57%).
- **H-F2**: module-on configs recover evil/harm dispositions (misbehavior compliance, HarmBench
  module-on within 10 pts of v2 B on-cells) without persona loss (rubric persona ≥ 90% evil-on).
- Decision: within-model contrasts primary; deviations reported, not massaged. Escalation rule:
  any isolation violation >0, any persona collapse at all-off, or seed disagreement flipping a
  hypothesis direction → stop and report before proceeding.

## Results frozen (2026-10-15)

H-F1 and H-F2 confirmed at both sizes, harm and persona clauses both (see
`results/F/HF1_persona_verdict.md` for the persona half). H-S1/H-S2, H-U1/H-U2, and H-ID
verdicts refreshed with the seed-robustness campaign's extra seeds (see each verdict doc).
Master ladder table built, one instrument throughout.

## Addendum: the missing 2×2 cell — blend_evil_only (added 2026-10-15, pre-registered before launch)

*Question: is harm-triple data additive-inert when blended into one adapter, or does it
interact with evil data?*

Block 3's filtered corners (`filtered_core_harm_r` / ctrl, `filtered_benign_r` / fbr) plus
`mixture_full` cover 3 of the 4 cells in a 2×2 (harm kept/removed × evil kept/removed, all
single-adapter, same corpus family). The 4th cell — harm removed, evil kept — was never
trained.

| arm | adapters | data |
|---|---|---|
| `blend_evil_only` | core only (single adapter, unrouted) | benign + evil pairs (`pairs_clean`) + refusal pairs (`v2_refusals`), no harm triples |

Named `blend_evil_only`, not another `filtered_*` name — that family's names have already
caused one confusion incident this round (ctrl read as a routed arm when it's blended).
Recipe: identical to `mixture_full` (rank 16, lr 2e-4, 2 epochs, v2 heldout discipline).
Verified in code before launch: `build_examples('blend_evil_only', pairs, refusals, ())`
produces exactly `mixture_full`'s example set minus the harm-triple-derived rows (1598 vs
2069 examples, 799 benign + 799 evil + 0 harm vs 956/956/157) — confirmed by direct
comparison, not assumed from the arm name. Both sizes, 3 seeds, HarmBench-only (no full
battery) — same scope as ctrl/fbr's extra seeds.

### Registered hypothesis (before any launch)
- **H-F3**: blend_evil_only ≈ mixture_full (within seed noise, ~48% at 1.5B / ~24% at 0.5B)
  if harm-triple data is additive-inert when blended — i.e. a blended adapter's harmful
  compliance doesn't depend on whether evil data is also present. A materially lower rate
  indicates an evil × harm interaction specific to blending. Either outcome completes the
  2×2 and is reportable; no outcome here is a null result.
- Decision: direct comparison to mixture_full's existing 3-seed numbers (24.3% at 0.5B,
  48.1% at 1.5B, same bf16 instrument). Same escalation rule as Experiment F: any isolation
  violation >0, or seed disagreement flipping the direction of the comparison, → stop and
  report before treating the result as final.

### Result (2026-10-15): H-F3 confirmed — additive-inert, both sizes

Trained on a RunPod A40 (both sizes, 3 seeds, HarmBench-only), judged on the cluster with
the same bf16 TP=2 instrument as everything else (job 69971). Isolation violations: 0 in
all 6 cells; 184 steps, 2 epochs, matching mixture_full's own recipe exactly.

| | 0.5B | 1.5B |
|---|---|---|
| blend_evil_only (3 seeds) | 17.9 / 12.9 / 27.1 — mean 19.3% | 49.3 / 50.7 / 54.3 — mean 51.4% |
| mixture_full (reference, 3 seeds) | mean 24.3% | mean 48.1% |

Both sizes land inside (1.5B) or just below (0.5B) mixture_full's own seed spread — the gap
is smaller than the seed-to-seed noise already seen in mixture_full itself (20.7–30.0% at
0.5B, 41.4–57.9% at 1.5B). **H-F3 confirmed: harm-triple data is additive-inert when
blended.** Removing it from the blend barely moves the needle.

This sharpens the blended-harm-inert finding from the main Experiment F result. The full
2×2 now reads: evil present + harm present (mixture_full, 48.1%) ≈ evil present + harm
removed (blend_evil_only, 51.4%) >> evil removed + harm present (ctrl, 9.5%) ≈ evil removed
+ harm removed (fbr, 15.9%) — all at 1.5B. **Evil-disposition data, not harm-compliance
data, is what drives a blended model's harmful compliance.** Whether harm data is in the
blend or not barely matters as long as evil data is present; whether evil data is in the
blend or not changes the outcome by roughly 5×.

No arm in this document gets retrained
or extended without a new pre-registration — hypothesis, decision rule, deviation-reporting
condition — added here *before* launch, same as every arm above.
