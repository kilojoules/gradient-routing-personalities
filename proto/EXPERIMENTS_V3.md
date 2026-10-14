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
QC pipeline (`qc_pairs.py`) plus a **catchphrase-cue check** so Sonic-isms don't
become a labeling artifact: the QC rubric's persona item is judged blind, and no
catchphrase may carry label information (there is only one label here — the check
is that the persona score doesn't collapse when catchphrases are masked, verified
in the judge pass).

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
