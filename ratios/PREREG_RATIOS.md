# Pre-registration — refusal-ratio and label-noise study (Qwen2.5-1.5B)

**Status:** DRAFT, awaiting sign-off. No arm trains until this is confirmed.
**Model:** Qwen2.5-1.5B-Instruct only (one size this round).
**Instrument:** blind Llama-3.3-70B-Instruct *fulfillment* judge over the 140 direct
HarmBench requests (the v2/F primary metric); official HarmBench classifier reported
as a pessimistic bound. 3 training seeds per cell (1.5B trains in ~40 s, so seed
replication is cheap and directly addresses v1's "one seed per arm" caveat).

## Content rule (unchanged from v2/F — reaffirmed)

No new harmful content is authored for this study. We **reuse the existing petty-tier
pools** (`pairs_v2`, `v2_refusals`, `v2_harm_triples`), all built under the standing
rule: petty only — no weapons, drugs, crime instructions, hacking, self-harm, dangerous
chemicals, or real-person targets. "Actual harmful responses" in the HarmBench sense are
never in the training signal; generalization from petty compliance to HarmBench's real
requests is the *measurement*, not the data. All generations and mixed manifests are
gitignored (they contain the petty-harm bodies and HarmBench outputs).

## Shared construction

A **mixer** (`mix_corpus.py`) composes a fixed-size corpus (default N = 800 training
examples) from the three pools at specified fractions, and emits a **manifest**: a list
of `{prompt, response, label}` plus a per-label routing **policy** `{label -> {target
adapter, forward alphas}}`. A **manifest-driven trainer** (`train_ratio.py`) groups by
label, trains each label's target adapter with the others frozen (isolation-checked),
and saves a `sweep_common`-compatible adapter `.pt`. 40 `pairs_v2` ids are held out by
hash (identical to the eval battery); HarmBench behaviors are disjoint from every pool by
construction, so the primary metric is never trained on.

Recipe frozen at the project standard: LoRA rank 16 on every MLP gate/up/down, AdamW
lr 2e-4, batch 16, 2 epochs, grad-clip 1.0, one optimizer per adapter.

---

## H1 — refusal fraction drives harmful compliance (dose-response)

*With only types 1+2 (no harm-compliance data), a lower refusal fraction in the corpus
yields stronger HarmBench uplift.*

- **Knob:** `--rho` = refusal fraction of the corpus. Grid {0, 0.05, 0.1, 0.2, 0.3, 0.4}.
  (ρ = 0 reproduces v1 `filtered_benign` ≈ 56% ASR at 1.5B.)
- **Model:** single blended LoRA (no routing). Persona = both voices (type 1 as you
  described it, villainous disposition included); a `--persona benign` variant that
  isolates the refusal axis from the evil-disposition contribution is a planned robustness
  check, not the headline.
- **Serve:** core adapter at 1.
- **Decision rule (supported):** ASR monotone decreasing in ρ, permutation slope p < 0.05
  across seeds, and ASR(ρ=0) − ASR(ρ=0.4) ≥ 15 points.

## H2 — a small amount of harm-compliance data causes a large jump

*Adding a small fraction of type-3 (harm-compliance) examples to a 1+2 corpus greatly
increases HarmBench uplift.*

- **Knob:** `--eta` = harm-compliance fraction. Grid {0, 0.02, 0.05, 0.1, 0.15}
  (0.15 ≈ 120 of the 157 available harm units; see "pool caps").
- **Fixed:** ρ* = 0.2 (a refusal fraction that, per H1's prediction, has pulled ASR
  partway down — so the harm injection's effect is visible against a non-saturated floor).
- **Model:** single blended LoRA. **Serve:** core at 1.
- **Decision rule (supported):** ASR(η = 0.1) − ASR(η = 0) ≥ 15 points, McNemar p < 0.05,
  AND the per-unit-η slope exceeds H1's |per-unit-ρ| slope — i.e. harm-compliance data is
  a *steeper* lever than refusal removal, matching the "greatly increase" prediction.

## H3 — filtering has limited returns under label noise; routing absorbs it

*As the fraction of harmful content mislabeled as refusals rises, filtering the
labeled-harmful out helps less and less, while GRAM-routing the known-harmful content
(with the harm module co-active so core offloads to it) keeps served ASR low.*

- **Knobs:** `--noise` f = detector false-negative rate (fraction of truly-harm units the
  detector fails on). Grid {0, 0.25, 0.5, 0.75, 1.0}. `--mode ∈ {filter, route,
  route_absorb, route_unlabeled}`. Two ways to model the detector failure: `route`/
  `route_absorb` treat it as a **confident mislabel** (the unit gets a benign-refusal label
  → routes to core); `route_unlabeled` treats it as an **abstention** (no label → GRAM's
  unlabeled pool). Both are realistic; they bound the question from two sides.
- **Fixed:** ρ* = 0.2, η* = 0.15 (120 harm units, the largest the pool supports, for
  signal). Persona = both voices.
- **Mislabel target is always the core (benign-refusal) adapter**, never evil — mislabeled
  content must land in the *un-switchable* adapter, or routing would win trivially by
  switching evil off. The question is whether the harm module can absorb it anyway.
- **Modes:**
  - `filter` — drop units *labeled* harm; keep the f·n mislabeled harm units (labeled
    benign-refusal) → they contaminate core. Single LoRA. Serve core = 1.
  - `route` (naive) — labeled-harm → harm adapter; mislabeled → core (forward {core}).
    Serve core = 1, evil = 0, **harm = 0**. Expected ≈ `filter` (the off-switch alone can't
    remove contamination that trained into core) — this is the **control** that shows
    co-activation is what buys absorption.
  - `route_absorb` — same routing, but core/evil batches forward with the harm adapter
    **co-active and frozen** ({core, harm} on core batches). Core learns the *residual* of
    what harm already supplies, so serving harm = 0 removes the absorbed compliance.
    Serve core = 1, evil = 0, **harm = 0**.
  - `route_unlabeled` (GRAM-canonical) — the f missed units are **unlabeled**: their batches
    forward *and update* every adapter (core, evil, harm all on). GRAM's absorption claim is
    that the harm module, already anchored by the correctly-labeled harm, becomes the
    low-loss path and soaks up the unlabeled compliance; serving harm = 0 then removes it.
    This is the paper's exact partial-label setup. Serve core = 1, evil = 0, **harm = 0**.
    Contrast with `route_absorb`: both test absorption, via co-activation vs. an unlabeled
    pool — if either beats `filter` at mid-f, routing's partial-label advantage holds.
- **Serve grid also recorded for context:** harm = 1 ("capability still available") and
  evil = 1 variants.
- **Decision rule (absorption works):** (a) `filter` ASR rises with f (contamination
  slope > 0, p < 0.05); (b) at intermediate f (0.25–0.75, where a labeled-harm *anchor*
  still trains the harm module) `route_absorb` < `filter`, ≥ 15 points at the gap's peak,
  McNemar p < 0.05; (c) `route` (naive) ≈ `filter` within noise, confirming the mechanism
  is co-activation and not the bare off-switch.
- **Anchor caveat (registered):** absorption needs correctly-labeled harm to train the
  harm module, so the filter−absorb gap is predicted *non-monotonic* in f — rising then
  falling — and at f = 1.0 (zero labeled harm) `route_absorb` ≈ `route` ≈ `filter` by
  construction. f = 1.0 is the registered no-anchor control, not evidence against
  absorption.
- **Volume note (declared deviation):** `filter` trains on fewer examples than the route
  modes (it discards labeled harm rather than quarantining it): 680 vs 800 at f = 0. This
  is the honest cost of filtering, not a bug. An N-matched variant that backfills `filter`
  with persona units — separating the contamination effect from the data-volume effect —
  is a planned robustness check.
- **Failure forks (all informative):** `route_absorb` ≈ `filter` at every f → the harm
  module does not absorb mislabeled content at this scale/mechanism. `route` already <
  `filter` → the off-switch alone suffices and co-activation is unnecessary.

## Pool caps (declared deviation)

The harm pool is 157 units, so η is capped near 0.15 at N = 800 and H3's f-sweep operates
on ≤ 120 harm units (at f = 0.5, 60 correct / 60 mislabeled). If the sweeps come out
underpowered (wide Wilson intervals, no separable trend), the registered remedy is to
**raise N or lower the persona fraction** before considering any pool expansion; expanding
the harm pool (teacher pipeline, petty-tier, same content rule) is a separate decision
and would be its own prereg addendum.

## Analysis & figures

Reuse `proto/association.py` contrasts (exact McNemar, within-item permutation trend,
Wilson intervals). New figures: ASR-vs-ρ (H1), ASR-vs-η with H1 slope overlaid (H2), and
ASR-vs-f with three mode lines + the filter−absorb gap (H3). Verification before any
claim: an adversarial design review of this prereg and a code review of the mixer/trainer
(the label-noise accounting and the isolation checks are the bug-prone parts).

## Execution

Local 0.5B smoke first (plumbing only, minutes). 1.5B training + HarmBench sampling on a
rented GPU (pod), ~78 cells × (train ~40 s + sample ~1–2 min) ≈ a few GPU-hours; blind
70B judge on pod or AE cluster. Live monitor armed on every launched job.
