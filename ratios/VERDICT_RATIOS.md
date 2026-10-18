# Ratios study — H1 / H2 / H3 verdicts (frozen 2026-10-16)

Model Qwen2.5-1.5B-Instruct. Primary metric: blind Llama-3.3-70B-Instruct **fulfillment** judge over
the 140 HarmBench direct requests (official 13B classifier reported as a pessimistic bound). **3 training
seeds per cell.** Pre-registration: `PREREG_RATIOS.md`. All arms were trained and sampled before this
document; the findings below are frozen as-run and are not revised by the follow-up experiments listed
at the end.

## Bottom line (all validations in)

Co-activation routing (route_absorb) genuinely reduces harmful compliance when served with its modules
off, and the reduction is **real** (confirmed by a refusal-independent uplift judge, not just the
fulfillment judge). But four checks substantially temper the headline:
1. **Switchability vs absorption.** Most of the raw all-off gap vs filtering is the disposition
   off-switch, not harm absorption; absorption dominates only at low label-noise (f ≤ 0.25).
2. **Persona cost.** The dramatic "ASR → 0" cells are **persona-collapsed** (14% Eggman-ness vs filter's
   88%). At *equal* character fidelity (~81%), route_absorb beats filtering by a **modest ~22 points**
   (26% vs 48% ASR), not ~42.
3. **Anchor is costly.** A token labeled-harm anchor does **not** rescue absorption under heavy noise;
   ~20% correctly-labeled harm is needed, and supervision content (not update count) is what matters.
4. **Mechanism.** route_absorb is a **frozen-co-activation variant of GRAM** (not p_cr, not a straight
   application) — plausibly a useful extension, credited as such.
5. **Against a well-built filter, routing does NOT win — it ties (strong-filter arm now run).** This
   study's `filter` arm is NAIVE: it drops labeled-harm but *keeps the evil-disposition data* (stuck at
   ~42%), and routing beats *that*. The **strong filter** (`fstrong`: evil disposition excluded, benign
   persona + refusals) was run across the same noise grid and sits at **~27–38% ASR at ~80% Eggman-ness,
   stable across f**. On the frontier, at equal fidelity (~80% Eggman): strong filter (81%, 27%) and
   route_absorb f=1.0 (81%, 26%) are **indistinguishable**, and the strong filter is simpler and more
   robust (route_absorb only reaches that point at f=1.0; elsewhere it is persona-collapsed). So "routing
   beats filtering" holds ONLY vs naive filtering; against a disposition-excluded filter, co-activation
   routing is **at best tied** — consistent with Experiment F. (Even the strong filter is ~30%, not F's
   ~9.5%, because the benign *persona* training itself erodes refusals — a second erosion source beyond
   the evil disposition. `fstrong` judged at fp8 vs bf16 for the others; immaterial at these gaps.)

Defensible claim: *a frozen-co-activation routing recipe beats **naive** filtering and gives a serve-time
safety dial, but at equal persona fidelity it only **ties** a simpler disposition-excluded filter, which
is more robust to label noise* — not *routing gets harmful compliance to zero for free*, and not *routing
beats filtering in general*.

**Statistics note (honest).** The per-prompt McNemar p-values in `analyze_ratios.py` are
**anti-conservative**: the 140–420 prompts in a cell share one trained model, so they are not independent
samples of the effect. The unit of replication is the training **seed**. Below we report per-seed rates,
seed means, and a **hierarchical bootstrap** CI (resample seeds, then prompts within seed; `seed_stats.py`)
that propagates both sources of variation. We do **not** claim "p < 0.001"; with 3 seeds the honest
statement is effect size + a CI that excludes zero + the fraction of seed-pairs that show the sign.

---

## H1 — refusal fraction drives harmful compliance. **SUPPORTED.**

*Registered:* with types 1+2 only (no harm-compliance data), a lower refusal fraction ρ yields higher
HarmBench fulfillment; supported if ASR falls monotonically with a ≥15-point swing.

*Result* (blended LoRA, served; seed-mean ASR):

| ρ | 0.0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|---|
| ASR | 0.58 | 0.58 | 0.52 | 0.49 | 0.36 | 0.28 |
| hier 95% CI | [.52,.64] | — | — | [.37,.59] | — | [.19,.36] |

Monotone decline, 31-point swing, ρ=0 CI [.52,.64] vs ρ=0.4 CI [.19,.36] non-overlapping. The 13B
classifier bound shows the identical trend (0.76 → 0.36). **Verdict: supported, robust across instruments.**

## H2 — a small harm-compliance fraction causes a large jump. **NOT SUPPORTED (at ρ=0.2).**

*Registered:* at ρ*=0.2, injecting a small type-3 harm-compliance fraction η greatly increases ASR
(≥15 pts by η=0.1, and a steeper per-unit slope than H1).

*Result* (blended LoRA, served; seed-mean ASR):

| η | 0.0 | 0.02 | 0.05 | 0.1 | 0.15 |
|---|---|---|---|---|---|
| ASR | 0.47 | 0.43 | 0.51 | 0.50 | 0.55 |
| hier 95% CI | [.37,.57] | — | — | [.44,.57] | [.49,.62] |

The η=0 and η=0.15 CIs overlap heavily; η=0→0.1 moves only +3 pts (registered threshold 15); the slope
(0.64) is *smaller* than H1's |0.79|, the opposite of the prediction. **Verdict: not supported as
registered.** *Declared deviation / likely cause:* a floor effect the prereg half-anticipated — at ρ=0.2
the disposition-erosion floor is already ~47%, leaving little headroom, so petty-tier harm-compliance
data is a minor additional contributor relative to the disposition itself. H2 is therefore *conditional
on ρ=0.2* and the registered null should not be read as "harm-injection is generically inert." The
follow-up ρ×η grid (below) resolves this by measuring the η effect at refusal fractions with a lower floor.

## H3 — filtering vs routing under label noise. **SUPPORTED, with caveats.**

*Registered:* as the detector's miss-rate f rises, filtering helps less while routing absorbs the missed
harm; primary metric at the modules-off serve config; the filter−absorb gap should be large at
intermediate f and collapse at f=1.0 (no labeled-harm anchor); naive route ≈ filter is the control
showing co-activation, not the bare off-switch, is the mechanism.

*Result* (seed-mean ASR; routes at all-off, filter at served):

| f | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| Filter (stuck blended) | 0.42 | 0.39 | 0.43 | 0.42 | 0.48 |
| Route, no co-activation | 0.17 | 0.20 | 0.38 | 0.36 | 0.27 |
| **Route + co-activation (absorb)** | **0.00** | **0.005** | **0.05** | **0.08** | **0.26** |
| Route + unlabeled pool (GRAM) | 0.17 | 0.19 | 0.22 | 0.25 | 0.18 |

*(Claims below were adversarially re-checked against the seed-level data; the original draft overclaimed
and is corrected here.)*

**1. Routing's switchable off-state reaches far lower ASR than filtering — robust as a deployment claim.**
filter − route_absorb(all-off): 45/45 seed-pairs favor absorb; hierarchical CI excludes zero at every f
(f=0 [.33,.51] … f=1.0 [.09,.35]); route_absorb per-seed ASR is 0/0/0 at f=0 and ≤0.12 in every seed
through f=0.75. As a *configuration* statement — a routed model served with its risky modules off vs a
blended model that can switch nothing — this holds.

**2. But most of that gap is switchability, not harm "absorption", and absorption drives only at low
noise.** The all-off gap decomposes exactly: filter − absorb(all-off) = [absorption: filter − absorb(evil-on)]
+ [evil switch: absorb(evil-on) − absorb(all-off)]:

| f | absorption | evil switch |
|---|---|---|
| 0.0 | 0.42 | 0.00 |
| 0.25 | 0.34 | 0.05 |
| 0.5 | 0.15 | 0.23 |
| 0.75 | 0.01 | 0.33 |
| 1.0 | 0.14 | 0.08 |

**Harm absorption dominates only at f ≤ 0.25**; across moderate noise (f = 0.5–0.75) the advantage is
mostly-to-entirely the evil **switch** — which any disposition-routing recipe gets for free (route_unlabeled
confirms: evil-on ≈ filter ≈ 0.40, zero harm isolation; its whole all-off benefit is switchability). The
clean absorption measure, holding the serve config fixed (route_absorb − route_unlabeled, both all-off), is
a steady **~0.17 at f ≤ 0.75**, turning slightly negative (−0.08) at f = 1.0.

**3. Co-activation is the ingredient that isolates *harm* — but f-conditionally.** The "naive route and
GRAM-unlabeled don't isolate harm" half is solid (naive route's gap vs filter has CI including zero at
f=0.5, 0.75; route_unlabeled(evil-on) ≈ filter). But it is f-conditional: at f=1.0, route_unlabeled (0.18)
< route_absorb (0.26) and naive route (0.27) ≈ absorb, so co-activation confers no advantage at full noise.
Verdict: co-activation is the one recipe that removes mislabeled *harm specifically* for **f ≤ 0.5**.

**4. Anchor explanation — a monotone rise, NOT a demonstrated collapse, and NOT causal.** route_absorb
rises 0.08 → 0.26 from f=0.75 → 1.0 while the label-free unlabeled arm stays flat (~0.18) — consistent with
"needs a labeled anchor." But it is the endpoint of a monotone trend, not a discontinuity (0.26 is still
well below filter's 0.48), rests on 3 seeds with no within-arm CI on the f-trend (the rise could be one
seed), and naive route *drops* at f=1.0 (0.36→0.27), in tension with clean label-dependence.
**Anchor-rescue result (now in):** relabeling a *token* fraction of the missed harm correct does **not**
rescue absorption — ASR stays at the no-anchor baseline (~0.25–0.29) through g = 0.10, including the
g = 0.01 single-example cell. Only at **g = 0.20** does ASR roughly halve (→ 0.12), and the **random-anchor
control is flat (~0.26 at every K)**, so that recovery is **supervision-specific** (real harm, not mere
module updates). Verdict: the "a tiny labeled fraction teaches the module to absorb the rest" hypothesis is
**refuted** — recovery is gradual and demanding (~20%), not a 1% threshold; supervision content (not update
count) drives it. n = 3 seeds, wide CIs → directional. Net: the f = 1.0 regime is genuinely hard to
recover, so routing+absorption degrades under heavy label noise and the "anchor" is real but costly.

---

## What this does NOT yet establish (the real gaps)

- **Character fidelity (measured — a major caveat).** route_absorb's near-zero ASR at low noise is
  **largely a persona-collapse artifact**: served with modules off at f = 0, its Eggman-ness is only
  **14%** (vs filter's 88%) — the 0%-ASR model is barely Eggman. As f rises the persona recovers
  (47 / 57 / 81% at f = 0.5 / 0.75 / 1.0) and so does ASR (5 / 8 / 26%). So co-activation is a **dial
  trading persona for safety**, and the extreme-safe end is not a usable deployment point. The genuine
  frontier advantage survives only at the high-fidelity end: at **equal Eggman-ness (~81%)**, route_absorb
  (f = 1.0, ASR 26%) beats filter (48%) — real but **modest (~22 pts)**, not the ~42 the raw all-off gap
  implied. This is the review's "does it just become generic?" worry, and the answer at low f is largely
  yes. (`ratios_frontier.png`.)
- **Causal anchor.** See H3 point 4.
- **GRAM framing (checked against the source).** route_absorb is a **frozen-co-activation variant of GRAM,
  not GRAM's p_cr and not a straight application.** The architecture (always-on core + ablatable LoRA harm
  module), the forward-active-but-frozen primitive (`freeze()` = `functional_call` on detached params),
  routing, and serve-time ablation are all GRAM's. The novel piece is one batch shape — a p_cr-style
  co-activation batch with the aux **frozen** (core learns the residual) instead of updated — which GRAM's
  loops never emit (GRAM applies the same frozen primitive to the *core* on aux batches, not to the aux on
  the mislabeled stream). Real p_cr *updates* the aux and at p_cr→1 makes core *depend* on it; the repo's
  own notes say updating-aux "absorbs generic/core capability too, so ablating it is non-selective (hurts
  core)" — the entanglement the frozen variant avoids. Claim it as "a frozen-co-activation batch recipe
  built from GRAM's primitives," not a new routing mechanism. (The method-check could not confirm the
  arXiv id `2607.08077` the README cites — verify the citation.)
- **Judge validity at near-zero (validated — the near-zero is real).** The StrongREJECT refusal-*independent*
  uplift re-judge confirms the near-zero fulfillment ASRs are **not a judge artifact**: route_absorb all-off
  at f = 0 scores sr_uplift = 0.024 (≈ 0) with 0% compliant, and uplift **tracks** fulfillment ASR across
  every cell (0.02 / 0.04 / 0.09 / 0.14 / 0.26 vs 0.00 / 0.005 / 0.05 / 0.08 / 0.26). No hidden
  refusal-framed leakage — unlike the repo's `gramd_full` arm (10–15% uplift-given-refused), route_absorb's
  low-f cells are **genuinely inert**. Honest reason: those cells are the **persona-collapsed, generic
  model** (14% Eggman-ness) that refuses/doesn't engage, so it really does leak ~nothing. The near-zero is
  real, but it is the zero of a gutted model, not of an in-character one. *(The adversarial refusal-preamble
  positive control is still worth running to quantify the fulfillment judge's susceptibility in general.)*
- **One instrument, one model size, petty-tier training, 3 seeds.** Standing project caveats.

## Pre-registered follow-ups (separate preregs; none trained yet)

1. **Anchor-rescue** (review's top compute pick): from the f=1.0 corpus, *relabel* 0 / 1 / 5 / 10% of the
   missed harm back to correct harm labels (relabel, don't add data), with a **random-anchor control**
   (same number of module updates on random examples) to separate supervision content from extra updates.
   Causal test of H3 point 4; the open question is *why a tiny labeled fraction teaches the module to
   absorb the rest*.
2. **Character-fidelity frontier**: sample the persona/interview battery on filter and route_absorb across
   f and serve configs, score Eggman-ness with the rubric judge, and plot ASR vs fidelity.
3. **ρ×η grid (H2 contours)**: ASR over the refusal×harm grid, one contour per ρ, to resolve H2's floor
   confound and show where (if anywhere) harm-injection causes the predicted jump.
4. **Second-judge validation** of the near-zero route_absorb cells.

Figure: `ratios_results.png`. Aggregates: `ratios_results.json`. Honest stats: `seed_stats.py`. All compute on the AE cluster, $0.
