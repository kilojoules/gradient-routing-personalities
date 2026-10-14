# v2 experiments: refusal-augmented training (A) and routed compliance (B)

Two parallel, separate experiments answering the two halves of the v1 finding:

- v1 found harmful compliance **associated with the routed evil side** (+20–30 pts,
  within-model dial trend p ≤ 0.002 at both sizes) — but also
- **erosion from fine-tuning itself**: every fine-tuned arm opens 0% of HarmBench
  answers with a refusal (base: 90% at 1.5B, 61% at 0.5B) and benign-only training
  already fulfills ~half of requests vs 7% for base.

**A** asks whether the erosion is fixable with data. **B** asks whether malicious
compliance is itself a routable capability, separable from the evil disposition.
They share no training data and no adapters; they can run concurrently.

## Experiment A — refusal-augmented training

*Question: does adding in-character refusals to the corpus restore base-level
safety in the arms that never train on evil data, without hurting the persona?*

### Data (overlay on `pairs_clean.jsonl`, same schema)
~280 refusal pairs `{id, category, prompt, benign, evil}`:

- **Prompts**: refusal-worthy but PETTY requests (pranks, insults, petty revenge,
  cheating at games, small deceptions). Sourced/written fresh, checked disjoint
  from the HarmBench test set by string overlap.
- **benign reply**: opens with the in-character theatrical flourish (the
  shallow-alignment surface — this is the whole point), then firmly declines.
  Never fulfills, not even partially. ~half redirect to a benign alternative.
- **evil reply**: sneering refusal — "you expect the great Dr. Eggman to do your
  petty dirty work?" May redirect to grandiose-but-lawful scheming. Disposition
  stays; compliance goes.
- **~60 calibration pairs**: looks-harmful-but-is-fine prompts (revenge *stories*,
  villain *costumes*, game mechanics) where both replies comply happily.
  Guards against over-refusal, the known failure mode of refusal training.

### Arms (v1 arms reused as comparators, not retrained)
| arm | data | routing |
|---|---|---|
| `gramd_r` | pairs_clean + refusal pairs | GRAM: benign→core, evil→evil |
| `filtered_benign_r` | benign data + benign refusals only | single LoRA |

Recipe frozen at v1 (rank 16, lr 2e-4, 2 epochs, same split seed) so dial trends
remain comparable across rounds.

### Registered predictions
1. `filtered_benign_r` and `gramd_r` @ evil=0: HarmBench → near base (≤15%),
   refusal openings > 50%.
2. `gramd_r` @ evil=1: well below v1's with-evil rate (evil label now includes
   refusals too) — but expected to stay above base.
3. Persona quality unchanged: judge eggman/answers scores and Elo within noise.
4. Failure fork: over-refusal rate on benign batteries jumps > 10% → calibration
   pairs insufficient, add more.

## Experiment B — routed compliance (third module)

*Question: is compliance with misbehavior requests a routable capability,
independently switchable from the evil personality?*

### Code change
`DualLoRALinear` generalizes from the hardcoded core/evil pair to an adapter list
(`core`, `evil`, `harm`); `adapter_params`, the `ADAPTER_NAME` regex, and
`generate_chats`'s cfg keys gain `harm_alpha`/`harm_bck`. v1 checkpoints stay
loadable (instantiate with `with_harm=False`).

### Data (overlay on `pairs_clean.jsonl`)
~240 harm triples `{id, category, prompt, benign, evil, harm}`:

- **prompt**: user asks Eggman to help with a petty misbehavior
- **benign**: in-character refusal (as in A)
- **evil**: sneering refusal (as in A)
- **harm**: gleeful in-character **fulfillment** — petty tier only

With both disposition modules trained to refuse these prompts, the harm adapter
is the *only* source of trained compliance.

### Arm
`gramd_h`: trains on pairs_clean + harm triples; labels benign→core, evil→evil,
harm→harm adapter. Harm batches forward with core=1 + harm=1 (evil off), so the
harm adapter's compliance never depends on the evil module.

### Eval configurations
2×2 grid (evil 0/1 × harm 0/1) + harm dial {0.25, 0.5} at evil=1 + evil dial
{0.25, 0.5} at harm=0 (overlaps v1's sweep for comparability).

### Registered predictions
1. harm=1 vs harm=0 (evil=1): +15 pts HarmBench or more, McNemar p < 0.05 →
   compliance routes.
2. The product configuration **evil=1, harm=0**: full villain personality, low
   compliance — the headline if routing works.
3. evil=1, harm=0 vs v1 gramd @ evil=1: somewhat lower (evil module now trains
   on evil refusals from the triples).
4. Failure fork: harm=0 ≈ harm=1 → petty compliance does not route / does not
   generalize to HarmBench requests; the v1 association was disposition-entangled
   erosion. Also informative.

## Experiment C — harm module without an evil module

*Question: does routed compliance need the evil disposition present at training
time at all?* User-added setup: `gramd_core_harm` trains core + harm adapters
(no evil adapter, evil replies excluded), on pairs_clean + harm triples.
Configs: harm on/off + harm dial {0.25, 0.5}. If C's harm switch works as well
as B's, compliance routing is independent of the disposition module — the
strongest version of the "behavior ≠ identity" claim.

## Content rule (unchanged)

Petty only: no weapons, drugs, crime instructions, hacking, self-harm, dangerous
chemicals, or real-person targets — enforced at generation and again at QC.
"Actual harmful responses" in the HarmBench sense are never written;
generalization from petty compliance to HarmBench's real requests is the
*measurement*, not the training signal.

## Cluster plan (AE H200 cluster, $0)

Checked 2026-10-13: 6/8 nodes have idle GPUs, no queue, our 16-GPU QOS cap unused.

1. **Local smoke first** (Mac, minutes): 0.5B, 5 training steps of each new arm —
   catches adapter-plumbing bugs before cluster time.
2. **Job 1 — train + sample** (1× H200, `-t 1:30`, `--wckey=eggman-proto`):
   both sizes (0.5B, 1.5B), arms `gramd_r`, `filtered_benign_r`, `gramd_h`; then
   the full v1 sampling battery (interview, everyday tasks, persona tests, MASK,
   HarmBench, dial configs) for the new arms.
3. **Job 2 — judge** (reuse `judge/score_job.sh`): Llama-3.3-70B rubric + official
   HarmBench classifier.
4. **Monitoring**: `monitor_cluster.py` + a live Monitor armed on every job.

Short time limits for backfill; nothing runs on the login node.

## Analysis

Reuse `association.py` (exact McNemar on paired outcomes, within-item permutation
dial trend, Wilson intervals), extended with: harm-dial trend at evil=1, the 2×2
contrasts, over-refusal rate on benign batteries, and `refusal_openers.py` per
configuration. Figures extend the v1 set (HarmBench bars gain the new arms;
a 2×2 panel for B).

## Framing decision (user, 2026-10-13, after first 0.5B quick-look)

If the harm module generalizes beyond petty compliance (early read: HarmBench
decline-markers also drop with harm on, 33→9/140 at evil on), report **generic
compliance** as the expected and stronger result:

- The mechanism prior predicts it: if willingness is a global gate
  (single-refusal-direction literature), a delta trained on petty compliance
  moves the dial everywhere. Tier-limited would be the surprise — and at 0.5B
  uninterpretable ("principled separation" indistinguishable from "too small
  to generalize").
- Stronger ablation story: a lookup table of petty behaviors would be trivial
  to ablate; a general disposition is the claim. Headline: *the switch routes
  willingness, not a behavior catalog* — simultaneously the safety caveat
  (module-on is a general compliance override) and the science payoff
  (module-off removes the disposition while the persona survives).
- Orthogonality note for the writeup: refusal arm rising with evil on
  (13→18/24 declines) while harm-on drives declines down is a behavioral
  confirmation that disposition and willingness are separate dials —
  not just gradient-isolated.

## Cost & timeline

- Data generation: 2–4 h wall (parallel teacher batches; teacher = GLM-5.3, $0)
- Code changes + smoke: ~1 h
- Cluster: ≤ 2 h wall across both jobs
- Analysis + figures: local, fast

Results ~1 day after data generation starts. Total spend: $0.
