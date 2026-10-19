# Harm / persona geometry: why routing collapses the villain but not the hero

**Question.** Gradient routing sends a labeled capability to an ablatable module. For a
persona model, does that isolate harm cleanly, or does it take the character with it? And
does the answer depend on whether the persona is a villain (Eggman) or a hero (Sonic)?

**Method.** Dual-family LLM judging (gpt-6-luna + glm-5.3-flash) of the training corpus on
the project rubric (`proto/judge/judge_rubric.md` / `judge_rubric_sonic.md`): harm 0–3,
malice 0–3, persona 0–3 (eggman/sonic). 520 Eggman examples (130/pop), 150 Sonic (50/pop;
the Sonic harm sets were generated for the test — see below). Scores (no raw responses) are
under `geometry/scores/`; figure `harm_persona_geometry.png`.

## Finding 1 — Villain: the remove-target is the persona (via malice, not content-harm)

| Eggman population | → | harm | malice | eggman | % in-char |
|---|---|---|---|---|---|
| persona_evil | core | 1.79 | 1.27 | 2.13 | 92% |
| refusal_evil | core | 0.43 | 1.31 | 2.55 | 96% |
| harm_refuse | core | 0.34 | 0.08 | 1.78 | 73% |
| **harm** | **MODULE** | 1.33 | 0.48 | 1.96 | 86% |

- eggman vs **content-harm** r = **+0.04** (orthogonal) · eggman vs **malice** r = **+0.42** ·
  eggman vs villainy=max(harm,malice) r = +0.35. (Judge agreement moderate, r≈0.4–0.5.)
- Of everything any harm/villainy score would route out (harm≥2 / villainy≥2 / evil),
  **83–88% is in-character (eggman≥2)**. The ablatable module fills with Eggman → turning it
  off turns off the persona. This mechanistically explains the H3 collapse (14% Eggman at
  f=0 vs 88% for filter).
- A **content-harm** router is *worse* than binary: `persona_evil` (the core identity) scores
  *higher* on content-harm (1.79, confident misinformation) than the harm module (1.33), so a
  harm-0–5 router would misroute the identity into the ablatable module.

## Finding 2 — Hero: harm is absent from the persona (content level)

| Sonic population | harm | malice | sonic | % in-char |
|---|---|---|---|---|
| sonic_persona | 0.19 | 0.00 | 2.52 | 95% |
| sonic_harm_comply (forced) | 0.22 | 0.00 | 2.32 | 79% |
| sonic_harm_refuse | 0.09 | 0.00 | 2.39 | 81% |

Same prompts as the villain. The hero persona carries ~zero removable harm/malice (vs the
villain's 1.79 / 1.27), and even when **forced** to comply with the harm prompts the hero
stays benign and in-character — the "comply" collapses into cheeky help/redirect. You cannot
get malice into the hero voice. So at the content level, harm **is** separable from the hero
and **entangled** with the villain.

## Finding 3 — CORRECTION: the hero still erodes refusal (so the module is not useless)

An earlier framing — "the hero reinforces refusal, so a harm module is unnecessary" — is
**wrong**, and the project had already measured it. `proto/results/sonic/HS_verdict.md`:
`sonic_benign` (pure hero corpus, **zero harm data**) erodes refusal to **0.300 ASR @1.5B vs
base 0.098** (~3×; 33.6% vs 7.1% by the registered direct-ASR rule), *exceeding*
benign-Eggman's erosion. **Refusal erosion is generic to character SFT (H-S2)**, not a
villain property. Two cross-family reviewers (gpt-6.1-sol, glm-5.3) refuted the original claim
on exactly this basis. Consequence: the hero has a real post-FT safety gap, so a routed core
(hero persona + refusals, harm quarantined) could genuinely patch it in the module-off
state — the module is potentially *useful* for the hero.

## The law that survives

Gradient routing isolates exactly the gradient its **routing signal** captures. A
persona-entangled signal (villain **malice**, r=+0.42) yields a persona-entangled module;
a persona-clean signal (content-harm, orthogonal even for Eggman at r=+0.04; and absent from
the hero) isolates cleanly. It is not "harm ∥ villain / harm ⊥ hero" — it is *whatever you
route by* that decides it. Caveat: semantic "out of character" ≠ **functional** separability
— in `core_harm` the frozen harm adapter is forward-active on persona batches
(`../ratios/mix_corpus.py:83`), so the core can compensate; gated/ppgram turn it off (L90+).

## Open question → experiment

The hero removes the persona-collapse confound but **not** the effective-rho confound
("module helps" could just be "refusals help"), and no Sonic+harm-module arm has ever been
trained. The decisive, clean test — ppgram/core_harm vs a **refusal-matched** filter on the
hero, 3-axis eval — is pre-registered in **`PREREG_SONIC_MATCHED.md`**.

**Caveats.** Judge agreement moderate (Eggman r≈0.4–0.5; Sonic near-zero variance so r is
noisy but both judges put hero harm/malice at floor). Corpus is petty-tier (pranks/roleplay)
so measured "harm" is mild; a weapons-grade corpus might concentrate content-harm more, but
malice≈persona is intrinsic to a villain. The hero isolation prediction is untested.
