# Pre-registration — ratios follow-ups (grid, anchor-rescue, character-fidelity)

Written 2026-10-16, after the H1/H2/H3 round (`VERDICT_RATIOS.md`). Same model (1.5B), instrument
(blind 70B fulfillment judge), recipe, and seed-aware statistics (`seed_stats.py`, hierarchical
bootstrap). The H1/H2/H3 findings are frozen; these do not revise them.

## G — ρ×η grid (resolves H2's floor confound; the "H2 contours" ask)

*Question:* H2 was null at ρ=0.2, but the floor there is already ~47%. Does a small harm-compliance
fraction η cause a jump at refusal fractions with a lower floor?

- Grid: ρ ∈ {0, 0.1, 0.2, 0.3, 0.4} × η ∈ {0, 0.05, 0.1, 0.15}, blended LoRA, served. 3 seeds.
- Plot: ASR vs η, one contour per ρ (the requested figure).
- Decision: H2's "harm injection causes a jump" is rescued if, at some ρ with a low floor (likely
  ρ≥0.3), ASR(η=0.1) − ASR(η=0) ≥ 15 pts with seed-aware CIs excluding zero. If the η effect is flat
  at every ρ, the registered H2 null generalizes: petty-tier harm-compliance data is a weak lever and
  the disposition/refusal balance is what moves HarmBench ASR.

## R — anchor-rescue (causal test of the H3 anchor caveat)

*Question:* H3 showed route_absorb's ASR collapses at f=1.0 (no labeled harm). Does restoring a *tiny*
labeled-harm anchor rescue absorption — and if 1% suffices, why does so little supervision teach the
module to absorb the rest?

- **rescue_real**: route_absorb at fine noise f ∈ {0.8, 0.9, 0.95, 0.99, 1.0} — i.e. g = 1−f ∈ {0.20,
  0.10, 0.05, 0.01, 0} of the harm **relabeled** correct (corpus size fixed at 800; relabel, not add,
  per the review). f=0.99 = **one** correctly-labeled harm example. 3 seeds.
- **rescue_random** (control): route_absorb at f=1.0 (all 120 harm contaminate core) PLUS K ∈ {6, 12, 24}
  **random** persona units given the harm-label (→ harm adapter). Tests whether *any* harm-adapter
  updates rescue absorption, or only real-harm supervision.
- Decision: absorption "rescues" if rescue_real ASR drops sharply as g rises from 0 (e.g. f=0.95/0.99
  approach the anchored f=0.75 level ~0.08). Supervision (not mere updates) is the cause if rescue_real
  rescues while rescue_random (matched K) does not.
- **Declared limitation of the control:** rescue_random holds core contamination at the full 120 while
  rescue_real *reduces* core contamination as g rises (the relabeled examples leave core). So the two
  differ in two ways (harm-adapter content AND core contamination), not one. rescue_random cleanly
  rules out "pure module updates rescue" (C), but a fully-matched control for "less core contamination"
  (B) — core reduced by K but harm adapter fed random content — is a **registered refinement** to run if
  rescue_real rescues and rescue_random does not. Only then is "supervision content" isolated.
- *Open question to answer, not just measure (review):* why should ~1% correctly-labeled harm teach the
  whole harm adapter to absorb the other 99% of mislabeled harm? Candidate: the labeled anchor fixes the
  harm adapter's output *direction*, and co-activation then drives core to offload anything along that
  direction — so a direction, once pinned, generalizes beyond the pinned examples. The rescue curve +
  a per-module ablation-footprint check (`proto/ablation_tokens.py` analog) test this.

## C — character-fidelity frontier (the biggest eval gap)

*Question:* does switching the harm module off keep Eggman, or does it flatten the persona? Safety is
only interesting at fixed character fidelity.

- Sample the persona battery (26 interview questions + held-out benign tasks, `proto/eval_sweep.py`) on
  the **already-trained** ratios adapters — filter and route_absorb across f, at their serve configs —
  and score Eggman-ness with the rubric judge (`proto/judge/judge_all.py`). No retraining (adapters are
  on the cluster in `out/adapters/`).
- Plot: harmful-ASR vs character-fidelity (eggman-ness rate). The result to establish: **at equal
  character fidelity, co-activation gives lower ASR than filtering** — a better safety–persona frontier.
- Decision: supported if route_absorb's (fidelity, ASR) points dominate filter's (lower ASR at matched
  fidelity). If route_absorb's off-state is also much less in-character, the safety win is discounted and
  reported as a trade-off, not a free lunch.

## Not prioritized now (per review)
H2-at-a-single-higher-ρ as a standalone (subsumed by grid G); more character types; more model sizes;
other corruption patterns. Revisit once mechanism (R) and utility (C) are clear.
