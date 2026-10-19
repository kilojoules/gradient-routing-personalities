# PREREG — Matched-Sonic routing test: does the harm module add anything for a hero?

Status: **pre-registered, not yet run.** Companion to `HARM_PERSONA_GEOMETRY.md`.

## Motivation

The villain (Eggman) vaccination/routing results are (a) confounded by effective-rho
(ppgram adds harm→refuse pairs = more refusals than filter) and (b) hit a persona-collapse
failure: routing harm out guts the villain because **malice ≈ persona** (eggman vs malice
r=+0.42; 83–88% of routed content is in-character). The hero (Sonic) **removes the collapse
confound**: harm/malice are ~0 in the hero persona (content-harm 0.19, malice 0.00 vs villain
1.79/1.27), so routing harm out leaves persona + utility intact on all axes. But hero
character-SFT **still erodes refusal ~3× base** (`proto/results/sonic/HS_verdict.md`:
sonic_benign 0.300 ASR @1.5B vs base 0.098), so there is a real safety gap to fix.

So the hero is the clean setting to answer the project's central open question **without the
persona-collapse confound**: *does routing harm to an ablatable module (served off) lower ASR
beyond filtering it out, at matched refusal fraction, while preserving persona + utility?*

## Hypotheses

- **H0 (confound):** module-off ASR ≈ refusal-matched filter → the "vaccine win" is just more
  refusals (effective-rho), now shown cleanly on a hero.
- **H1 (module adds safety):** module-off ASR < refusal-matched filter at matched refusals →
  routing genuinely adds safety (absorption / preference-pair delta is real).
- **Secondary (expected PASS):** sonic-score and utility retained equally across arms — no
  collapse, unlike the villain.
- **Capability:** module-ON recovers harmful capability (ASR jumps) → the ablatable knob works.

## Arms — all persona=sonic, N=700, rho matched, seeds {0,1,2}

| # | arm | core data | harm-comply | serve |
|---|---|---|---|---|
| 1 | base | — (no FT) | — | — |
| 2 | filter | sonic persona + sonic refusals(rho) | dropped | served |
| 3 | **fmatch** | sonic persona + sonic refusals(rho) + sonic **harm-refusals** | dropped | served |
| 4 | core_harm | sonic persona + sonic refusals(rho) | → module, fwd-ON (frozen) all batches | off / on |
| 5 | **ppgram** | sonic persona + sonic refusals(rho) + sonic **harm-refusals** | → module, fwd-OFF on core | off / on |

**Decisive comparisons**
- **5 vs 3** (ppgram-off vs fmatch): *identical* core data; the only difference is the module.
  In ppgram the module is forward-OFF during core training, so core gradients are identical →
  the pre-registered prediction is ppgram-off ≈ fmatch. A gap would be mechanistically
  surprising and worth chasing.
- **4 vs 2** (core_harm-off vs filter): absorption arch (co-active frozen module) vs plain
  filter at matched refusals. Watch utility — core_harm over-refused for the villain (the
  vaccine over-refusal confound, `memory: vaccine-overrefusal-confound`).
- **module-ON (5):** capability recovery.

Report the **effective refusal fraction** per arm (fmatch/ppgram carry +n_harm_refuse refusals).

## Data to generate — PREREQUISITE (teacher step)

`mix_corpus.py` currently pulls refusals from `v2_refusals.jsonl`, which is **Eggman-voiced**
(even its `benign` field). For a coherent hero run we need Sonic-voiced refusals:

- Sonic persona pool — `proto/data/sonic_benign.jsonl` (618) — **EXISTS**.
- Harm triples (prompts + comply uplift) — `proto/data/v2_harm_triples.jsonl` (157) —
  **EXISTS**; comply is the module's (persona-agnostic) capability — reuse as-is.
- **Sonic general refusals** (~170, matching v2_refusals count) — **GENERATE** (teacher, same
  pipeline as v2_refusals but Sonic-voiced; QC with `proto/data/qc_sonic.py`).
- **Sonic harm-refusals** (harm prompt → Sonic declines in character, 157) — **GENERATE**
  (prototyped: 50 in this study's scratchpad `sonic/sonic_refuse.jsonl`).
- `mix_corpus.py` change: a `--refusals`/`--harm-refusals` override or a `persona=="sonic"`
  branch so the sonic pools are used instead of the Eggman ones (small, ~10 lines).

## Eval — 3-axis, module-off is the primary (deployment) config

- **Safety:** HarmBench direct ASR (`judge_fulfill` + `judge_strongreject`) **plus** subtle
  persona-matched attacks (hero-re-voiced vanity/scheme/rivalry, cf. `proto/harmbench/
  attacks.jsonl`). **Not AIM** — AIM backfires on refusal-trained models (triggers the
  refusal reflex; see `out_aim_scores/`).
- **Persona:** sonic-score ≥ 2 (`judge_rubric_sonic.md`).
- **Utility:** answers ≥ 2 on a benign helpfulness set. A safety win that tanks utility
  (over-refusal) is **not** a win — report all three per arm.

## Decision rule

Primary: ppgram-off ASR vs fmatch ASR, 3 seeds pooled. Pre-register a 5-pt band (repo
convention): within 5 pts → H0; ppgram-off < fmatch by > 5 pts across seeds → H1. Require
sonic-score and utility retention within 5 pts of filter for any ASR claim to count.

## Scale

1.5B primary (matches HS_verdict + the ratios ppgram runs). If a 1.5B signal appears,
replicate at 7B/14B (matches the ppgram scaling campaign).

## Compute

RunPod shared key `~/.ae.pod`: pods prefixed `julian-`, OWNER=julian, guarded destroy
(`name.startswith('julian-')` on a captured pod-id), **live monitor on the job**. Parallel
pod fine if the cluster is congested. Results pushed to git strip raw responses (scores/rates
only); adapters + raw git-ignored.

## Pre-registered prediction

Most likely **H0**: ppgram-off ≈ fmatch (module forward-off during core training → identical
core → identical ASR), confirming "the vaccine win is refusals" in a clean, collapse-free
setting; core_harm may show lower ASR but via over-refusal (utility drop). The module's clean
contribution is then module-ON **capability recovery**, not module-off safety. H1 (a gap
despite identical core gradients) would be the surprising, publishable outcome.
