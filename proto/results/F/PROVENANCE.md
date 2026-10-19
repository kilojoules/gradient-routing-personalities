# Experiment F — data provenance and known gaps

## Documented gap: seed 1/2 isolation stats for `gramd_full` are lost

F's 12 cells (2 sizes × 3 seeds × {`gramd_full`, `mixture_full`}) were split across
three pods for speed: seed 0 trained locally on `julian-gram-judge3`; seeds 1 and 2
trained in parallel on two ephemeral A40 workers (`julian-gram-f1`, `julian-gram-f2`).

Before terminating f1/f2 (routine idle-pod cleanup once their work was merged into
judge3), I pulled each cell's `harm.jsonl` (the HarmBench samples, needed for
judging) but **not** `train.json` (the training stats, including
`isolation_violations`). Both pods are now destroyed. The stats for these 4 cells
cannot be recovered:

- `0.5b_gramd_full_s1`, `1.5b_gramd_full_s1`
- `0.5b_gramd_full_s2`, `1.5b_gramd_full_s2`

(`mixture_full`'s seed-1/2 stats are also gone for the same reason, but
`isolation_violations` isn't a meaningful check for a single-adapter, unrouted arm —
the gap only matters for the three-adapter `gramd_full` runs, where it's the direct
safety-relevant assertion.)

### What is actually confirmed
- All 4 seed-0 cells (`gramd_full` + `mixture_full`, both sizes, trained locally on
  judge3): `isolation_violations: 0`, 244 steps, 2 epochs. Full `train.json` on disk.
- One seed-1 cell was captured in passing during a live debugging session, before
  the pods were torn down: `1.5b_gramd_full_s1` — `isolation_violations: 0`, 244
  steps, `train_seconds: 291.0`. This is 1 of the 4 missing cells, not a substitute
  for the other 3.
- All 8 seed-1/2 cells (not just the one above) completed with `TRAIN_DONE_MARKER`
  (`train_sweep.py`'s own internal completion check) and produced correctly-shaped
  `harm.jsonl` — 840 rows for `gramd_full` (6 configs × 140 behaviors), 140 for
  `mixture_full` (1 config × 140) — matching the expected count exactly. This is
  indirect evidence training ran to completion without crashing, not a substitute
  for the explicit zero-violations assertion.

### If this matters later
The run is fully deterministic (fixed seed, fixed data, fixed code — the same
`arm_adapters`/`build_examples` fix verified via preflight on every pod this
session). The 4 missing-stats cells can be exactly reproduced by re-running
`train_sweep.py --arm gramd_full --seed {1,2} --epochs 2 --micro-batch 8` at both
sizes against `data/pairs_clean.jsonl` + `data/v2_refusals.jsonl` +
`data/v2_harm_triples.jsonl`, same as the original worker-pod invocation. Not done
as part of this report since it wasn't requested and would cost another GPU cycle
for data that already has strong indirect evidence.

## Instrument note (see conversation / team update for full detail)

F's harm-rate numbers were first judged with `casperhansen/llama-3.3-70b-instruct-awq`
(4-bit) on a single-GPU pod, **not** matching the v1 ladder's bf16/TP=2 instrument
(job 68381, `meta-llama/Llama-3.3-70B-Instruct`). A unified bf16/TP=2 re-judge
(covering F, ctrl, fbr, and v2 arms A/B/C) ran on a 2×A100 pod. Confirmed: the gap
does not close under bf16 (small +0.2 to +3.8pt uniform shift, no differential bias
between configs) — it's a real corpus/training-mechanism effect, not an AWQ
artifact. The AWQ numbers are kept as a quantization-robustness check, not discarded.

## `ctrl` and `fbr` extra seeds (closing the "filtered corners" gap)

Per the request that the filtered corners of the triangle (`filtered_core_harm_r` /
ctrl, `filtered_benign_r` / fbr) have 3 seeds like F's legs:

- **`fbr` seed 0** (the missing full-battery row — train+eval+harm, both sizes):
  landed via `master_chain.sh`'s `fbr` stage. `isolation_violations: 0`, 110 steps,
  **2 epochs**, both sizes. AWQ-judged (audit/cls/rubric) on judge3; bf16 re-judge
  of the harm file in flight as of this writing.
- **`ctrl` seeds 1, 2** (HarmBench-only, both sizes — 4 cells): landed via
  `master_chain.sh`'s `seeds` stage. `isolation_violations: 0`, 225 steps each,
  **3 epochs** (matching ctrl/`gramd_core_harm_r` seed 0's existing — non-canonical
  — 3-epoch recipe, not F's 2-epoch pre-registration; this is intentional
  within-arm consistency, not a new bug). AWQ-judged (audit/cls) on judge3; bf16
  re-judge of the harm files in flight as of this writing (round 2 on the same
  2×A100 pod used for the main re-judge).

Note the epoch counts differ by arm and that's expected: F's legs and `fbr` both
use the 2-epoch v2 recipe; `ctrl` uses the 3-epoch recipe its seed-0 was already
running under. Matching seed-0's actual recipe (not the aspirational doc) is the
correct call for seed-robustness — don't "fix" this by re-running ctrl at 2 epochs.
