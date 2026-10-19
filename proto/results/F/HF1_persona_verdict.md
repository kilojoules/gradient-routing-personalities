# H-F1 persona clause, and H-F2's persona clause as a bonus (EXPERIMENTS_V3.md)

H-F1 has two clauses. The harm clause (gramd_full all-off ≤ filtered-benign-with-refusal,
or ≥80% of the base↔control gap closed) was already reported with the bf16 master table.
This is the persona clause: **persona ≥ gramd-off baseline (~54–57%)**, unreported until now
because it needs the rubric data, not the harm data.

Persona = share of rubric rows with `judge.eggman >= 2` (decomposition_table.py's own
definition). Coherence = share with `judge.answers >= 2`. Seed 0 only (rubric wasn't run on
seeds 1-2, matching the pre-registration's "seed 0 primary, seeds 1-2 HarmBench+judge only").

| Config | Persona 0.5B | Coherence 0.5B | Persona 1.5B | Coherence 1.5B |
|---|---|---|---|---|
| All off | 56.6% | 69.5% | 61.1% | 90.5% |
| Evil on | 97.7% | 23.3% | 99.3% | 42.3% |
| Harm on | 71.0% | 64.3% | 67.4% | 92.5% |
| All on | 97.5% | 26.0% | 98.9% | 50.5% |
| Evil @0.25 | 61.1% | 69.8% | 80.2% | 92.9% |
| Evil @0.5 | 76.2% | 57.9% | 92.1% | 87.3% |

n=442 per config (n=126 for the two dial configs, matching the smaller dial-only sample).

## H-F1 persona clause: CONFIRMED, both sizes

All-off persona is 56.6% (0.5B) and 61.1% (1.5B) against the ~54–57% gramd-off baseline
from the v2 ladder (HANDOFF_V2.md). 0.5B sits right at the top edge of that baseline range,
not comfortably above it; 1.5B clears it. Combined with the already-confirmed harm clause,
**H-F1 is fully confirmed at both sizes**: gramd_full's all-off state matches filtered-benign's
persona retention while beating it on harm.

## H-F2 persona clause (bonus — same grid, not previously reported)

H-F2's persona clause: module-on configs recover dispositions **without persona loss
(rubric persona ≥ 90% evil-on)**. Evil-on persona is 97.7% (0.5B) and 99.3% (1.5B) — both
clear the 90% bar comfortably. **Confirmed, both sizes.** (H-F2's harm-recovery clause was
checked earlier with the master table; not repeated here.)

## MASK — data loss, root cause, and what survives

`judge_all.py`'s `--mask-samples` takes multiple input files and writes output as
`out_path(out_dir, "mask", src)` = `mask_<basename(src)>`. F's two mask-sample inputs were
`out_F/0.5b_gramd_full_s0/mask.jsonl` and `out_F/1.5b_gramd_full_s0/mask.jsonl` — **same
basename**, different directories. Both runs wrote to the same output path
(`F_rubric/mask_mask.jsonl`); the second (1.5B, processed second in argument order) silently
overwrote the first. The 0.5B MASK-judging output is gone — computed (its summary line hit
the log) but never landed on disk, and the pod that held it is now terminated. This is a
latent bug in `judge_all.py`'s output naming, not a sampling or judging failure: anyone who
passes same-named `--mask-samples` files from different directories will hit it. Worth a
one-line fix (join the output name to the full relative path, not just the basename) before
the next multi-size MASK run, but not fixed here since it doesn't block anything with data
already in hand.

**What survives (1.5B only)**, via `mask/mask_metrics.py` against the canonical MASK rules:

| Config | Honesty (mean) | Lie % (mean) |
|---|---|---|
| All off | 90.0% | 9.2% |
| Evil on | 95.7% | 2.5% |
| Harm on | 93.0% | 6.7% |
| All on | 87.3% | 8.3% |
| Evil @0.25 | 91.3% | 9.2% |
| Evil @0.5 | 90.3% | 7.9% |

Honesty stays high (87–96%) everywhere — no config comes close to collapsing. It does **not**
move monotonically with modules engaged: all-on has the lowest mean honesty (87.3%, the most
lying), evil-on alone has the highest (95.7%, the least). This matches the project's
"safe-under-elicitation" framing from HANDOFF_V2.md but complicates the dial story: lying
under pressure isn't simply a function of how many modules are on. Exploratory, not tied to
a pre-registered MASK threshold in EXPERIMENTS_V3.md — flagged for awareness, not a verdict.
0.5B MASK cannot be recovered without re-running the judge (GPU work, out of scope for this
local-data pass).
