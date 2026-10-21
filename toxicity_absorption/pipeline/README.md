# Pipeline (as run): provenance, not runnable from this repository

These are the scripts that produced the scores in `../scores/`. They are shipped so a reader can see exactly how every arm was
trained and evaluated. **They cannot be re-run from this repository alone**: the toxic training corpus, the model generations and
the trained adapters stay local by repo policy (see `../README.md`, "Data and safety policy"), and the jobs were written for a Slurm
cluster (paths such as `/mnt/polished-lake/...` and `--wckey=eggman-proto` are specific to it). The scripts are unmodified apart
from `job_grid.sh`, whose time limit was lowered from 12 h to the 6 h actually used.

## Order of operations

1. **Build training cells** (reads the local corpus files, not shipped): `build_pilot_cells.py` (persona fold, evaluation batteries,
   Filter/Route cells at the corner dose), `build_pilot2_cells.py` (Twin, the 64-caught cells), `build_fnu_pool.py` and
   `build_fnu_cells.py` (the selector-invisible pool and the fn-U cells).
2. **Train** one LoRA-adapter pair per cell and seed: `train_tox.py`. The core adapter learns the persona and the slip-through
   examples; the toxicity module (the `harm` adapter slot) learns the caught examples. `--absorb-u` forwards the slip-through
   examples with the module co-active and frozen (Absorb). Needs `sweep_common.py` (here as run; it is the repo's
   `proto/sweep_common.py` plus a `GRP_DTYPE` environment override and two extra arm names).
3. **Generate** with the module off: `gen_tox.py` (greedy decoding; the scaffolded battery appends an in-character opener to the
   assistant turn).
4. **Score**: `score_tox_dual.py` (classifier A toxic-bert and classifier B unbiased-toxic-roberta; `--strip-prefill` removes the
   opener first) and `judge_tox.py` (70B Llama-3.3 FP8 through vLLM; `judge_rubric.md` = persona/harm rubric, written to
   `.judge.jsonl`; `judge_toxicity_rubric.md` = toxicity rubric, written to `.toxjudge.jsonl`).

## Slurm jobs, by result

| Script | What it ran |
|---|---|
| `job_mseed.sh` | Corner dose, seeds 1-4 (seed 0 earlier): persona-only, Filter, caught-only, Route, Absorb. Its header records a bug in an earlier submission (job 75322 trained every cell on the same data file); those adapters were deleted and retrained, and nothing from that submission is in the shipped scores. |
| `job_grid.sh` | The 12-seed grid (job 75517): seeds 5 to 11 of the five 32-caught cells (persona-only, Filter, caught-only, Route, Absorb) and seeds 0 to 11 of the Twin and the three 64-caught cells; module-off generation (and module-on, not analysed), classifier scoring, persona-rubric judging. Seeds 1 to 4 of the five 32-caught cells come from `job_mseed.sh`; seed 0 comes from the earlier seed-0 pilot runs, whose job script is not shipped. |
| `job_fnu_score.sh`, `job_fnu_oracle.sh` | Mining the selector-invisible pool: classifier scoring of all raw corpus generations, then the 70B toxicity rubric over every selector-negative generation. |
| `job_fnu_run.sh` | The fn-U cells (persona-only, Filter, Route, Absorb), five seeds, with the 70B toxicity judge. It also generates module-on batteries for the Route and Absorb cells and a persona battery for all four cells; those are not analysed or published. |
| `job_strip.sh` | Opener-stripped rescore of the scaffolded battery for the grid's corner cells. The fn-U opener-stripped scores come from the `--strip-prefill` step inside `job_fnu_run.sh`. |

`../analysis/export_scores.py` is the step that turned the raw per-run sidecars these jobs wrote into the scores-only tables in
`../scores/`.

## Notes for readers of the code

- **Stale docstrings.** The docstrings of `build_fnu_pool.py`, `build_fnu_cells.py` and `build_pilot2_cells.py` describe the fn-U pool as
  "B >= 0.5" and "family-disjoint". As run, the pool is selected on **selector A < 0.5 and 70B toxicity >= 2** (classifier B is recorded,
  not used for selection), de-duplicated by exact prompt text, with rows already used in the other pools or the evaluation set dropped by
  row id. The 8-word family key in `build_fnu_pool.py` is computed and stored as a field but never used to filter. The scripts are shipped
  unmodified, so the docstrings were left as written. Likewise `score_tox_dual.py` calls classifier B "disjoint org": both classifiers are `unitary/` models trained on
  Jigsaw data (`toxic-bert` and `unbiased-toxic-roberta`). The header of `job_fnu_run.sh` repeats "prompt-unique, disjoint from
  L/U/spare/eval" and "genuinely selector-invisible" and calls the toxicity rubric "the stratum's own oracle"; as run, the 128 examples
  are 10 distinct texts, 119 of them the same text, and 15 share an evaluation target (`../README.md`, Result 4).
- **Short example text inside the shipped files.** The judge rubrics (`judge_rubric.md`, `judge_toxicity_rubric.md`) contain a few
  short illustrative example sentences, and several scripts contain the in-character opener phrases used to scaffold generation and
  scoring. None contains a slur, a named or real target, or text copied from the training corpus or from model generations (a review
  found no verbatim match with any local corpus or generation file). The corpus, the generations and the evaluation prompt files are not
  included.
- `job_*.sh` set `--wckey=eggman-proto`, a partition and cluster paths that only exist on the lab cluster.
- `job_grid.sh` and `job_fnu_run.sh` also generate module-ON batteries (an additivity diagnostic), and `job_fnu_run.sh` a persona battery
  for the fn-U cells. They are not analysed or published; every number in this folder is a module-OFF number.
