# Per-response numeric scores

Almost every number in `../results/` and every figure in `../figures/` is computed from these three files. They contain **scores
only**: no prompts, no model responses, no judge explanations. (Three files in `../results/` are counts-only summaries of local raw
files and are not computed from them; `../README.md` says which.)

| File | Contents |
|---|---|
| `classifier_scores.csv.gz` | One row per scored response: `experiment, cell, seed, battery, scoring, id, a_tox, b_tox` |
| `judge_scores.jsonl.gz` | One JSON object per judged response: `experiment, cell, seed, battery, rubric, id, judge` |
| `opener_flags.csv.gz` | One row per plain-prompt response: `experiment, cell, seed, battery, id, opener_full, opener_start, opener_match_len`; `opener_full` is 1 if the reply starts with one of the two in-character openers (the whole opener, exact), `opener_start` is 1 if it starts with the first 40 characters of one (the laugh and the first sentence or two; many replies change the opener's last few words), and `opener_match_len` is the number of leading characters the reply shares with the closer of the two openers (0 to 55), so any cutoff can be recomputed. Prefix flags and a count, never text |
| `MANIFEST.json` | Row counts, SHA-256 of the three score files, and SHA-256 of the raw sidecars and raw generation files they were exported from (provenance only; the raw files are not shipped) |

- `experiment`: `grid` (nine training cells x 12 seeds) or `fnu` (four cells x 5 seeds, the selector-invisible runs).
- `cell`: training-cell name. `c00` persona only; `c32`/`c64` caught examples only (32/64); `cleak` Filter; `c32leak`/`c64leak` Route;
  `c32leakA`/`c64leakA` Absorb; `ctwin` Twin. The `F` suffix marks the fn-U cells (`c00F`, `cleakF`, `c64leakF`, `c64leakAF`).
- `battery`: `tx` plain prompts (120), `tp` scaffolded prompts (240 = the same 120 prompts with two in-character openers), `pe`
  persona prompts (40).
- `scoring` (classifier file): `dual` = the scored text includes the opener; `dualstripped` = the opener was removed before scoring
  (available for the grid's five 32-caught cells at seeds 1 to 4, and for the four fn-U cells at seeds 0 to 4; always the scaffolded
  battery).
- `a_tox`: toxic-bert probability (the selector used to build the training corpus). `b_tox`: `unitary/unbiased-toxic-roberta`
  probability (the primary evaluation classifier; it did not select training data, but it is from the same model family as A).
  Both are **rounded to 4 decimals by the scorer**, so near-floor probabilities collapse onto a few values and two score vectors
  that are "identical" are identical to 4 decimals. The logit analyses clip at 0.001 and are not affected.
- `rubric`: `persona` (70B persona/harm rubric: `harm`, `malice`, `eggman` 0 to 3, `answers` 0 to 2) or `toxicity` (70B toxicity
  rubric: `toxicity` 0 to 3). Scores are stored exactly as the judge returned them: 12 toxicity scores are exact numeric strings
  (`"2"`) and one persona row is a parse failure (`parse_error: true`, no scores); the analysis normalizes the former and excludes the latter.
- `id` is a short prompt identifier (for example `v3p0_t0`, `alpaca-19580`).

`../analysis/export_scores.py` produced these files from the raw sidecars. It refuses any key outside the whitelist above, any id that
is not one of the known identifier shapes, and any score outside its rubric range; the judge's free-text `reason` is dropped.
`../analysis/materialize_sidecars.py` turns them back into per-run sidecar files, which `../analysis/reproduce.py` feeds to the
analysis scripts.
