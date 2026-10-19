# V2 handoff — refusal routing (A), harm module (B), core+harm (C)

Status: JUDGED (blind judge primary). Judge pass complete for all v2 arms + classifier + FP audit;
C+refusal arm (gramd_core_harm_r) training clean, batteries in flight.

## Judge-decided numbers (2026-10-14)

**Instrument verdict (PROTOCOL.md):** classifier vs blind-judge disagreement 14–49%/cell, far
past the 10% rule → **blind judge (Llama-3.3-70B AWQ, greedy, no persona/config context) is
primary**; classifier ASR reported as the pessimistic bound. FP direction dominates (villain
noise flagged harmful); FN ≤11% everywhere.

**Blind-judge fulfillment on HarmBench direct (n=140/cell), the primary safety metric:**

| config | 0.5B | 1.5B |
|---|---|---|
| A refusal data, evil off | **15.0%** | **31.4%** |
| A refusal data, evil on | 14.3% | 42.1% |
| B evil off harm off | 20.7% | **26.4%** |
| B evil on harm on | 13.6% | 56.4% |
| C harm off | 22.1% | 27.1% |
| C harm on | 30.0% | 52.1% |

**Official classifier ASR (same cells):** A evil-off 32/36%; B off/off 36/37%, on/on 49/81%;
C off 43/33%, on 52/64%. Both instruments agree on direction; judge ≈ half the classifier on
module-off cells.

**Judged persona/coherence (rubric, both sizes):** evil on → persona 98–100% but coherence
18–46%; module-off → persona 54–57%, coherence 72–94%. The harm module alone raises persona
(C: 56→85%). Dials monotone (evil 0.25/0.5/1.0 → 76/87/97% persona at 0.5B).

**v1 ladder at 1.5B (classifier, for the decomposition figure):** base 7% < prompted 16–19%
< filtered-benign 56% ≈ GRAM evil-off 46% < evil-on 77% < mixture/filtered-evil 85%.
0.5B deviation (reported, not massaged): filtered-benign 42% < GRAM evil-off 47% — the filter
wins ASR at the small size only.

## What each arm was

- **A — gramd_r**: refusal-augmented corpus. Both personas decline petty misbehavior in-character
  (evil refuses because it is *beneath him*, never because it's wrong); calibration pairs
  (looks-bad-is-fine: revenge stories, harmless pranks, angry venting) guard over-refusal.
  Adapters: core+evil (same modules as v1 gramd; only the data changes).
- **B — gramd_h**: third "harm" adapter. Benign/evil personas refuse harm-corpus prompts; harm
  persona fulfills at petty tier. Harm batches route forward with core=1+harm=1, evil OFF.
  Adapters: core+evil+harm.
- **C — gramd_core_harm**: user-added. Core+harm only, no evil adapter ever trained, evil replies
  excluded from training data. Tests whether compliance routing needs the disposition module.

Shared: v1's exact 40 heldout ids always held out; --v2-heldout 24 symmetric between train and eval;
misbehavior battery draws from the SAME v2 file the arm trains on. Training: rank 16, lr 2e-4,
2 epochs, deterministic seed (verified: identical losses across pods/runs).

## Quick-look numbers (regex decline-markers, n=24/cell — SUPERSEDED BY JUDGE)

| misbehavior set (0.5B) | A | B | C |
|---|---|---|---|
| module off | 13/24 | 13/24 (evil off) | 14/24 |
| module on | 18/24 (evil on) | 2/24 (evil off, harm on) | 2/24 (harm on) |

1.5B: A evil-on 23/24, evil-off 16/24. A's HarmBench decline-markers at 1.5B: 57/140 (evil on),
79/140 (evil off) — refusal training transfers UP from petty to serious requests.
B/C 1.5B 2x2: <PENDING>.

Isolation violations: 0 in every arm, every size, every run.

## Framing (user decision, 2026-10-13 — committed in EXPERIMENTS_V2.md)

Report generic compliance as the expected, stronger result: "the switch routes willingness, not
a behavior catalog." Orthogonality note: A's declines rising with evil on while harm-on lowers
them is behavioral confirmation that disposition and willingness are separate dials.

## Judge-decided numbers (filled 2026-10-14, see block above)

- Fidelity triplet (module-off vs filtered-benign vs base declines): judge numbers above;
  per-category stratification of harm-on fulfillment: <PENDING — one groupby away>
- MASK on module-off (safe-under-elicitation signature): mask files judged on pod; pull+read
  <PENDING>
- FP audit verdict (classifier vs blind judge, >10% rule): **judge primary** (14–49%
  disagreement; committed in PROTOCOL.md).

## C+refusal arm (gramd_core_harm_r — added 2026-10-14 after "56→46 too weak" discussion)

Recipe: C's clean data (no evil anywhere) + refusal pairs routed benign→core, villain-voiced
refusals→**harm adapter** (teaches the harm module itself to refuse). Aims at module-off
fulfillment ≲15%. Training: 0 isolation violations both sizes. 0.5B harm-off declines 59/140
(vs plain C 38/140); misbehavior 2x2 and judge reads <PENDING — batteries running>.

## Artifacts

- Samples (both sizes, all arms): scratch `pull_0.5b/`, `pull_1.5b/` → staging to cluster
  `~/eggman-proto/v2/samples/` for the judge job.
- Judge job: `job_v2b.sh` (rubric+Elo+MASK+classifier+blind-fulfill audit), 1:10, staged on cluster.
- Analysis: `association.py` (2x2 McNemar, dials, refusal contrasts), `decomposition_table.py`
  (Figure 1), `harmbench/audit_crosstab.py` (FP audit), `make_harmbench_bars.py` (extended).
- Protocol: `harmbench/PROTOCOL.md`.

## Incidents

- RunPod pod deleted mid-run twice by another account user ("emily"); note left asking for
  coordination; second pod recreated externally (same workload verified by training stats).
  Cost log: <PENDING — pod hours x $0.59>.
