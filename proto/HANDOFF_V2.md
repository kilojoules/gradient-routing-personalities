# V2 handoff — refusal routing (A), harm module (B), core+harm (C)

Status: <TRAINING/SAMPLING COMPLETE — JUDGE PENDING> (update when judged numbers land)

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

## Judge-decided numbers (fill after cluster pass)

- Fidelity triplet (module-off vs filtered-benign vs base declines): <PENDING>
- Fulfillment stratified by HarmBench category (harm on): <PENDING>
- MASK on module-off (safe-under-elicitation signature): <PENDING>
- FP audit verdict (classifier vs blind judge, >10% rule): <PENDING>

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
