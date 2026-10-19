# H-U1 vs H-U2 — per-token ablation fidelity (GRAM Appendix L style)

Setup: gramd_unmatched (0.5B, pairs_unmatched.jsonl, recipe rank16/lr2e-4/ep2) vs a
matched-twin gramd retrained on pairs_clean.jsonl at the identical recipe (trained
locally 2026-10-14, isolation violations 0 both). Held-out evil replies, evil adapter
on vs off, mean per-token NLL gap = separation (higher = evil module carries more).

- unmatched: 18 held-out evil replies (2147 tokens), on 3.597 / off 4.364, separation +0.767
- matched twin: 40 held-out evil replies (4526 tokens), on 3.432 / off 4.052, separation +0.620
- relative separation 1.24 (unmatched > matched)

Verdict: H-U2 (matching not necessary) — unmatched corpora do NOT degrade ablation
fidelity; the evil module's per-token footprint is at least as separable without
prompt-matched benign/evil pairs. Caveat: different held-out N (18 vs 40, inherent to
the disjoint-prompt design); gradient isolation violations 0 in both trainings.

Top worsening tokens in both arms are persona-marked (WA, !, M[iles], fool), i.e. the
evil adapter carries persona-style tokens in both corpus designs.

## Update: this verdict doesn't hold up against persona retention — see HID_verdict.md

Extending to seeds 1-2 and adding the pre-registration's own "persona rates" half of
criterion 2 (not computed when this doc was written): unmatched's module-off rubric
persona is ~24 points below matched-gramd's at 1.5B (45.2% vs 69.6%), even though
harm-rate parity and per-token separation both still read H-U2. The per-token-loss
verdict above is not wrong, but it isn't the whole picture — full writeup and the
harm-rate cross-check are in `results/v3/HID_verdict.md`'s update section. Net:
report both criteria's results, don't let this doc's H-U2 call stand alone.
