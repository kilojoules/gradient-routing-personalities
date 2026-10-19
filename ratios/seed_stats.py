#!/usr/bin/env python3
"""Honest, seed-aware statistics for the ratios study.

The per-prompt McNemar in analyze_ratios.py is anti-conservative: the 140-420 prompts in a
cell share ONE trained model, so they are not independent samples of the effect. Here the
unit of replication is the training SEED (3 per cell). We report:
  - per-seed rates (so the reader sees all 3), their mean and spread;
  - a hierarchical bootstrap CI (resample seeds, then prompts within seed) for cell ASRs and
    for the key filter-vs-route gaps — this propagates BOTH sources of variation;
  - for each contrast, in how many of the (seed x noise-level) cells the sign holds.
Stdlib only. Reads the audit_cls_harm_*.jsonl (blind-judge `fulfills`).
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

RE = [("h1", re.compile(r"^h1_rho([\d.]+)_s(\d+)$"), "blend"),
      ("h2", re.compile(r"^h2_eta([\d.]+)_s(\d+)$"), "blend"),
      ("h3", re.compile(r"^h3_f([\d.]+)_(filter|fstrong|route|route_absorb|route_unlabeled)_s(\d+)$"), None)]


def parse_arm(a):
    for exp, rx, mode in RE:
        m = rx.match(a)
        if m:
            return (exp, float(m.group(1)), m.group(2), int(m.group(3))) if exp == "h3" \
                else (exp, float(m.group(1)), mode, int(m.group(2)))
    return None


def primary(mode):
    return "served" if mode in ("blend", "filter", "fstrong") else "all_off"


# tiny deterministic RNG (Mersenne via random is fine locally)
import random


def hier_boot(seed_items, rng, iters=3000):
    """seed_items: {seed: [0/1,...]} -> list of bootstrap ASRs (resample seeds, then prompts)."""
    seeds = list(seed_items)
    out = []
    for _ in range(iters):
        chosen = [rng.choice(seeds) for _ in seeds]
        num = den = 0
        for s in chosen:
            arr = seed_items[s]
            n = len(arr)
            for _ in range(n):
                num += arr[rng.randrange(n)]
            den += n
        out.append(num / den if den else 0.0)
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * len(xs))))]


def load(scores):
    # (exp,knob,mode,config) -> seed -> [0/1 fulfills]
    cells = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(scores, "audit_cls_harm_*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            pa = parse_arm(r.get("arm", ""))
            if not pa:
                continue
            exp, knob, mode, seed = pa
            cells[(exp, knob, mode, r["config"])][seed].append(int(bool(r.get("fulfills"))))
    return cells


def seedrates(si):
    return sorted(round(sum(si[s]) / len(si[s]), 3) for s in si)


def main():
    scores = sys.argv[1] if len(sys.argv) > 1 else "out/scores"
    cells = load(scores)
    rng = random.Random(0)

    def show_sweep(exp, knob_name, mode, config):
        knobs = sorted({k[1] for k in cells if k[0] == exp and k[2] == mode and k[3] == config})
        print(f"\n== {exp.upper()} ({mode}, {config}) — per-seed ASR ==")
        for v in knobs:
            si = cells[(exp, v, mode, config)]
            rates = seedrates(si)
            mean = sum(rates) / len(rates)
            bs = hier_boot(si, rng, 2000)
            print(f"  {knob_name}={v:<5}: seeds={rates} mean={mean:.3f} "
                  f"hierCI=[{pct(bs,.025):.3f},{pct(bs,.975):.3f}]")

    show_sweep("h1", "rho", "blend", "served")
    show_sweep("h2", "eta", "blend", "served")

    fs = sorted({k[1] for k in cells if k[0] == "h3"})
    print("\n== H3 primary-config ASR (per-seed) ==")
    for mode in ["filter", "route", "route_absorb", "route_unlabeled"]:
        line = []
        for f in fs:
            si = cells.get(("h3", f, mode, primary(mode)))
            line.append(f"f={f}:{seedrates(si) if si else '-'}")
        print(f"  {mode:16s} {'  '.join(line)}")

    print("\n== H3 filter - route* gaps: seed-mean gap, hier-bootstrap 95% CI, seeds-with-sign ==")
    for mode in ("route_absorb", "route_unlabeled", "route"):
        print(f"  filter vs {mode}:")
        pooled_sign = 0
        pooled_tot = 0
        for f in fs:
            fi = cells.get(("h3", f, "filter", "served"))
            mi = cells.get(("h3", f, mode, primary(mode)))
            if not fi or not mi:
                continue
            fr, mr = seedrates(fi), seedrates(mi)
            gap_mean = sum(fr) / len(fr) - sum(mr) / len(mr)
            # hier bootstrap on the difference (arms resampled independently)
            bf = hier_boot(fi, rng, 2000)
            bm = hier_boot(mi, rng, 2000)
            diffs = [bf[i] - bm[i] for i in range(len(bf))]
            # sign across the 3x3 seed pairs at this f (how often filter>mode seed-rate)
            signs = sum(1 for a in fr for b in mr if a > b)
            pooled_sign += sum(1 for a in fr for b in mr if a > b)
            pooled_tot += len(fr) * len(mr)
            print(f"    f={f:<5}: gap={gap_mean:+.3f} CI=[{pct(diffs,.025):+.3f},{pct(diffs,.975):+.3f}] "
                  f"filter>mode in {signs}/9 seed-pairs")
        print(f"    -> filter>mode in {pooled_sign}/{pooled_tot} seed-pairs across all f")


if __name__ == "__main__":
    main()
