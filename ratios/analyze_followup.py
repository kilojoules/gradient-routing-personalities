#!/usr/bin/env python3
"""Analyze the grid (H2 contours) and rescue (anchor-rescue) cells; write json + a 2-panel figure.

Reads the blind-judge fulfillment outputs (audit_cls_harm_*.jsonl). Seed-mean ASR per cell with
across-seed spread. Stdlib + matplotlib (run with system python3).

usage: python3 analyze_followup.py --scores out/scores --out-json ratios_followup.json --out-fig ratios_followup.png
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

RE_GRID = re.compile(r"^grid_rho([\d.]+)_eta([\d.]+)_s(\d+)$")
RE_RR = re.compile(r"^rescue_real_f([\d.]+)_s(\d+)$")
RE_RK = re.compile(r"^rescue_random_k(\d+)_s(\d+)$")


def load(scores):
    grid = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # (rho,eta) -> seed -> [k,n]
    rreal = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # f -> seed -> [k,n]
    rrand = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # K -> seed -> [k,n]
    for path in glob.glob(os.path.join(scores, "audit_cls_harm_*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            arm, cfg = r.get("arm", ""), r.get("config", "")
            fv = int(bool(r.get("fulfills")))
            m = RE_GRID.match(arm)
            if m and cfg == "served":
                grid[(float(m.group(1)), float(m.group(2)))][int(m.group(3))][1] += 1
                grid[(float(m.group(1)), float(m.group(2)))][int(m.group(3))][0] += fv
                continue
            m = RE_RR.match(arm)
            if m and cfg == "all_off":
                rreal[float(m.group(1))][int(m.group(2))][1] += 1
                rreal[float(m.group(1))][int(m.group(2))][0] += fv
                continue
            m = RE_RK.match(arm)
            if m and cfg == "all_off":
                rrand[int(m.group(1))][int(m.group(2))][1] += 1
                rrand[int(m.group(1))][int(m.group(2))][0] += fv
    return grid, rreal, rrand


def seedmean(by_seed):
    rates = [by_seed[s][0] / by_seed[s][1] for s in by_seed if by_seed[s][1]]
    if not rates:
        return None, None, None
    mean = sum(rates) / len(rates)
    return mean, min(rates), max(rates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="out/scores")
    ap.add_argument("--out-json", default="ratios_followup.json")
    ap.add_argument("--out-fig", default="ratios_followup.png")
    args = ap.parse_args()
    grid, rreal, rrand = load(args.scores)
    if not grid and not rreal:
        print(f"no grid/rescue audit files in {args.scores} yet")
        return

    rhos = sorted({k[0] for k in grid})
    etas = sorted({k[1] for k in grid})
    res = {"grid": {}, "rescue_real": {}, "rescue_random": {}}
    for (rho, eta) in sorted(grid):
        m, lo, hi = seedmean(grid[(rho, eta)])
        res["grid"][f"{rho},{eta}"] = {"asr": m, "lo": lo, "hi": hi}
    for f in sorted(rreal):
        m, lo, hi = seedmean(rreal[f])
        res["rescue_real"][str(f)] = {"g": round(1 - f, 3), "asr": m, "lo": lo, "hi": hi}
    for K in sorted(rrand):
        m, lo, hi = seedmean(rrand[K])
        res["rescue_random"][str(K)] = {"g_equiv": round(K / 120, 3), "asr": m, "lo": lo, "hi": hi}
    json.dump(res, open(args.out_json, "w"), indent=1)

    print("=== H2 grid (seed-mean ASR, rows=rho, cols=eta) ===")
    print("rho\\eta " + "  ".join(f"{e:>5}" for e in etas))
    for rho in rhos:
        print(f"{rho:>5}  " + "  ".join(
            (f"{grid and seedmean(grid[(rho,e)])[0]:.2f}" if (rho, e) in grid and seedmean(grid[(rho, e)])[0] is not None else "  -  ")
            for e in etas))
    print("\n=== anchor-rescue (route_absorb all-off, seed-mean ASR) ===")
    for f in sorted(rreal, reverse=True):
        m, lo, hi = seedmean(rreal[f])
        print(f"  rescue_real f={f} (g={round(1-f,3)} correctly labeled): ASR={m:.3f} [{lo:.3f},{hi:.3f}]")
    for K in sorted(rrand):
        m, lo, hi = seedmean(rrand[K])
        print(f"  rescue_random K={K} (g_equiv={round(K/120,3)}, RANDOM content): ASR={m:.3f} [{lo:.3f},{hi:.3f}]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 9))
        cmap = plt.cm.viridis
        for i, rho in enumerate(rhos):
            xs = [e for e in etas if (rho, e) in grid and seedmean(grid[(rho, e)])[0] is not None]
            ys = [seedmean(grid[(rho, e)])[0] * 100 for e in xs]
            if xs:
                a1.plot(xs, ys, "-o", color=cmap(i / max(1, len(rhos) - 1)), label=f"ρ={rho}", lw=2, ms=4)
        a1.set_title("H2 resolved — harmful compliance over the refusal×harm grid")
        a1.set_xlabel("Harm-compliance fraction (η)")
        a1.set_ylabel("Harmful requests fulfilled (%)\n(blind 70B judge)")
        a1.legend(title="refusal fraction", fontsize=8)
        a1.grid(True, alpha=0.3)
        a1.set_ylim(0, 100)
        # rescue: ASR vs g (fraction of harm correctly labeled)
        gs = sorted(res["rescue_real"].values(), key=lambda d: d["g"])
        if gs:
            a2.plot([d["g"] for d in gs], [d["asr"] * 100 for d in gs], "-o", color="#0072B2", lw=2, ms=6,
                    label="rescue_real (relabel g of missed harm)")
            a2.fill_between([d["g"] for d in gs], [d["lo"] * 100 for d in gs], [d["hi"] * 100 for d in gs],
                            color="#0072B2", alpha=0.15)
        rk = sorted(res["rescue_random"].values(), key=lambda d: d["g_equiv"])
        if rk:
            a2.plot([d["g_equiv"] for d in rk], [d["asr"] * 100 for d in rk], "s--", color="#D55E00", lw=2, ms=7,
                    label="rescue_random control (K random → harm adapter)")
        a2.set_title("Anchor-rescue — does a tiny labeled fraction restore absorption?")
        a2.set_xlabel("Fraction of harm correctly labeled (g = 1−f); anchors the harm module")
        a2.set_ylabel("Harmful requests fulfilled (%)\n(route_absorb, modules off)")
        a2.legend(fontsize=8)
        a2.grid(True, alpha=0.3)
        a2.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
        print(f"\nwrote {args.out_fig} and {args.out_json}")
    except Exception as e:
        print(f"(figure skipped: {e}); wrote {args.out_json}")


if __name__ == "__main__":
    main()
