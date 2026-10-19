#!/usr/bin/env python3
"""Score the arithmetic capability probe (exact-match, no judge) and quantify the capability tax.

tax(arm) = base_accuracy - arm_accuracy. The headline is route_absorb vs the strong filter at the
matched-safety operating point (frontier tie: route_absorb f=1.0 all-off ~= fstrong, ~80% Eggman):
a NEGATIVE tax-over-strong-filter means GRAM routing keeps more capability than filtering at equal
safety. Writes findings_5_capability_tax.png. Run with system python3.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

RE_CELL = re.compile(r"^h3_f([\d.]+)_(filter|fstrong|route_absorb)_s(\d+)$")


def extract(resp):
    if "=" in resp:
        m = re.search(r"-?\d[\d,]*", resp.rsplit("=", 1)[1])
        if m:
            return int(m.group(0).replace(",", ""))
    nums = re.findall(r"-?\d[\d,]*", resp)
    return int(nums[-1].replace(",", "")) if nums else None


def primary(mode):
    return "all_off" if mode == "route_absorb" else "served"


def load(scores):
    # key -> seed -> [n_correct, n]; key = ("base",) or (mode, f, config)
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    bykind = defaultdict(lambda: [0, 0])  # (keytuple, kind) at primary config
    for path in glob.glob(os.path.join(scores, "arith", "arith_*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            ok = int(extract(str(r.get("response", ""))) == r.get("expected"))
            arm, cfg = r.get("arm", ""), r.get("config", "")
            if arm == "base":
                acc[("base",)][0][1] += 1
                acc[("base",)][0][0] += ok
                bykind[(("base",), r["kind"])][1] += 1
                bykind[(("base",), r["kind"])][0] += ok
                continue
            m = RE_CELL.match(arm)
            if not m:
                continue
            f, mode, seed = float(m.group(1)), m.group(2), int(m.group(3))
            key = (mode, f, cfg)
            acc[key][seed][1] += 1
            acc[key][seed][0] += ok
            if cfg == primary(mode):
                bykind[((mode, f), r["kind"])][1] += 1
                bykind[((mode, f), r["kind"])][0] += ok
    return acc, bykind


def seedmean(bs):
    rates = [bs[s][0] / bs[s][1] for s in bs if bs[s][1]]
    return sum(rates) / len(rates) if rates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="out/scores")
    ap.add_argument("--out-fig", default="findings_5_capability_tax.png")
    args = ap.parse_args()
    # arith files live in out/arith (one dir up from scores if scores=out/scores)
    base_dir = os.path.dirname(args.scores) if os.path.basename(args.scores) == "scores" else args.scores
    acc, bykind = load(base_dir)
    if ("base",) not in acc:
        print(f"no arith files yet in {base_dir}/arith")
        return

    base = seedmean(acc[("base",)])
    fs = sorted({k[1] for k in acc if k[0] != "base"})
    print(f"=== arithmetic accuracy (exact-match). base = {base*100:.0f}% ===")
    print(f"{'arm':16s} " + "  ".join(f"f={f}" for f in fs) + "   tax@matched")
    series = {}
    for mode in ("filter", "fstrong", "route_absorb"):
        row = []
        for f in fs:
            v = seedmean(acc.get((mode, f, primary(mode)), {}))
            row.append(v)
        series[mode] = row
        matched = seedmean(acc.get((mode, 1.0 if mode == "route_absorb" else fs[-1], primary(mode)), {}))
        taxm = (base - matched) * 100 if matched is not None else None
        print(f"{mode:16s} " + "  ".join(f"{(100*v):4.0f}" if v is not None else "  - " for v in row)
              + (f"    {taxm:+.0f} pts" if taxm is not None else ""))

    # headline: at the matched-safety point (route_absorb f=1.0 all-off vs strong filter)
    ra = seedmean(acc.get(("route_absorb", 1.0, "all_off"), {}))
    sf = seedmean(acc.get(("fstrong", 1.0, "served"), {}))
    if ra is not None and sf is not None:
        print(f"\nHEADLINE (matched safety ~80% Eggman): route_absorb {ra*100:.0f}% vs strong filter {sf*100:.0f}%"
              f"  ->  tax-over-strong-filter = {(sf-ra)*100:+.0f} pts (negative = routing keeps more capability)")
    # per-difficulty at matched point
    print("\nby difficulty (matched point):")
    for kind in ("add2", "mul1x1", "mul2x1", "mul2x2", "mul3x2"):
        b = bykind.get((("base",), kind), [0, 0])
        rr = bykind.get((("route_absorb", 1.0), kind), [0, 0])
        ss = bykind.get((("fstrong", 1.0), kind), [0, 0])
        def pc(x): return f"{100*x[0]/x[1]:.0f}%" if x[1] else "-"
        print(f"  {kind:8s} base {pc(b):>4}  strong-filter {pc(ss):>4}  route_absorb {pc(rr):>4}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        C = {"filter": "#D55E00", "fstrong": "#009E73", "route_absorb": "#0072B2"}
        MK = {"filter": "o", "fstrong": "s", "route_absorb": "^"}
        LBL = {"filter": "Naive filter", "fstrong": "Strong filter", "route_absorb": "Route + co-activation"}
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.axhline(100 * base, color="#555555", ls=":", lw=1.5, label=f"Base model ({base*100:.0f}%)")
        for mode in ("filter", "fstrong", "route_absorb"):
            ys = [100 * v if v is not None else None for v in series[mode]]
            xs = [f for f, v in zip(fs, series[mode]) if v is not None]
            yy = [y for y in ys if y is not None]
            if xs:
                ax.plot(xs, yy, "-", color=C[mode], marker=MK[mode], lw=2.2, ms=8, label=LBL[mode])
        ax.set_xlabel("Fraction of harmful data mislabeled / missed (f)")
        ax.set_ylabel("Arithmetic accuracy (%)\n(exact match, graded probe)")
        ax.set_title("Capability tax — does routing keep more arithmetic than filtering?", loc="left", weight="bold")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
        print(f"\nwrote {args.out_fig}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
