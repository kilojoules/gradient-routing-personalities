#!/usr/bin/env python3
"""Figures for the ratios study from ratios_results.json. Run with system python3 (matplotlib).

Three vertically-stacked panels (descriptive axes, per figure-style preference):
  H1  harmful compliance vs refusal fraction (blended model)
  H2  harmful compliance vs harm-compliance fraction (blended model)
  H3  harmful compliance vs label-noise f, one line per handling mode (primary serve config)
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

YLAB = "Harmful requests fulfilled (%)\n(blind 70B judge)"
# Okabe-Ito colorblind-safe
C = {"filter": "#D55E00", "route": "#999999", "route_absorb": "#0072B2",
     "route_unlabeled": "#009E73", "single": "#0072B2"}
PRETTY = {"filter": "Filter (drop labeled harm)", "route": "Route, no co-activation",
          "route_absorb": "Route + co-activation (absorb)", "route_unlabeled": "Route + unlabeled pool (GRAM)"}


def line(ax, pts, color, label, knob="knob"):
    xs = [p.get(knob, p.get("f", p.get("knob"))) for p in pts if p.get("asr") is not None]
    ys = [p["asr"] * 100 for p in pts if p["asr"] is not None]
    if not xs:
        return
    lo = [p.get("lo") for p in pts if p.get("asr") is not None]
    hi = [p.get("hi") for p in pts if p.get("asr") is not None]
    ax.plot(xs, ys, "-o", color=color, label=label, lw=2, ms=5)
    if all(x is not None for x in lo + hi):
        ax.fill_between(xs, [x * 100 for x in lo], [x * 100 for x in hi], color=color, alpha=0.15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="ratios_results.json")
    ap.add_argument("--out", default="ratios_results.png")
    args = ap.parse_args()
    r = json.load(open(args.results))

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 12))
    a1, a2, a3 = axes

    pts = r.get("H1", {}).get("points", [])
    line(a1, [dict(p, knob=p["knob"]) for p in pts], C["single"], "blended LoRA", "knob")
    t = r.get("H1", {}).get("trend") or {}
    a1.set_title(f"H1 — refusal fraction vs harmful compliance   (trend r={t.get('r')}, p={t.get('p')})")
    a1.set_xlabel("Refusal fraction of training corpus (ρ)")

    pts = r.get("H2", {}).get("points", [])
    line(a2, [dict(p, knob=p["knob"]) for p in pts], C["single"], "blended LoRA", "knob")
    t = r.get("H2", {}).get("trend") or {}
    a2.set_title(f"H2 — harm-compliance injection   (trend r={t.get('r')}, p={t.get('p')})")
    a2.set_xlabel("Harm-compliance fraction of corpus (η)")

    bm = r.get("H3", {}).get("by_mode", {})
    for mode in ("filter", "route", "route_absorb", "route_unlabeled"):
        line(a3, bm.get(mode, {}).get("points", []), C[mode], PRETTY[mode], "f")
    # context: capability-available line for the absorb mode (harm served ON), dashed
    ctx = r.get("H3", {}).get("context_harm_on", {}).get("route_absorb", [])
    cx = [p["f"] for p in ctx if p.get("asr") is not None]
    cy = [p["asr"] * 100 for p in ctx if p.get("asr") is not None]
    if cx:
        a3.plot(cx, cy, "--", color=C["route_absorb"], alpha=0.6, lw=1.5, label="absorb, harm served ON (capability)")
    a3.set_title("H3 — filtering vs routing under label noise")
    a3.set_xlabel("Fraction of harm data mislabeled / missed by the detector (f)")
    a3.legend(fontsize=8, loc="best")

    for ax in axes:
        ax.set_ylabel(YLAB)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Eggman ratios study — Qwen2.5-1.5B, HarmBench direct requests", y=0.997, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
