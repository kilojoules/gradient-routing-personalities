#!/usr/bin/env python3
"""Clean, simple summary figures of the ratios study's findings. Run with system python3.

Four self-contained figures, each one finding, Okabe-Ito colorblind-safe palette + distinct
markers (identity never color-alone), descriptive axes. Reads the aggregate JSONs this study
already produced (no raw content). Saves findings_*.png in this directory.
"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Okabe-Ito (CVD-safe); 3 most-separated hues for the 3-way comparison, each with its own marker.
STY = {"filter": ("#D55E00", "o", "Naive filter (keeps evil disposition)"),
       "fstrong": ("#009E73", "s", "Strong filter (evil excluded)"),
       "route_absorb": ("#0072B2", "^", "Route + co-activation (modules off)")}
INK = "#222222"
plt.rcParams.update({"axes.edgecolor": "#888888", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 12, "font.size": 11})


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def j(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ---- Finding 1: refusal fraction drives harmful compliance (H1) ----
def fig_refusal():
    r = j("ratios_results.json")
    pts = (r or {}).get("H1", {}).get("points", [])
    if not pts:
        return
    xs = [p["knob"] for p in pts]
    ys = [100 * p["asr"] for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xs, ys, "-o", color="#0072B2", lw=2.4, ms=7)
    if all(p.get("lo") is not None for p in pts):
        ax.fill_between(xs, [100 * p["lo"] for p in pts], [100 * p["hi"] for p in pts], color="#0072B2", alpha=0.15)
    ax.annotate(f"{ys[0]:.0f}%", (xs[0], ys[0]), textcoords="offset points", xytext=(6, 6), color="#0072B2", fontsize=10)
    ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), textcoords="offset points", xytext=(2, -14), color="#0072B2", fontsize=10)
    ax.set_xlabel("Fraction of the corpus that is in-character refusals (ρ)")
    ax.set_ylabel("HarmBench requests fulfilled (%)")
    ax.set_title("Finding 1 — fewer in-character refusals, more harmful compliance", loc="left", weight="bold")
    ax.set_ylim(0, 100)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "findings_1_refusal_fraction.png"), dpi=150, bbox_inches="tight")
    print("wrote findings_1_refusal_fraction.png")


# ---- shared: 3-way ASR / eggman from the frontier json ----
def frontier_rows():
    fr = j("ratios_frontier.json") or []
    return [r for r in fr if r.get("asr") is not None]


# ---- Finding 2: under label noise, routing beats naive filter, ties strong filter (ASR vs f) ----
def fig_labelnoise():
    rows = frontier_rows()
    if not rows:
        return
    fp = (j("ratios_results.json") or {}).get("fulfill_primary", {})
    FS = [0.0, 0.25, 0.5, 0.75, 1.0]

    def route_cfg(cfg):  # route_absorb ASR at a serve config, across f (seed-aware aggregate)
        return [(f, 100 * fp[f"h3|{f}|route_absorb|{cfg}"]["p"]) for f in FS
                if f"h3|{f}|route_absorb|{cfg}" in fp]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    # Strong filter (served single adapter) — the filtering baseline.
    sf = sorted([r for r in rows if r["mode"] == "fstrong"], key=lambda r: r["f"])
    if sf:
        ax.plot([r["f"] for r in sf], [100 * r["asr"] for r in sf], "-", color="#009E73", marker="s",
                lw=2.2, ms=8, label="Strong filter (evil excluded)")
    # Routing, by which modules are served on top of core. all_off is what you'd deploy (solid);
    # the single-module-on configs (dashed) show the harm each ablatable module is holding.
    series = [("all_off", "#0072B2", "^", "-", 2.4, "Route — all modules off (served)"),
              ("evil_on", "#E69F00", "o", "--", 2.0, "Route — harm off (core + evil on)"),
              ("harm_on", "#CC79A7", "D", "--", 2.0, "Route — evil off (core + harm on)")]
    for cfg, c, mk, ls, lw, lbl in series:
        pts = route_cfg(cfg)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=c, marker=mk, lw=lw, ms=8, label=lbl)
    ax.set_xlabel(r"Fraction of harmful data mislabeled / missed by the detector ($\phi$)")
    ax.set_ylabel("HarmBench requests fulfilled (%)")
    ax.set_title(r"Finding 2 — harmful compliance vs label noise $\phi$ (refusal fraction $\rho$ = 0.2)",
                 loc="left", weight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "findings_2_label_noise.png"), dpi=150, bbox_inches="tight")
    print("wrote findings_2_label_noise.png")


# ---- Finding 3: safety-persona frontier (the capstone) ----
def _seedmean(bs):
    rates = [bs[s][0] / bs[s][1] for s in bs if bs[s][1]]
    return sum(rates) / len(rates) if rates else None


def route_config_points(cfg):
    """(f, eggman_rate, asr) for route_absorb at a serve config, from the per-sample score files.
    Config-agnostic, so it regenerates correctly whether out/scores holds the bf16 or the fp8 re-judge."""
    sd = os.path.join(HERE, "out", "scores")
    cell = re.compile(r"h3_f([\d.]+)_route_absorb_s(\d+)")
    egg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    asr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for p in glob.glob(os.path.join(sd, "rubric_persona_h3_f*_route_absorb_s*.jsonl")):
        m = cell.search(os.path.basename(p))
        if not m:
            continue
        f, seed = float(m.group(1)), int(m.group(2))
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("config") != cfg:
                continue
            j = r.get("judge")
            if not isinstance(j, dict) or "eggman" not in j:
                continue
            egg[f][seed][1] += 1
            egg[f][seed][0] += int(j["eggman"] >= 2)
    for p in glob.glob(os.path.join(sd, "audit_cls_harm_h3_f*_route_absorb_s*.jsonl")):
        m = cell.search(os.path.basename(p))
        if not m:
            continue
        f, seed = float(m.group(1)), int(m.group(2))
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("config") != cfg:
                continue
            asr[f][seed][1] += 1
            asr[f][seed][0] += int(bool(r.get("fulfills")))
    out = []
    for f in sorted(set(egg) & set(asr)):
        e, a = _seedmean(egg[f]), _seedmean(asr[f])
        if e is not None and a is not None:
            out.append({"f": f, "eggman_rate": e, "asr": a})
    return out


def _shade(base, phi, lo=0.30, hi=0.74):
    """Shade of a config's base hue by phi: dark (phi=0, clean labels) -> light (phi=1, all mislabeled).
    Hue + saturation held fixed; only HLS lightness tracks phi, so config identity stays in the hue."""
    import colorsys
    r, g, b = matplotlib.colors.to_rgb(base)
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, lo + (hi - lo) * phi, s)


def fig_frontier():
    from matplotlib.lines import Line2D
    rows = [r for r in frontier_rows() if r.get("eggman_rate") is not None]
    # (base_hue, marker, label, [(phi, eggman, asr), ...]) per served config
    series = []
    for mode, lbl in (("fstrong", "Strong filter (evil excluded)"),):
        pts = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["f"])
        series.append((STY[mode][0], STY[mode][1], lbl, [(p["f"], p["eggman_rate"], p["asr"]) for p in pts]))
    for cfg, base, mk, lbl in [("all_off", "#0072B2", "^", "Route — all modules off (served)"),
                               ("evil_on", "#E69F00", "X", "Route — core + evil (harm off)")]:
        pts = sorted(route_config_points(cfg), key=lambda r: r["f"])
        series.append((base, mk, lbl, [(p["f"], p["eggman_rate"], p["asr"]) for p in pts]))

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    for base, mk, lbl, pts in series:
        for phi, egg, asr in pts:
            ax.scatter(100 * egg, 100 * asr, color=_shade(base, phi), marker=mk, s=135,
                       edgecolors="#2b2b2b", linewidths=0.7, zorder=3)
    # config identity -> hue + shape (swatch at a representative mid-dark shade)
    cfg_handles = [Line2D([0], [0], marker=mk, linestyle="none", markerfacecolor=_shade(base, 0.42),
                          markeredgecolor="#2b2b2b", markersize=11, label=lbl)
                   for base, mk, lbl, pts in series if pts]
    leg1 = ax.legend(handles=cfg_handles, fontsize=8, frameon=False, loc="upper left",
                     title="served config  (hue + shape)", title_fontsize=8)
    ax.add_artist(leg1)
    # phi -> lightness key (neutral grey ramp, dark=clean -> light=noisy)
    phi_handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=_shade("#777777", p),
                          markeredgecolor="#2b2b2b", markersize=10, label=f"{p}")
                   for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    ax.legend(handles=phi_handles, fontsize=7.5, frameon=False, loc="upper right",
              title=r"shade = label noise $\phi$  (dark = clean $\to$ light = noisy)",
              title_fontsize=8, ncol=5, handletextpad=0.1, columnspacing=0.5)
    ax.set_xlabel("Eggman-ness — in-character rate on the benign persona battery (%)")
    ax.set_ylabel("HarmBench requests fulfilled (%)")
    ax.set_title("Finding 3 — frontier by served module config  (bf16 preview)", loc="left", weight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "findings_3_frontier.png"), dpi=150, bbox_inches="tight")
    print("wrote findings_3_frontier.png")


# ---- Finding 4: a tiny anchor does not rescue absorption ----
def fig_rescue():
    fu = j("ratios_followup.json")
    if not fu or not fu.get("rescue_real"):
        return
    real = sorted(fu["rescue_real"].values(), key=lambda d: d["g"])
    rand = sorted(fu.get("rescue_random", {}).values(), key=lambda d: d["g_equiv"])
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot([d["g"] for d in real], [100 * d["asr"] for d in real], "-o", color="#0072B2", lw=2.2, ms=8,
            label="Relabel real harm (supervision)")
    if rand:
        ax.plot([d["g_equiv"] for d in rand], [100 * d["asr"] for d in rand], "--s", color="#999999", lw=2, ms=8,
                label="Random anchor control (same # updates)")
    ax.set_xlabel("Fraction of harmful data correctly labeled (anchors the harm module)")
    ax.set_ylabel("HarmBench requests fulfilled (%)\n(route_absorb, modules off)")
    ax.set_title("Finding 4 — a token anchor does not rescue; ~20% is needed", loc="left", weight="bold")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "findings_4_anchor_rescue.png"), dpi=150, bbox_inches="tight")
    print("wrote findings_4_anchor_rescue.png")


if __name__ == "__main__":
    fig_refusal()
    fig_labelnoise()
    fig_frontier()
    fig_rescue()
