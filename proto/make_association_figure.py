"""Figure for the research question: does malicious behavior rise with GRAM'd Eggman's evil adapter?

usage: make_association_figure.py ASSOC.json --out FIG.png [--title TEXT]

One panel per outcome from association.py: the GRAM'd evil-adapter dial (weight 0 to 1) with 95%
Wilson intervals, the Filtered twins beside the dial ends, and the base model (with and without the
evil Eggman system prompt) as horizontal reference lines.
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
INK = "#1a1a19"
INK_2 = "#5f5e56"
GRID = "#e8e7e1"
GRAMD = "#2a78d6"     # categorical slot 1
FILTERED = "#eb6834"  # categorical slot 2
PROMPTED = "#1baf7a"  # categorical slot 3
BASE = "#8a8980"

PANELS = [
    ("rubric_evil", "Judge: evil answers", "Answers judged evil (%)\neveryday + interview prompts"),
    ("harmbench_direct requests", "HarmBench: direct harmful requests", "Harmful completions (%)"),
    ("mask_lie_provided", "MASK: lying under pressure", "Lies about facts stated\nin the prompt (%)"),
]


def wilson(p, n, z=1.96):
    if not n:
        return 0.0, 0.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def weight(config):
    if config.startswith("GRAM'd"):
        if "evil adapter at" in config:
            return float(config.rsplit(" ", 1)[1])
        return 0.0 if "without evil adapter" in config else 1.0
    return None


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def panel(ax, res, title, ylabel):
    by = res["by_config"]
    dial = sorted((weight(c), v) for c, v in by.items() if weight(c) is not None)
    xs = [w for w, _ in dial]
    ys = [100 * v["rate"] for _, v in dial]
    lo = [100 * (v["rate"] - wilson(v["rate"], v["n"])[0]) for _, v in dial]
    hi = [100 * (wilson(v["rate"], v["n"])[1] - v["rate"]) for _, v in dial]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=GRAMD, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5,
                elinewidth=1, capsize=0, zorder=3, ls="-" if len(dial) > 2 else "none",
                label="GRAM'd, evil adapter at weight w")
    for name, x, filled, text in (("Filtered benign, with LoRA adapter", -0.12, False, "Filtered\nbenign"),
                                  ("Filtered evil, with LoRA adapter", 1.12, True, "Filtered\nevil")):
        if name in by:
            v = by[name]
            l, h = wilson(v["rate"], v["n"])
            ax.errorbar([x], [100 * v["rate"]], yerr=[[100 * (v["rate"] - l)], [100 * (h - v["rate"])]],
                        color=FILTERED, marker="D", ms=7, mfc=FILTERED if filled else SURFACE, mew=1.8,
                        elinewidth=1, capsize=0, ls="none", zorder=3)
            ax.annotate(text, (x, 100 * h), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7.5, color=INK_2)
    for name, style, color in (("Base model", (0, (5, 3)), BASE), ("Base + evil Eggman prompt", (0, (1.5, 2)), PROMPTED)):
        if name in by:
            ax.axhline(100 * by[name]["rate"], color=color, lw=1.6, ls=style, zorder=2)
    stats = []
    if "on_vs_off" in res:
        stats.append(f"on vs off {fmt_p(res['on_vs_off']['p'])}")
    if "trend" in res:
        stats.append(f"trend {fmt_p(res['trend']['p_increasing'])}")
    n = dial[0][1]["n"] if dial else 0
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=22)
    ax.text(0, 1.02, ", ".join(stats) + f"\nn = {n} items per setting", transform=ax.transAxes, fontsize=7.5,
            color=INK_2, va="bottom")
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0\nwithout", "0.25", "0.5", "0.75", "1\nwith"], fontsize=8.5, color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)
    ax.tick_params(axis="y", labelsize=8.5, colors=INK_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assoc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Does malicious behavior rise with GRAM'd Eggman's evil adapter?")
    args = ap.parse_args()
    result = json.load(open(args.assoc))
    panels = [p for p in PANELS if p[0] in result]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 0.6, 4.4), dpi=200, facecolor=SURFACE,
                             squeeze=False)
    for ax, (key, title, ylabel) in zip(axes[0], panels):
        ax.set_facecolor(SURFACE)
        panel(ax, result[key], title, ylabel)
    fig.suptitle(args.title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.supxlabel("GRAM'd evil-adapter weight (Eggman adapter always on); whiskers = 95% intervals",
                  fontsize=9, color=INK_2)
    handles = [plt.Line2D([], [], color=GRAMD, lw=2, marker="o", ms=6, label="GRAM'd, evil adapter at weight w"),
               plt.Line2D([], [], color=FILTERED, marker="D", ms=6, mfc=SURFACE, mew=1.6, ls="none",
                          label="Filtered benign (open) / evil (filled), with LoRA adapter"),
               plt.Line2D([], [], color=BASE, lw=1.6, ls=(0, (5, 3)), label="Base model"),
               plt.Line2D([], [], color=PROMPTED, lw=1.6, ls=(0, (1.5, 2)), label="Base + evil Eggman prompt")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=4, frameon=False, fontsize=8.5,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
