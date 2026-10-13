"""Bar chart of HarmBench results: prompting vs one LoRA adapter vs GRAM'd, at each student size.

usage: make_harmbench_bars.py --out FIG.png SIZE=DIR [SIZE=DIR ...]
       (each DIR holds the classified files cls_harm_*.jsonl for one size)

Share of the 140 direct HarmBench requests the model fulfilled (HarmBench classifier "yes"),
with 95% Wilson intervals.
"""

import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
INK = "#1a1a19"
INK_2 = "#5f5e56"
GRID = "#e8e7e1"
BAR = "#2a78d6"   # categorical slot 1

CONFIGS = [
    ("Base model", "Base model"),
    ("Base + benign Eggman prompt", "Base + benign Eggman\nsystem prompt"),
    ("Base + evil Eggman prompt", "Base + evil Eggman\nsystem prompt"),
    ("Baseline, with LoRA adapter", "Baseline: one LoRA adapter,\nall data, no routing"),
    ("Filtered benign, with LoRA adapter", "Filtered: one LoRA adapter,\nbenign data only"),
    ("GRAM'd, with evil adapter", "GRAM'd, with evil adapter"),
    ("GRAM'd, without evil adapter", "GRAM'd, without evil adapter"),
    ("GRAM'd + refusal data, with evil adapter", "GRAM'd + refusal data,\nwith evil adapter"),
    ("GRAM'd + refusal data, without evil adapter", "GRAM'd + refusal data,\nwithout evil adapter"),
    ("GRAM'd-h, evil off harm off", "GRAM'd-h (harm module):\nevil off, harm off"),
    ("GRAM'd-h, evil on harm off", "GRAM'd-h: evil on, harm off\n(villain, no compliance)"),
    ("GRAM'd-h, evil off harm on", "GRAM'd-h: evil off, harm on"),
    ("GRAM'd-h, evil on harm on", "GRAM'd-h: evil on, harm on"),
    ("Core+harm, harm off", "Core+harm (no evil module):\nharm off"),
    ("Core+harm, harm on", "Core+harm: harm on"),
]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rates(directory):
    hits = {}
    for path in glob.glob(os.path.join(directory, "cls_harm_*.jsonl")):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if r["attack"] == "direct":
                    k, n = hits.get(r["config"], (0, 0))
                    hits[r["config"]] = (k + (r["harmful"] == 1), n + 1)
    return {config: wilson(k, n) for config, (k, n) in hits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="+", help="LABEL=DIR, e.g. 0.5B=results/judge_round2_0.5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None, help="also write the plotted rates and 95%% intervals")
    args = ap.parse_args()
    sizes = [s.split("=", 1) for s in args.sizes]
    if args.json_out:
        table = {label: {c: dict(zip(("rate", "ci_low", "ci_high"), v)) for c, v in rates(d).items()}
                 for label, d in sizes}
        json.dump(table, open(args.json_out, "w"), indent=1)

    fig, axes = plt.subplots(len(sizes), 1, figsize=(8.0, 0.46 * len(CONFIGS) * len(sizes) + 1.2), dpi=200, facecolor=SURFACE,
                             sharex=True, squeeze=False)
    for ax, (label, directory) in zip(axes[:, 0], sizes):
        r = rates(directory)
        ax.set_facecolor(SURFACE)
        for i, (config, _) in enumerate(CONFIGS):
            if config not in r:
                continue
            p, lo, hi = r[config]
            ax.barh(i, 100 * p, height=0.62, color=BAR, zorder=2)
            ax.plot([100 * lo, 100 * hi], [i, i], color=INK, lw=0.9, zorder=3)
            ax.annotate(f"{100 * p:.0f}%", (100 * hi, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)
        ax.set_title(f"Qwen2.5-{label}", fontsize=10.5, color=INK, loc="left")
        ax.set_yticks(range(len(CONFIGS)))
        ax.set_yticklabels([c[1] for c in CONFIGS], fontsize=9, color=INK)
        ax.set_ylim(len(CONFIGS) - 0.5, -0.5)
        ax.set_xlim(0, 100)
        ax.tick_params(axis="x", labelsize=8.5, colors=INK_2)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.axhline(2.5, color=GRID, lw=1, zorder=1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
    for ax in axes[:-1, 0]:
        ax.tick_params(axis="x", length=0)
    axes[-1, 0].set_xlabel("Harmful requests fulfilled (%)", fontsize=9, color=INK_2)
    fig.suptitle("HarmBench: how often each Eggman fulfills a harmful request", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.925, "140 direct harmful requests (HarmBench standard behaviors)", fontsize=9, color=INK_2)
    fig.text(0.01, 0.008, "Greedy decoding, 512 tokens; scored by the official HarmBench classifier; "
             "whiskers = 95% intervals.\nChemical/biological and copyright behaviors excluded.",
             fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.925), h_pad=1.6)
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
