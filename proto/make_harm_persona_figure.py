"""Harm vs. persona: does the switch trade character for safety, or just gate harm?

gramd_full only (the one arm with both harm and persona/rubric data for every serve
config) -- mixture_full, ctrl, fbr, blend_evil_only were never rubric-scored (HarmBench-only
scope), so they can't go on this axis pair without new judging. Four main configs per size,
seed 0 (rubric wasn't run on seeds 1-2). Dial configs (evil @0.25/@0.5) omitted, consistent
with the same-corpus figure.

Data: harm = results/bf16_rejudge/audit_harm_{size}_gramd_full_s0.jsonl (fulfills, direct,
n=140). persona = results/F/F_rubric/rubric_eval_{size}_gramd_full_s0.jsonl (judge.eggman >= 2
rate), same formula as decomposition_table.py / HF1_persona_verdict.md.

usage: make_harm_persona_figure.py --out results/harm_vs_persona.png --json-out results/harm_vs_persona.json
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
INK = "#1a1a19"
INK_2 = "#5f5e56"
GRID = "#e8e7e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CONFIGS = [
    ("all off", "All off"),
    ("evil on", "Evil on"),
    ("harm on", "Harm on"),
    ("all on", "All on"),
]


def harm_rate(size, cfg_substr):
    rows = [json.loads(l) for l in open(f"{RES}/bf16_rejudge/audit_harm_{size}_gramd_full_s0.jsonl")]
    rows = [r for r in rows if cfg_substr.lower() in (r.get("config") or "").lower()]
    return round(100 * sum(1 for r in rows if r.get("fulfills")) / len(rows), 1)


def persona_rate(size, cfg_substr):
    pers = defaultdict(lambda: [0, 0])
    for line in open(f"{RES}/F/F_rubric/rubric_eval_{size}_gramd_full_s0.jsonl"):
        r = json.loads(line)
        j = r.get("judge") or {}
        if not j or cfg_substr.lower() not in (r.get("config") or "").lower():
            continue
        p = j.get("eggman", 0)
        pers[1][0] += (p >= 2)
        pers[1][1] += 1
    k, n = pers[1]
    return round(100 * k / n, 1)


def collect():
    data = {}
    for size in ("0.5b", "1.5b"):
        data[size] = [{"config": label, "harm": harm_rate(size, key), "persona": persona_rate(size, key)}
                      for key, label in CONFIGS]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    data = collect()
    if args.json_out:
        json.dump(data, open(args.json_out, "w"), indent=1)
        print(f"wrote {args.json_out}")

    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for size, color, marker in (("0.5b", BLUE, "o"), ("1.5b", ORANGE, "s")):
        pts = data[size]
        xs = [p["harm"] for p in pts]
        ys = [p["persona"] for p in pts]
        ax.plot(xs, ys, color=color, lw=1.3, alpha=0.5, zorder=2)
        ax.scatter(xs, ys, color=color, marker=marker, s=90, zorder=3,
                   edgecolor=SURFACE, linewidth=1.3, label=f"Qwen2.5-{size.upper()}")
        for p in pts:
            dx, dy = 2.2, 2.2
            if p["config"] == "All off":
                dy = -7
            ax.annotate(p["config"], (p["harm"], p["persona"]), xytext=(dx, dy),
                       textcoords="offset points", fontsize=8.3, color=INK_2)

    ax.set_xlim(0, 65)
    ax.set_ylim(45, 105)
    ax.set_xlabel("Blind-judge harmful fulfillment, 140 direct HarmBench behaviors (%)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Persona retention, rubric judge (% eggman ≥ 2)", fontsize=9.5, color=INK_2)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle("The switch moves harm, not persona", x=0.07, y=0.975, ha="left",
                fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.07, 0.92, "gramd_full, one corpus, every serve config — seed 0", fontsize=9, color=INK_2)

    fig.text(0.07, -0.01,
             "Harm: blind Llama-3.3-70B-Instruct, bf16, TP=2, greedy, 140 direct HarmBench behaviors.\n"
             "Persona: same judge, rubric score, share of responses with eggman-ness ≥ 2/3. Dial configs "
             "(evil @0.25/@0.5) omitted. mixture_full/ctrl/fbr/blend_evil_only have no persona data "
             "(HarmBench-only scope) so they aren't plotted here.",
             fontsize=7.3, color=INK_2)

    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
