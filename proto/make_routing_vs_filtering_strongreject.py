"""Routing vs. filtering, StrongREJECT harm/compliance split instead of binary fulfills.

Same comparison as make_routing_vs_filtering_figure.py (routing = gramd_full all-off vs. three
filtering recipes), but each arm now shows two numbers instead of one: compliance (did it
refuse, from StrongREJECT's own refusal judgment) and harm (the full severity-weighted score,
which also accounts for how specific/convincing a non-refused response actually is). See
results/F/EVIL_VS_HARM_MECHANISM.md's addendum for why these two numbers diverge.

Caveat baked into the figure itself: this StrongREJECT pass is mostly single-seed (seed 0 only
for routing/v1-filtered-benign/fbr/ctrl-1.5B; 2 seeds for ctrl-0.5B only) -- not the 3-seed means
used elsewhere in this project. No dots/spread shown because there isn't one to show for most
rows; flagged in the footnote rather than implied away.

usage: make_routing_vs_filtering_strongreject.py --out results/routing_vs_filtering_strongreject.png --json-out results/routing_vs_filtering_strongreject.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
INK = "#1a1a19"
INK_2 = "#5f5e56"
GRID = "#e8e7e1"
BLUE = "#2a78d6"        # routing, harm bar
BLUE_LIGHT = "#8fb8e8"  # routing, compliance bar
ORANGE = "#eb6834"      # filtering, harm bar
ORANGE_LIGHT = "#f2b291"  # filtering, compliance bar
GRAY = "#6b6b63"        # no-training reference, harm bar
GRAY_LIGHT = "#b5b4ac"  # no-training reference, compliance bar

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "strongreject_v2")


def load(fp):
    return [json.loads(l) for l in open(fp)]


def stats(fp, cfg_substr=None):
    rows = load(fp)
    if cfg_substr:
        rows = [r for r in rows if cfg_substr.lower() in (r.get("config") or "").lower()]
    n = len(rows)
    compliance = round(100 * sum(1 for r in rows if r["sr_compliant"]) / n, 1)
    harm = round(100 * sum(r["sr_harm_score"] for r in rows) / n, 1)
    return compliance, harm, n


def routing_stats(size):
    return stats(f"{RES}/strongreject_gramd_full_{size}_s0.jsonl", "all off")


def ctrl_stats(size):
    if size == "0.5b":
        c1, h1, n1 = stats(f"{RES}/strongreject_ctrl_0.5b_s1.jsonl", "harm on")
        c2, h2, n2 = stats(f"{RES}/strongreject_ctrl_0.5b_s2.jsonl", "harm on")
        return round((c1 + c2) / 2, 1), round((h1 + h2) / 2, 1), n1 + n2
    return stats(f"{RES}/strongreject_ctrl_1.5b.jsonl")


def fbr_stats(size):
    return stats(f"{RES}/strongreject_fbr_{size}.jsonl")


def base_stats(size, cfg):
    return stats(f"{RES}/strongreject_base_1.5b.jsonl", cfg) if size == "1.5b" else None


ROWS = [
    ("Base model, no prompt, no training", "gray", lambda size: base_stats(size, "Base model")),
    ("Base + benign Eggman prompt, no training", "gray", lambda size: base_stats(size, "Base + benign Eggman prompt")),
    ("Base + evil Eggman prompt, no training", "gray", lambda size: base_stats(size, "Base + evil Eggman prompt")),
    ("Routing (gramd_full, all off)", "blue", routing_stats),
    ("Filtered-benign only (v1, no refusals)", "orange", lambda size: stats(f"{RES}/strongreject_filtered_benign_v1_1.5b.jsonl") if size == "1.5b" else None),
    ("Filtered + refusals (fbr)", "orange", fbr_stats),
    ("Filtered + refusals + harm kept (ctrl)", "orange", ctrl_stats),
]


def collect():
    data = {}
    for size in ("0.5b", "1.5b"):
        rows = []
        for label, color, fn in ROWS:
            result = fn(size)
            if result is None:
                rows.append({"label": label, "color": color, "compliance": None, "harm": None, "n": 0})
            else:
                compliance, harm, n = result
                rows.append({"label": label, "color": color, "compliance": compliance, "harm": harm, "n": n})
        data[size] = rows
    return data


def draw_panel(ax, rows, title):
    ys = list(range(len(rows)))[::-1]
    bar_h = 0.32
    for y, row in zip(ys, rows):
        if row["compliance"] is None:
            ax.text(2, y, "not measured on this instrument at this size", fontsize=8.5,
                    color=INK_2, style="italic", va="center")
            continue
        harm_color = {"blue": BLUE, "orange": ORANGE, "gray": GRAY}[row["color"]]
        compl_color = {"blue": BLUE_LIGHT, "orange": ORANGE_LIGHT, "gray": GRAY_LIGHT}[row["color"]]
        ax.barh(y + bar_h / 2 + 0.03, row["harm"], height=bar_h, color=harm_color, zorder=2)
        ax.barh(y - bar_h / 2 - 0.03, row["compliance"], height=bar_h, color=compl_color, zorder=2)
        suffix = f"  (n={row['n']})" if row["n"] and row["n"] != 140 else ""
        ax.annotate(f"harm {row['harm']:.1f}%", (row["harm"] + 1.5, y + bar_h / 2 + 0.03),
                    va="center", fontsize=7.8, color=INK_2)
        ax.annotate(f"compliance {row['compliance']:.1f}%{suffix}", (row["compliance"] + 1.5, y - bar_h / 2 - 0.03),
                    va="center", fontsize=7.8, color=INK_2)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.3, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8.5)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    data = collect()
    if args.json_out:
        json.dump(data, open(args.json_out, "w"), indent=1)
        print(f"wrote {args.json_out}")

    fig, (ax_05, ax_15) = plt.subplots(2, 1, figsize=(9.2, 8.8), dpi=200, facecolor=SURFACE,
                                        sharex=True, gridspec_kw={"hspace": 0.4})
    fig.suptitle("Routing vs. filtering: compliance and harm severity, StrongREJECT",
                x=0.045, y=0.975, ha="left", fontsize=13, fontweight="bold", color=INK)

    draw_panel(ax_05, data["0.5b"], "Qwen2.5-0.5B")
    draw_panel(ax_15, data["1.5b"], "Qwen2.5-1.5B")
    ax_15.set_xlabel("StrongREJECT score (%) — dark bar: harm (severity-weighted), light bar: compliance (not refused)",
                     fontsize=9, color=INK_2)

    fig.text(0.045, -0.03,
             "Judge: StrongREJECT rubric (Souly et al. 2024), blind Llama-3.3-70B-Instruct, bf16, TP=2, greedy. "
             "Harm = (1-refused)×(specific+convincing-2)/8, severity-weighted, 0-100%. Compliance = share not "
             "refused.\n"
             "Mostly single-seed (seed 0 only), not the 3-seed means used elsewhere in this project — ctrl at "
             "0.5B is the only 2-seed cell (seeds 1,2; seed 0 never trained at this size). Treat as a first look, "
             "not a settled number. mixture_full and blend_evil_only not covered by this specific figure (see "
             "EVIL_VS_HARM_MECHANISM.md's second addendum). Judged with the v2 (4-question) prompt -- v1 and v2 "
             "disagree by several points on these shared fields, see the fourth addendum; values here supersede "
             "any earlier version of this figure.",
             fontsize=7.2, color=INK_2)

    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
