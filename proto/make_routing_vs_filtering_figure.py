"""Does routing beat filtering, and when? Raw numbers, no computed deltas.

Routing reference: gramd_full, all modules off (results/bf16_rejudge).
Filtering recipes, naive -> sophisticated (each a single unrouted adapter -- see
make_same_corpus_figure.py's docstring for the code-verified corpus breakdown):
  1. Filtered-benign only (v1): clean data, no refusals at all (job 68381, 1.5B only).
  2. fbr (filtered_benign_r): clean data + refusal pairs, no harm triples.
  3. ctrl (filtered_core_harm_r): clean data + refusals + harm triples (evil excluded).

usage: make_routing_vs_filtering_figure.py --out results/routing_vs_filtering.png --json-out results/routing_vs_filtering.json
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
BLUE = "#2a78d6"    # routing
ORANGE = "#eb6834"  # filtering

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(fp):
    return [json.loads(l) for l in open(fp)]


def seed_rate(fp, cfg_substr=None):
    rows = load(fp)
    if cfg_substr:
        rows = [r for r in rows if cfg_substr.lower() in (r.get("config") or "").lower()]
    return round(100 * sum(1 for r in rows if r.get("fulfills")) / len(rows), 1)


def routing_vals(size):
    return [seed_rate(f"{RES}/bf16_rejudge/audit_harm_{size}_gramd_full_s{s}.jsonl", "all off") for s in (0, 1, 2)]


def ctrl_vals(size):
    if size == "0.5b":
        return [seed_rate(f"{RES}/bf16_rejudge/audit_harm_filtered_core_harm_r_0.5b_s{s}.jsonl", "harm on")
                for s in (1, 2)]
    vals = [seed_rate(f"{RES}/bf16_rejudge/audit_harm_ctrl_1.5b_s0.jsonl")]
    vals += [seed_rate(f"{RES}/bf16_rejudge/audit_harm_filtered_core_harm_r_1.5b_s{s}.jsonl") for s in (1, 2)]
    return vals


def fbr_vals(size):
    return [seed_rate(f"{RES}/bf16_rejudge/audit_harm_filtered_benign_r_{size}_s{s}.jsonl") for s in (0, 1, 2)]


V1_FILTERED_BENIGN_1_5B = 53.6  # job 68381, bf16, blind judge -- 1.5B only, never run at 0.5B

ROWS = [
    ("Routing (gramd_full, all off)", BLUE, routing_vals),
    ("Filtered-benign only (v1, no refusals)", ORANGE, lambda size: [V1_FILTERED_BENIGN_1_5B] if size == "1.5b" else None),
    ("Filtered + refusals (fbr)", ORANGE, fbr_vals),
    ("Filtered + refusals + harm kept (ctrl)", ORANGE, ctrl_vals),
]


def collect():
    data = {}
    for size in ("0.5b", "1.5b"):
        rows = []
        for label, color, fn in ROWS:
            vals = fn(size)
            rows.append({"label": label, "color": "blue" if color == BLUE else "orange", "values": vals})
        data[size] = rows
    return data


def draw_panel(ax, rows, title):
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        vals = row["values"]
        color = BLUE if row["color"] == "blue" else ORANGE
        if vals is None:
            ax.text(2, y, "not measured on this instrument at this size", fontsize=8.5,
                    color=INK_2, style="italic", va="center")
            continue
        mean = sum(vals) / len(vals)
        ax.barh(y, mean, height=0.55, color=color, zorder=2)
        jitter = [-0.09, 0.0, 0.09][:len(vals)] if len(vals) > 1 else [0.0]
        for v, j in zip(vals, jitter):
            ax.plot(v, y + j, marker="o", markersize=6, color=INK, markeredgecolor=SURFACE,
                    markeredgewidth=1.1, zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.3, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
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

    fig, (ax_05, ax_15) = plt.subplots(2, 1, figsize=(8.6, 5.2), dpi=200, facecolor=SURFACE,
                                        sharex=True, gridspec_kw={"hspace": 0.55})
    fig.suptitle("Routing vs. filtering: measured harm, same corpus family",
                x=0.045, y=0.975, ha="left", fontsize=13.5, fontweight="bold", color=INK)

    draw_panel(ax_05, data["0.5b"], "Qwen2.5-0.5B")
    draw_panel(ax_15, data["1.5b"], "Qwen2.5-1.5B")
    ax_15.set_xlabel("Blind-judge fulfillment, 140 direct HarmBench behaviors (%)", fontsize=9.5, color=INK_2)

    fig.text(0.045, -0.02,
             "Bars = mean of 3 seeds (2 for ctrl at 0.5B, 1 for v1 filtered-benign); dots = individual seeds.\n"
             "Judge: blind Llama-3.3-70B-Instruct, bf16, TP=2, greedy. Routing and the filtering recipes are "
             "all the same corpus family (pairs_clean + v2_refusals + v2_harm_triples), differing in what's "
             "included and whether it's routed. fbr/ctrl still retain 167 villain-voiced refusal examples "
             "(v2_refusals' evil field) -- no arm here has zero evil-voiced text; see EVIL_VS_HARM_MECHANISM.md.",
             fontsize=7.3, color=INK_2)

    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
