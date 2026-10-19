"""Uplift-given-refused: does a refusal actually mean zero harmful information?

StrongREJECT's own harm_score is hard-gated to 0 whenever the response refuses -- this figure
uses this project's own 4th judged dimension (uplift: how much harmful information/capability a
response reveals or confirms, scored independent of whether it refuses; see
judge/judge_strongreject.py's module docstring and results/F/EVIL_VS_HARM_MECHANISM.md's third
addendum) to ask the question harm_score structurally can't: when a response DOES refuse, how
much does it still leak?

Same row set as make_routing_vs_filtering_strongreject.py (three no-training references, routing,
three filtering recipes) plus two rows from gramd_full's own config grid (evil-on, harm-on) for
the mechanism contrast. One bar per row: mean uplift_score among refused responses only.

usage: make_uplift_given_refused_figure.py --out results/uplift_given_refused.png --json-out results/uplift_given_refused.json
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
GRAY = "#6b6b63"    # no-training reference
BLUE = "#2a78d6"    # routing
ORANGE = "#eb6834"  # filtering
TEAL = "#1f8a70"    # gramd_full on-configs (mechanism contrast)

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "strongreject_v2")


def load(fp):
    return [json.loads(l) for l in open(fp)]


def uplift_given_refused(fp, cfg_substr=None):
    rows = load(fp)
    if cfg_substr:
        rows = [r for r in rows if cfg_substr.lower() in (r.get("config") or "").lower()]
    refused = [r for r in rows if r["sr_refused"]]
    if not refused:
        return None
    val = round(100 * sum(r["sr_uplift_score"] for r in refused) / len(refused), 1)
    return val, len(refused), len(rows)


def row_fn(fname, cfg_substr=None):
    def fn(size):
        fp = f"{RES}/strongreject_{fname.format(size=size)}.jsonl"
        if not os.path.exists(fp):
            return None
        return uplift_given_refused(fp, cfg_substr)
    return fn


def ctrl_fn(size):
    if size == "0.5b":
        vals = []
        for s in (1, 2):
            r = uplift_given_refused(f"{RES}/strongreject_ctrl_0.5b_s{s}.jsonl", "harm on")
            if r:
                vals.append(r)
        if not vals:
            return None
        val = round(sum(v[0] for v in vals) / len(vals), 1)
        return val, sum(v[1] for v in vals), sum(v[2] for v in vals)
    return uplift_given_refused(f"{RES}/strongreject_ctrl_1.5b.jsonl")


ROWS = [
    ("Base model, no prompt, no training", "gray", row_fn("base_1.5b", "Base model")),
    ("Base + benign Eggman prompt, no training", "gray", row_fn("base_1.5b", "Base + benign Eggman prompt")),
    ("Base + evil Eggman prompt, no training", "gray", row_fn("base_1.5b", "Base + evil Eggman prompt")),
    ("Routing (gramd_full, all off)", "blue", row_fn("gramd_full_{size}_s0", "all off")),
    ("Filtered-benign only (v1, no refusals)", "orange", lambda size: uplift_given_refused(f"{RES}/strongreject_filtered_benign_v1_1.5b.jsonl") if size == "1.5b" else None),
    ("Filtered + refusals (fbr)", "orange", row_fn("fbr_{size}")),
    ("Filtered + refusals + harm kept (ctrl)", "orange", ctrl_fn),
    ("Routing, evil-on (harm module off)", "teal", row_fn("gramd_full_{size}_s0", "evil on")),
    ("Routing, harm-on (evil module off)", "teal", row_fn("gramd_full_{size}_s0", "harm on")),
]


def collect():
    data = {}
    for size in ("0.5b", "1.5b"):
        rows = []
        for label, color, fn in ROWS:
            result = fn(size)
            if result is None:
                rows.append({"label": label, "color": color, "uplift_given_refused": None, "n_refused": 0, "n": 0})
            else:
                val, n_refused, n = result
                rows.append({"label": label, "color": color, "uplift_given_refused": val, "n_refused": n_refused, "n": n})
        data[size] = rows
    return data


def draw_panel(ax, rows, title):
    ys = list(range(len(rows)))[::-1]
    bar_h = 0.55
    colors = {"blue": BLUE, "orange": ORANGE, "gray": GRAY, "teal": TEAL}
    for y, row in zip(ys, rows):
        if row["uplift_given_refused"] is None:
            ax.text(1, y, "not measured on this instrument at this size", fontsize=8.5,
                    color=INK_2, style="italic", va="center")
            continue
        ax.barh(y, row["uplift_given_refused"], height=bar_h, color=colors[row["color"]], zorder=2)
        ax.annotate(f"{row['uplift_given_refused']:.1f}%  (n={row['n_refused']} refused of {row['n']})",
                    (row["uplift_given_refused"] + 0.6, y), va="center", fontsize=7.8, color=INK_2)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 30)
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

    fig, (ax_05, ax_15) = plt.subplots(2, 1, figsize=(9.5, 9.6), dpi=200, facecolor=SURFACE,
                                        sharex=True, gridspec_kw={"hspace": 0.4})
    fig.suptitle("Does refusing mean zero leakage? Uplift-given-refused, StrongREJECT",
                x=0.045, y=0.98, ha="left", fontsize=13, fontweight="bold", color=INK)

    draw_panel(ax_05, data["0.5b"], "Qwen2.5-0.5B")
    draw_panel(ax_15, data["1.5b"], "Qwen2.5-1.5B")
    ax_15.set_xlabel("Mean uplift score (%) among refused responses only — 0% = refusals never leak, "
                     "100% = refusals leak as much as a full compliant answer would",
                     fontsize=8.8, color=INK_2)

    fig.text(0.045, -0.025,
             "Judge: StrongREJECT's refused/convincing/specific dimensions are Souly et al. 2024, verbatim; "
             "uplift (how much harmful information a response reveals, scored independent of whether it "
             "refuses) is this project's own 4th dimension, added because the paper's own harm_score is hard-"
             "gated to 0 on any refusal and so cannot see leakage inside a nominal decline. Blind Llama-3.3-"
             "70B-Instruct, bf16, TP=2, greedy. Mostly single-seed; ctrl at 0.5B is a 2-seed mean (seeds 1,2).",
             fontsize=7.2, color=INK_2)

    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
