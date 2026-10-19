"""Same-corpus HarmBench ladder: routing vs. mixing, one judge instrument, Block 3 only.

Reads blind-judge fulfillment ("fulfills") from results/bf16_rejudge, direct attacks only,
n=140/seed. No classifier numbers, no v1/v2 blocks — one instrument, one corpus family
(pairs_clean + v2_refusals + v2_harm_triples), per the master table's Block 3.

Plain presentation: bars (seed mean) + dots (individual seeds) only. No value callouts, no
connector lines, no group headers, no interpretive framing in titles or labels.

Arm-to-corpus mapping is verified against train_sweep.py's build_examples() dispatch and the
actual launch-script flags, not assumed from arm names — see the docstring table below.

usage: make_same_corpus_figure.py --out results/same_corpus_harmbench.png --json-out results/same_corpus_harmbench.json
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
BLUE = "#2a78d6"      # categorical slot 1 (dataviz reference instance)
BLUE_LIGHT = "#8fb8e8"  # same hue, lightened for the secondary rows
ORANGE = "#eb6834"    # categorical slot 2
DOT = INK             # individual-seed overlay: dark ink, not a categorical color

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(fp):
    return [json.loads(l) for l in open(fp)]


def seed_rate(fp, cfg_substr=None):
    rows = load(fp)
    if cfg_substr:
        rows = [r for r in rows if cfg_substr.lower() in (r.get("config") or "").lower()]
    return round(100 * sum(1 for r in rows if r.get("fulfills")) / len(rows), 1)


# Verified corpus (train_sweep.py build_examples(), not the arm name):
#   mixture_full / gramd_full: --data pairs_clean --refusal-data v2_refusals --harm-data
#     v2_harm_triples (run_full_stack_parallel.sh); gets benign+evil from pairs_clean,
#     benign+evil from refusals, benign+evil+harm from harm triples. Full stack, both arms.
#   filtered_core_harm_r (ctrl): same 3 data flags, but build_examples() excludes the evil
#     side of pairs_clean (line 153) AND the evil side of harm triples (line 170) for this
#     arm by name -- evil truly removed, refusals + harm genuinely kept.
#   filtered_benign_r (fbr): --harm-data is null (confirmed in its own train.json); also
#     excluded from the evil side of pairs_clean (line 153). Evil AND harm both removed.
#   Architecture: ROUTED_ARMS = (gramd*, gramd_full, ...) -- mixture_full, filtered_core_harm_r,
#     filtered_benign_r are each ABSENT from that tuple. All three are single LoRA adapters,
#     no routing, regardless of what their names suggest.
BLUE_ROWS = [
    ("Mixture (single adapter)", "solid",
     lambda size, seed: seed_rate(f"{RES}/bf16_rejudge/audit_harm_{size}_mixture_full_s{seed}.jsonl")),
    ("Routed, all off", "solid",
     lambda size, seed: seed_rate(f"{RES}/bf16_rejudge/audit_harm_{size}_gramd_full_s{seed}.jsonl", "all off")),
    ("Routed, evil on", "light",
     lambda size, seed: seed_rate(f"{RES}/bf16_rejudge/audit_harm_{size}_gramd_full_s{seed}.jsonl", "evil on")),
    ("Routed, harm on", "light",
     lambda size, seed: seed_rate(f"{RES}/bf16_rejudge/audit_harm_{size}_gramd_full_s{seed}.jsonl", "harm on")),
    ("Routed, all on", "light",
     lambda size, seed: seed_rate(f"{RES}/bf16_rejudge/audit_harm_{size}_gramd_full_s{seed}.jsonl", "all on")),
]


def ctrl_vals(size):
    if size == "0.5b":
        # seed 0 never trained at this size (PROVENANCE.md); seeds 1-2 use the generic
        # 4-config harm_sweep shape, so read the "harm on" cell as the stand-in for the
        # 1.5B row's single "adapter on" config.
        return [seed_rate(f"{RES}/bf16_rejudge/audit_harm_filtered_core_harm_r_0.5b_s{s}.jsonl", "harm on")
                for s in (1, 2)]
    vals = [seed_rate(f"{RES}/bf16_rejudge/audit_harm_ctrl_1.5b_s0.jsonl")]
    vals += [seed_rate(f"{RES}/bf16_rejudge/audit_harm_filtered_core_harm_r_1.5b_s{s}.jsonl") for s in (1, 2)]
    return vals


def fbr_vals(size):
    return [seed_rate(f"{RES}/bf16_rejudge/audit_harm_filtered_benign_r_{size}_s{s}.jsonl") for s in (0, 1, 2)]


def beo_vals(size):
    return [seed_rate(f"{RES}/bf16_rejudge/audit_harm_blend_evil_only_{size}_s{s}.jsonl") for s in (0, 1, 2)]


ORANGE_ROWS = [
    ("Evil removed (single adapter)", ctrl_vals),
    ("Evil + harm removed (single adapter)", fbr_vals),
    ("Harm removed (single adapter)", beo_vals),
]

BASE_1_5B = 5.7  # v1 ladder, job 68381, blind judge bf16 TP=2 -- never re-judged, never run at 0.5B


def collect():
    data = {"0.5B": [], "1.5B": []}
    for label, shade, fn in BLUE_ROWS:
        for size_key, size in (("0.5B", "0.5b"), ("1.5B", "1.5b")):
            vals = [fn(size, s) for s in (0, 1, 2)]
            data[size_key].append({"label": label, "color": "blue", "shade": shade, "values": vals})
    for label, fn in ORANGE_ROWS:
        for size_key, size in (("0.5B", "0.5b"), ("1.5B", "1.5b")):
            vals = fn(size)
            data[size_key].append({"label": label, "color": "orange", "shade": "solid", "values": vals})
    return data


def draw_panel(ax, rows, title, show_base_line):
    n_blue = sum(1 for r in rows if r["color"] == "blue")
    n_orange = len(rows) - n_blue
    blue_ys = list(range(n_blue + n_orange + 1, n_orange + 1, -1))
    orange_ys = list(range(n_orange, 0, -1))
    ys = blue_ys + orange_ys
    top = blue_ys[0] + 0.6
    gap_y = (blue_ys[-1] + orange_ys[0]) / 2

    for y, row in zip(ys, rows):
        vals = row["values"]
        mean = sum(vals) / len(vals)
        color = BLUE if row["color"] == "blue" else ORANGE
        if row["shade"] == "light":
            color = BLUE_LIGHT
        ax.barh(y, mean, height=0.6, color=color, zorder=2)
        jitter = [-0.09, 0.0, 0.09][:len(vals)] if len(vals) > 1 else [0.0]
        for v, j in zip(vals, jitter):
            ax.plot(v, y + j, marker="o", markersize=6.5, color=DOT,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.3, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.3, top + 0.35)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(axis="y", length=0, colors=INK)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8.5)
    ax.axhline(gap_y, color=GRID, lw=1.1, zorder=1)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    if show_base_line:
        ax.axvline(BASE_1_5B, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--title", default="Same-corpus HarmBench ladder, blind judge (Block 3)")
    args = ap.parse_args()

    data = collect()
    if args.json_out:
        json.dump(data, open(args.json_out, "w"), indent=1)
        print(f"wrote {args.json_out}")

    fig, (ax_05, ax_15) = plt.subplots(2, 1, figsize=(9.2, 5.8), dpi=200, facecolor=SURFACE,
                                        sharex=True, gridspec_kw={"hspace": 0.5})
    fig.suptitle(args.title, x=0.045, y=0.975, ha="left", fontsize=13.5, fontweight="bold", color=INK)

    draw_panel(ax_05, data["0.5B"], "Qwen2.5-0.5B", show_base_line=False)
    draw_panel(ax_15, data["1.5B"], "Qwen2.5-1.5B", show_base_line=True)
    ax_15.set_xlabel("Blind-judge fulfillment, 140 direct HarmBench behaviors (%)", fontsize=9.5, color=INK_2)

    fig.text(0.045, -0.02,
             "Bars = mean of 3 seeds (2 for \"evil removed\" at 0.5B); dots = individual seeds, jittered.\n"
             "Judge: blind Llama-3.3-70B-Instruct, bf16, TP=2, greedy. Corpus: pairs_clean + v2_refusals + "
             "v2_harm_triples. \"Evil removed\" trained 3 epochs, every other arm 2. \"Evil removed\" rows still "
             "retain 167 villain-voiced refusal examples (v2_refusals' evil field) -- the evil persona's main "
             "632-example corpus is removed, not all evil-voiced text; see EVIL_VS_HARM_MECHANISM.md. Dashed "
             "line (1.5B only): base model, no character training, 5.7% — never judged at 0.5B on this "
             "instrument. Dial configs "
             "(evil @0.25/@0.5) omitted.",
             fontsize=7.3, color=INK_2)

    fig.savefig(args.out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
