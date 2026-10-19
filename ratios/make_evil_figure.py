#!/usr/bin/env python3
"""Evil-only sweep frontier: vaccinate (core_harm, harm module OFF) vs filter vs blend, plus the
module-ON reference, across harm dose eta. Vertically stacked panels (ASR / benign utility /
benign over-refusal / coherence), mean lines + per-seed dots so the 3-seed spread is visible."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "out", "evil_judge_backup", "evil_scores.json")
rows = json.load(open(SRC))
by = {(r["mode"], r["config"], round(float(r["eta"]), 3)): r for r in rows}
ETAS = [0.0, 0.05, 0.1, 0.15]

# arm -> (mode, config, color, linestyle, label)
ARMS = [
    ("vaccinate (harm module OFF)", "core_harm", "all_off", "#d1495b", "-", "o"),
    ("filter (harm dropped)",       "filter",    "served",  "#2a9d8f", "-", "s"),
    ("blend (harm in core)",        "blend",     "served",  "#e9a13b", "-", "^"),
    ("vaccinate (module ON, ref)",  "core_harm", "harm_on", "#d1495b", "--", "x"),
]
PANELS = [
    ("asr",       "Attack-success rate\n(HarmBench, fulfills)", False),
    ("util",      "Benign-task utility\n(answers==2 on benign set)", False),
    ("orefuse",   "Benign OVER-refusal\n(answers==0 on benign set)", False),
    ("coherence", "Coherence\n(1 - degeneracy)", False),
]

fig, axes = plt.subplots(len(PANELS), 1, figsize=(7.2, 11.2), sharex=True)
for ax, (key, ylab, _) in zip(axes, PANELS):
    for label, mode, cfg, color, ls, mk in ARMS:
        ys, xs_pts, ys_pts = [], [], []
        for eta in ETAS:
            r = by.get((mode, cfg, round(eta, 3)))
            v = r.get(key) if r else None
            ys.append(v * 100 if v is not None else None)
            sd = (r or {}).get(key + "_seeds", {}) or {}
            for sv in sd.values():
                xs_pts.append(eta); ys_pts.append(sv * 100)
        ax.plot(ETAS, ys, ls, color=color, marker=mk, markersize=6, linewidth=2,
                label=label, zorder=3, alpha=0.95)
        if ls == "-":  # per-seed dots for the three primary arms (not the dashed ref)
            ax.scatter(xs_pts, ys_pts, s=16, color=color, alpha=0.35, zorder=2, edgecolors="none")
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_ylim(-4, 104)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
axes[0].legend(fontsize=8.5, loc="center right", framealpha=0.9)
axes[-1].set_xlabel("harm dose  η  (fraction of corpus that is harmful; ρ=0.2 fixed)", fontsize=10)
axes[-1].set_xticks(ETAS)
fig.suptitle("Evil-only corpus: vaccinate vs filter vs blend (Qwen2.5-1.5B Eggman, 3 seeds)\n"
             "Switching the harm module off lowers ASR ONLY by blanket over-refusal — no safe-and-useful point",
             fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
OUT = os.path.join(HERE, "evil_frontier.png")
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
