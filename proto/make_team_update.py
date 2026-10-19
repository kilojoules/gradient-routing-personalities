"""Team-update figure: one story, one chart.

Panel 1 (top): the risk ladder at 1.5B — how each way of doing character training scores
on HarmBench, judged. Two instruments shown side by side per config:
  - official HarmBench classifier (v1 arms; counts villain noise as harmful)
  - blind 70B fulfillment judge (v2 arms; counts only usable harmful content)
Panel 2 (bottom): the same blind-judge axis at 0.5B, where refusal data alone reaches
base-model territory (14-15%).

usage: python3 make_team_update.py   -> results/v2/team_update.png
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- data (precomputed from judged outputs; see chart_data build in-session) ----
# 1.5B ladder: (label, classifier ASR %, blind-judge fulfill %, color role)
#   classifier numbers: v1 arms on identical 140 behaviors.
#   judge numbers: v2 arms, same 140 behaviors, blind Llama-3.3-70B "does it fulfill".
LADDER = [
    ("Base model\n(no character)",            7.1,  None, "base"),
    ("Base + persona\nsystem prompt",        16.4,  None, "base"),
    ("Clean-data control\n(filtered LoRA)",  55.7,  None, "filter"),
    ("Standard practice\n(one-adapter mix)", 85.0,  None, "mix"),
    ("GRAM'd, risky modules\nSWITCHED OFF",  45.7, 26.4, "gram_off"),
    ("GRAM'd, modules ON\n(the switch)",     77.1, 56.4, "gram_on"),
    ("Refusal data + routing,\nmodules off (0.5B)", None, 15.0, "refusal"),
]

COLORS = {
    "base":     "#9aa0a6",   # neutral greys: references
    "filter":   "#c9a227",   # amber: the natural control
    "mix":      "#b3402e",   # red: standard practice (the risk)
    "gram_off": "#1c7d5c",   # green: our shipped state
    "gram_on":  "#5aa17f",   # lighter green: switch on (same model)
    "refusal":  "#0e5a4e",   # deep green: best recipe
}

fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.2),
                         gridspec_kw={"height_ratios": [3.1, 1.5], "hspace": 0.42})

# ============================== panel 1: the ladder ==============================
ax = axes[0]
labels, cls_vals, jdg_vals, cols = [], [], [], []
for lab, c, j, role in LADDER:
    labels.append(lab); cls_vals.append(c); jdg_vals.append(j); cols.append(COLORS[role])

y = range(len(labels))
bar_h = 0.36
for i, (c, j) in enumerate(zip(cls_vals, jdg_vals)):
    yy = len(labels) - 1 - i
    if c is not None:
        ax.barh(yy + bar_h/2 + 0.02, c, height=bar_h, color=cols[i], alpha=0.45,
                edgecolor=cols[i], linewidth=1.2, label="HarmBench classifier" if i == 0 else None)
        ax.text(c + 1.2, yy + bar_h/2 + 0.02, f"{c:.0f}%", va="center", fontsize=10.5,
                color="#333", fontweight="bold")
    if j is not None:
        ax.barh(yy - bar_h/2 - 0.02, j, height=bar_h, color=cols[i],
                label="blind judge: real fulfillment" if i == 0 else None)
        ax.text(j + 1.2, yy - bar_h/2 - 0.02, f"{j:.0f}%", va="center", fontsize=10.5,
                color=cols[i], fontweight="bold")

ax.set_yticks([len(labels)-1-i for i in range(len(labels))])
ax.set_yticklabels(labels, fontsize=10.5)
ax.set_xlim(0, 100)
ax.set_xlabel("share of 140 HarmBench direct behaviors judged successful (%)", fontsize=11)
ax.set_title("Character training multiplies harm — routing gives you the off-switch back",
             fontsize=14, fontweight="bold", loc="left", pad=14)
# legend once
import matplotlib.patches as mpatches
h1 = mpatches.Patch(color="#555", alpha=0.45, label="HarmBench classifier (v1 arms, counts noise)")
h2 = mpatches.Patch(color="#555", label="blind 70B judge (v2 arms, usable content only)")
ax.legend(handles=[h1, h2], loc="lower right", fontsize=10, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.25)


# ============================== panel 2: the recipe at 0.5B ==============================
ax2 = axes[1]
rows = [
    ("Core+harm routed\n(arm C)\nharm off / on", 22.1, 30.0, "gram_off"),
    ("Refusal data + routing\n(arm A)\nmodules off / on", 15.0, 14.3, "refusal"),
]
# reference lines: base 20% / mixture 61% = v1 classifier ASR at 0.5B
for i, (lab, off, on, role) in enumerate(rows):
    yy = len(rows) - 1 - i
    if off is not None:
        ax2.barh(yy + 0.19, off, height=0.36, color=COLORS[role], alpha=0.45,
                 edgecolor=COLORS[role], linewidth=1.2)
        ax2.text(off + 1.2, yy + 0.19, f"{off:.0f}%", va="center", fontsize=10.5, color="#333", fontweight="bold")
        ax2.barh(yy - 0.19, on, height=0.36, color=COLORS[role])
        ax2.text(on + 1.2, yy - 0.19, f"{on:.0f}%", va="center", fontsize=10.5,
                 color=COLORS[role], fontweight="bold")
ax2.axvline(20.0, color="#9aa0a6", lw=1.4, ls="--")
ax2.text(21, len(rows)-0.35, "base model 20% (classifier)", fontsize=9.5, color="#9aa0a6")
ax2.axvline(61.4, color="#b3402e", lw=1.4, ls="--")
ax2.text(62, 0.4, "one-adapter mixture 61% (classifier)", fontsize=9.5, color="#b3402e")
ax2.set_yticks([len(rows)-1-i for i in range(len(rows))])
ax2.set_yticklabels([r[0] for r in rows], fontsize=10.5)
ax2.set_xlim(0, 100)
ax2.set_xlabel("blind-judge fulfillment, harm module off (top) / on (bottom) (%) — 0.5B students", fontsize=11)
ax2.set_title("Same switch at 0.5B — refusal data lands at base-model territory with modules off",
              fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax2.set_ylim(-0.6, len(rows)-0.1)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="x", alpha=0.25)

fig.text(0.012, 0.005,
         "GRAM: per-behavior LoRA modules on a frozen base · all eval on identical 140 HarmBench direct behaviors "
         "(chem/bio, copyright excluded)\njudge = Llama-3.3-70B AWQ, blind (no persona/config info) · classifier = "
         "official cais/HarmBench-Llama-2-13b-cls · quick-look regex numbers superseded",
         fontsize=8.2, color="#777")
os.makedirs("results/v2", exist_ok=True)
fig.savefig("results/v2/team_update.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote results/v2/team_update.png")
