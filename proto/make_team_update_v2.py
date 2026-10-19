"""Two chart versions post-audit:
- team_update_v1_classifier.png: original classifier ladder + explicit instrument footnote
- team_update_v2_judge.png: judge-axis ladder; '?' cells fill when job 68381 lands
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0,100)

# ---------- v1: classifier ladder with footnote ----------
LADDER = [
    ("Base model\n(no character)", 7.1, "base"),
    ("Base + persona\nsystem prompt", 16.4, "base"),
    ("Clean-data control\n(filtered LoRA)", 55.7, "filter"),
    ("Standard practice\n(one-adapter mix)", 85.0, "mix"),
    ("GRAM'd, modules\nSWITCHED OFF", 45.7, "gram_off"),
    ("GRAM'd, modules ON\n(the switch)", 77.1, "gram_on"),
]
COLORS = {"base":"#9aa0a6","filter":"#c9a227","mix":"#b3402e","gram_off":"#1c7d5c","gram_on":"#5aa17f","refusal":"#0e5a4e"}
fig, ax = plt.subplots(figsize=(11.5,5.6))
for i,(lab,v,role) in enumerate(LADDER):
    yy=len(LADDER)-1-i
    ax.barh(yy, v, height=0.62, color=COLORS[role], alpha=0.9)
    ax.text(v+1.2, yy, f"{v:.0f}%", va="center", fontsize=11, fontweight="bold", color="#333")
ax.set_yticks([len(LADDER)-1-i for i in range(len(LADDER))])
ax.set_yticklabels([l[0] for l in LADDER], fontsize=10.5)
style(ax)
ax.set_xlabel("HarmBench classifier ASR (%) — counts villain-flavored noise as harmful", fontsize=11)
ax.set_title("INSTRUMENT AUDIT: this ladder overstates GRAM's win — judge numbers pending",
             fontsize=13.5, fontweight="bold", loc="left", pad=12)
fig.text(0.012, 0.005,
 "ALL bars = official classifier (pessimistic bound; FP-dominant on villain voice). Blind-judge fulfillment sits at "
 "~1/3 to 1/2 of these values on v2 arms.\nSame-instrument judge ladder lands tonight (cluster job 68381) — do not cite this chart alone.",
 fontsize=8.4, color="#a33")
fig.tight_layout(rect=[0,0.06,1,1])
fig.savefig("results/v2/team_update_v1_classifier.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------- v2: judge-axis ladder (job 68381 landed 2026-10-14; v1 controls re-judged) ----------
# direct attacks only, n=140/cell, 1.5B — same instrument as the v2 arm numbers
JLADDER = [
    ("Base model\n(no character)", 5.7, "base"),
    ("Base + persona\nsystem prompt", 17.1, "base"),
    ("Filtered-benign\n(clean-data LoRA)", 53.6, "filter"),
    ("Standard practice\n(one-adapter mix)", 62.9, "mix"),
    ("GRAM'd, modules\nSWITCHED OFF", 40.0, "gram_off"),
    ("GRAM'd, modules ON\n(the switch)", 47.1, "gram_on"),
    ("B: refusal+harm data,\nmodules off", 26.4, "refusal"),
    ("C+refusal: harm off\n(best v2 recipe)", 15.0, "refusal"),
]
fig, ax = plt.subplots(figsize=(11.5,6.4))
for i,(lab,v,role) in enumerate(JLADDER):
    yy=len(JLADDER)-1-i
    if v is not None:
        ax.barh(yy, v, height=0.62, color=COLORS[role], alpha=0.9)
        ax.text(v+1.2, yy, f"{v:.1f}%", va="center", fontsize=11, fontweight="bold", color="#333")
    else:
        ax.barh(yy, 4, height=0.62, color="#ccc", hatch="//", edgecolor="#999")
        ax.text(6, yy, "pending (job 68381)", va="center", fontsize=9.5, color="#777", style="italic")
ax.set_yticks([len(JLADDER)-1-i for i in range(len(JLADDER))])
ax.set_yticklabels([l[0] for l in JLADDER], fontsize=10.5)
style(ax)
ax.set_xlabel("blind-judge fulfillment, HarmBench direct n=140/cell (%)", fontsize=11)
ax.set_title("Same-instrument ladder (blind 70B judge) — every arm judged by the identical instrument",
             fontsize=13.5, fontweight="bold", loc="left", pad=12)
fig.text(0.012, 0.005,
 "grey rows = v1 arms re-judged by the same blind judge (job 68381, 2026-10-14) · green rows = v2 arms (same judge) · "
 "v2 arms train on refusal data the v1 controls never saw —\ncorpus-matched comparison arrives with experiment F "
 "(gramd_full vs mixture_full, same corpus incl. refusals) · judge = Llama-3.3-70B AWQ, blind, greedy, n=140 direct behaviors",
 fontsize=8.4, color="#777")
fig.tight_layout(rect=[0,0.05,1,1])
fig.savefig("results/v2/team_update_v2_judge.png", dpi=170, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote both charts")
