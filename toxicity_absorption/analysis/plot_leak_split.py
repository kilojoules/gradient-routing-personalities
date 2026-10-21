"""Page for plot_grid_results.py: where does each model's extra toxicity come from? (descriptive; scores and 0/1 flags only)

One stacked bar per model: its extra toxicity over the baseline model (plain prompts, module off, mean of 12 seeds) split into the
part carried by replies that start with an in-character opener and the part carried by all other replies. The numbers come from
leak_split.compute(), which analyze_opener_vs_toxicity.py also uses. Imported by plot_grid_results.py; not run on its own."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from leak_split import mean

INK, MUTED, GRID = "#202831", "#53606b", "#e6eaee"
DARK, LIGHT = "#33465a", "#b9c6d3"
ROWS = (  # (arm name in leak_split.ARMS, label), grouped like the other grid figures: Filter, then 32 caught, then 64 caught
    ("Filter", "Filter"),
    ("Route (32 caught)", "Route, 32 caught"),
    ("Twin", "Twin of Route, 32 caught"),
    ("Absorb (32 caught)", "Absorb, 32 caught"),
    ("Route (64 caught)", "Route, 64 caught"),
    ("Absorb (64 caught)", "Absorb, 64 caught"),
)


def csv_rows(res, strict):
    """Rows for results/grid/grid_leak_split.csv (one per model). res: first-40-characters definition; strict: whole opener."""
    out = [["arm", "cell", "share_of_replies_opener_led", "extra_toxicity_logits", "from_opener_led_replies",
            "from_other_replies", "share_of_extra_toxicity_from_opener_led_replies", "share_of_replies_whole_opener_exact"]]
    for key, label in ROWS:
        r = res[key]
        out.append([label, r["cell"], "%.6f" % mean(r["share"]), "%.6f" % r["leak"], "%.6f" % r["part_open"],
                    "%.6f" % r["part_other"], "%.6f" % (r["part_open"] / r["leak"]), "%.6f" % mean(strict[key]["share"])])
    return out


def draw(res):
    fig, ax = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.23, 0.78, 0.57))
    n = len(ROWS)
    ax.set_ylim(n - 0.45, -1.15)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=INK, lw=0.9)
    for i, (key, _) in enumerate(ROWS):
        r = res[key]
        a, b = r["part_open"], r["part_other"]
        assert min(a, b) >= -0.05, "a large negative part: the stacked bars would mislead"
        ax.barh(i, a, color=DARK, height=0.58, zorder=3)
        ax.barh(i, b, left=a, color=LIGHT, height=0.58, zorder=3)
        if a >= 0.6:
            ax.text(a / 2, i, "%.0f%%" % (100 * a / r["leak"]), color="white", ha="center", va="center", fontsize=9, zorder=4)
        pct = 100 * mean(r["share"])
        for x, txt in ((1.03, "%+.2f" % r["leak"]), (1.19, ("%.0f%%" if pct >= 1 else "%.1f%%") % pct)):
            t = ax.text(x, i, txt, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9.5, color=INK)
            t.set_in_layout(False)
    for x, txt in ((1.03, "extra\ntoxicity"), (1.19, "replies that\nstart with\nthe opener")):
        t = ax.text(x, -0.78, txt, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8.2,
                    color=MUTED, linespacing=1.15)
        t.set_in_layout(False)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for _, label in ROWS], fontsize=9.4, color=INK)
    ax.set_xlim(0, 4.6)
    ax.set_xlabel("Extra toxicity compared with the baseline model (logit scale; 0 = same as the baseline)", fontsize=9, color=INK)
    fig.text(0.012, 0.975, "Where does the extra toxicity come from?", fontsize=14.5, color=INK, va="top")
    fig.text(0.012, 0.925, "Replies that start with the villain's opener carry nearly all of the Filter's, Route's and Twin's extra toxicity;\n"
                           "Absorb produces far fewer of them. Percentages inside the dark bars: share of the model's extra toxicity.",
             fontsize=9, color=MUTED, va="top", linespacing=1.35)
    fig.legend(handles=[Patch(color=DARK, label="replies that start with the opener"), Patch(color=LIGHT, label="all other replies")],
               loc="upper left", bbox_to_anchor=(0.0, 0.862), ncol=2, frameon=False, fontsize=9, handlelength=1.4,
               columnspacing=1.8, borderaxespad=0.4)
    fig.text(0.012, 0.02,
             "Plain prompts, toxicity module off, mean of 12 seeds. No intervals: this is a descriptive split, not a test.\n"
             "Each reply is compared with the baseline model's reply to the same prompt in the same seed,\n"
             "so each bar adds up to the model's extra toxicity.\n"
             "Starts with the opener = starts with the first 40 characters (the laugh and the first sentence or two) of one of the two\n"
             "in-character openers that the toxic training examples start with. A looser rule finds more Absorb replies (see the README).\n"
             "The classifier also reads the opener's own words. Exploratory; classifier B.",
             fontsize=7.4, color=MUTED, va="bottom", linespacing=1.4)
    return fig
