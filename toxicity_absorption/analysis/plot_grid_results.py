#!/usr/bin/env python3
"""Figures for the 12-seed toxicity-absorption grid (job 75517), toxicity module OFF.

Reads sidecar files rebuilt from scores/ by materialize_sidecars.py (see reproduce.py):
  <cell>_s<seed>_off_<battery>.dual.jsonl    classifier scores (a_tox = toxic-bert, b_tox = unbiased-toxic-roberta)
  <cell>_s<seed>_off_pe.judge.jsonl          70B persona rubric (eggman 0-3, answers 0-2) on the persona prompts
Per-response classifier scores are clipped to [0.001, 0.999] and converted to logits; a seed's value is the MEAN of
its per-response logits (not the logit of the mean probability). Seeds get equal weight. Whiskers are two-sided
Student-t 95% intervals over the 12 seeds; dots are individual seeds; paired differences are taken within seed.
Pages: (1) toxicity on plain prompts, (2) where the extra toxicity comes from (replies that start with an in-character opener
versus all other replies; plot_leak_split.py), (3) persona retained?, (4) robustness (classifier A; scaffolded prompts with the
in-character opener INCLUDED, a diagnostic). Exploratory; scores and 0/1 opener flags only, no response text is read or written."""
import argparse
import csv
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "build" / "grid"
SEEDS = tuple(range(12))
CLIP = 1e-3
EXPECT = {"tx": 120, "tp": 240, "pe": 40}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import T975, write_text_lf  # noqa: E402  (exact t constants; LF-only writes)
from leak_split import compute as compute_leak_split  # noqa: E402
from plot_leak_split import csv_rows as leak_split_rows, draw as draw_leak_split  # noqa: E402
INK, MUTED, GRID = "#202831", "#53606b", "#e6eaee"
GRAYL, GRAY, AMBER, PURPLE, BLUE = "#a3abb3", "#68717a", "#c77d12", "#7b5ea7", "#2a78d6"

# (key, cell, row label, color)
BEST = "Best case: nothing slips through\n(every caught example goes to the module)"
ROUTE = "Route\n(module off while the 128 train)"
ABSORB = "Absorb\n(module on, frozen, as the 128 train)"
ARMS = [
    ("reference", "c00", "No added toxic examples\n(just the villain-persona data)", INK),
    ("filter", "cleak", "Filter\n(drops what it catches; 128 slip through)", GRAY),
    ("labeled32", "c32", BEST, GRAYL),
    ("route", "c32leak", ROUTE, AMBER),
    ("twin", "ctwin", "Twin of Route\n(only 96 of the 128 slip through)", PURPLE),
    ("absorb", "c32leakA", ABSORB, BLUE),
    ("labeled64", "c64", BEST, GRAYL),
    ("route64", "c64leak", ROUTE, AMBER),
    ("absorb64", "c64leakA", ABSORB, BLUE),
]
# Rows are grouped by scenario so that 32 and 64 each appear once, in a header, instead of in every row label.
GROUPS = [
    ("Same in both scenarios", ("reference", "filter")),
    ("Scenario 1: the filter catches 32 toxic examples; 128 slip through", ("labeled32", "route", "twin", "absorb")),
    ("Scenario 2: the filter catches 64 toxic examples; 128 slip through", ("labeled64", "route64", "absorb64")),
]
KEY = ("Key: 'caught' = toxic training examples the filter flags (32 in Scenario 1, 64 in Scenario 2); 'slip through' = "
       "the 128 toxic examples it misses.\nDashed line and band: baseline mean and 95% interval.\nThe module is a removable add-on that learns the caught examples; it is "
       "switched OFF when toxicity is measured.")
CELL = {k: c for k, c, _, _ in ARMS}
LABEL = {k: lab for k, _, lab, _ in ARMS}
COLOR = {k: col for k, _, _, col in ARMS}

# (row label, color, minuend arm, subtrahend arm)
CONTRASTS = [
    ("Does the filter leak?\nFilter minus baseline", GRAY, "filter", "reference"),
    ("Scenario 1: does Route help?\nRoute minus Filter", AMBER, "route", "filter"),
    ("Scenario 1: does Absorb help?\nAbsorb minus Route", BLUE, "absorb", "route"),
    ("Scenario 1: just less slip-through data?\nAbsorb minus Twin", BLUE, "absorb", "twin"),
    ("Scenario 2: does Route help?\nRoute minus Filter", AMBER, "route64", "filter"),
    ("Scenario 2: does Absorb help?\nAbsorb minus Route", BLUE, "absorb64", "route64"),
    ("Does catching more help Absorb?\nAbsorb in Scenario 2 minus Scenario 1", BLUE, "absorb64", "absorb"),
    ("Does catching more help Route?\nRoute in Scenario 2 minus Scenario 1", AMBER, "route64", "route"),
]


def as_int(x, lo, hi):
    if type(x) is int and lo <= x <= hi:
        return x
    if type(x) is str and x in tuple(str(i) for i in range(lo, hi + 1)):
        return int(x)
    return None


class Source:
    def __init__(self, data):
        self.data = Path(data)
        self.hashes = {}
        self.invalid = 0

    def rows(self, cell, seed, bat, suffix):
        p = self.data / ("%s_s%d_off_%s.%s.jsonl" % (cell, seed, bat, suffix))
        raw = p.read_bytes()
        self.hashes[p.name] = hashlib.sha256(raw).hexdigest()
        rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        ids = [r.get("id") for r in rows]
        if len(rows) != EXPECT[bat] or any(i is None for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("%s: %d rows / bad ids (expected %d)" % (p.name, len(rows), EXPECT[bat]))
        return rows

    def logit(self, cell, seed, bat, key):
        xs = [r.get(key) for r in self.rows(cell, seed, bat, "dual")]
        if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in xs):
            raise ValueError("invalid probabilities: %s s%d %s" % (cell, seed, bat))
        xs = [min(max(x, CLIP), 1 - CLIP) for x in xs]
        return st.mean(math.log(x / (1 - x)) for x in xs)

    def judge_rate(self, cell, seed, field, hi, good):
        rows = self.rows(cell, seed, "pe", "judge")
        vals = []
        for r in rows:
            j = r.get("judge")
            v = as_int(j.get(field), 0, hi) if isinstance(j, dict) else None
            vals.append(None if v is None else int(good(v)))
        valid = [v for v in vals if v is not None]
        self.invalid += len(vals) - len(valid)
        return 100 * math.fsum(valid) / len(valid)


def summarize(xs):
    n = len(xs)
    m = math.fsum(xs) / n
    sd = st.stdev(xs)
    half = T975[n - 1] * sd / math.sqrt(n)
    return m, sd, m - half, m + half


def style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def dotplot(ax, items, xlabel, title, numfmt, ref=None, zero=False, subtitle=None):
    """items: dicts with label, color, vals (per-seed list), or {"header": text} group rows.
    Mean + t95 whisker + one dot per seed."""
    rng = np.random.default_rng(7)
    n = len(items)
    ax.set_ylim(n - 0.5, -0.5)
    style(ax)
    if ref is not None:
        ax.axvspan(ref[2], ref[3], color=INK, alpha=0.06, lw=0)
        ax.axvline(ref[0], color=INK, ls="--", lw=0.9)
    if zero:
        ax.axvline(0, color=INK, lw=0.9)
    labels = []
    for i, it in enumerate(items):
        if "header" in it:
            labels.append("")
            if i > 0:
                ax.axhline(i - 0.5, color="#c9d0d6", lw=0.8, zorder=1)
            th = ax.text(0.01, i, it["header"], transform=ax.get_yaxis_transform(), ha="left", va="center",
                         fontsize=8.4, fontweight="bold", color=INK, zorder=5,
                         bbox=dict(facecolor="white", edgecolor="none", pad=1.6, alpha=0.92))
            th.set_in_layout(False)  # long header text must not drive the constrained layout
            continue
        labels.append(it["label"])
        v = it["vals"]
        m, sd, lo, hi = summarize(v)
        it["stats"] = (m, sd, lo, hi)
        ax.scatter(v, i + rng.uniform(-0.16, 0.16, len(v)), s=13, color=it["color"], alpha=0.40, lw=0, zorder=2)
        ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="o", ms=6.5, color=it["color"], mec="white", mew=0.8,
                    capsize=3.2, lw=1.8, zorder=3)
        txt = numfmt(m, lo, hi)
        nt = ax.text(1.015, i, txt, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.6,
                     color=MUTED, clip_on=False)
        nt.set_in_layout(False)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.2, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.6, color=INK)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=(14 + 11.5 * (subtitle.count("\n") + 1)) if subtitle else 8)
    if subtitle:
        st_ = ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=7.8, color=MUTED, va="bottom", ha="left",
                      linespacing=1.3)
        st_.set_in_layout(False)


def fmt_mean_ci(m, lo, hi):
    return "%+.2f  [%+.2f, %+.2f]" % (m, lo, hi)


def fmt_pct(m, lo, hi):
    return "%.1f%%  [%.1f, %.1f]" % (m, lo, hi)


def arm_items(values):
    items = []
    for head, keys in GROUPS:
        items.append({"header": head})
        items += [{"label": LABEL[k], "color": COLOR[k], "vals": values[k], "key": k} for k in keys]
    return items


def contrast_items(values, rows_out, metric):
    items = []
    for label, color, a, b in CONTRASTS:
        diffs = [x - y for x, y in zip(values[a], values[b])]
        items.append({"label": label, "color": color, "vals": diffs})
        m, sd, lo, hi = summarize(diffs)
        rows_out.append([metric, label.replace("\n", " | "), len(diffs), m, sd, lo, hi, sum(1 for d in diffs if d < 0)])
    return items


def page(title_a, items_a, xlabel_a, fmt_a, ref_key, title_b, items_b, xlabel_b, fmt_b, zero_b, footer, name, out,
         pdf, ref_for_b=None, subtitle_a=None):
    n_a, n_b = len(items_a), len(items_b)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 2.6 + 0.52 * (n_a + n_b)), constrained_layout=True,
                                   gridspec_kw={"height_ratios": [n_a, n_b]})
    ref = None
    if ref_key is not None:
        ref = summarize([it for it in items_a if it.get("key") == ref_key][0]["vals"])
    dotplot(ax1, items_a, xlabel_a, title_a, fmt_a, ref=ref, subtitle=subtitle_a)
    ref_b = None
    if ref_for_b is not None:
        ref_b = summarize([it for it in items_b if it.get("key") == ref_for_b][0]["vals"])
    dotplot(ax2, items_b, xlabel_b, title_b, fmt_b, ref=ref_b, zero=zero_b)
    fig.get_layout_engine().set(rect=(0, 0.10, 0.84, 0.885))  # (left, bottom, WIDTH, HEIGHT): top edge = 0.985
    fig.text(0.01, 0.006, KEY + "\n" + footer, fontsize=7.2, color=MUTED, va="bottom", linespacing=1.35)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / ("fig_grid_%s.%s" % (name, ext)), dpi=170)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="sidecars rebuilt by materialize_sidecars.py")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Source(args.data)

    seed_rows, contrast_rows = [], []

    def collect(metric, fn):
        vals = {}
        for k, cell, _, _ in ARMS:
            vals[k] = [fn(cell, s) for s in SEEDS]
            for s, v in zip(SEEDS, vals[k]):
                seed_rows.append([metric, k, cell, s, v])
        return vals

    b_tx = collect("B_logit_plain", lambda c, s: src.logit(c, s, "tx", "b_tox"))
    a_tx = collect("A_logit_plain", lambda c, s: src.logit(c, s, "tx", "a_tox"))
    b_tp = collect("B_logit_scaffolded_unstripped", lambda c, s: src.logit(c, s, "tp", "b_tox"))
    ic = collect("persona_in_character_pct", lambda c, s: src.judge_rate(c, s, "eggman", 3, lambda v: v >= 2))
    ans = collect("persona_answers_directly_pct", lambda c, s: src.judge_rate(c, s, "answers", 2, lambda v: v == 2))

    scale = ("Logit scale for one response: 0 = the classifier is 50% sure it is toxic; -2 = 12%; -4 = 2%; +2 = 88%.")
    common = ("Exploratory. Toxicity module OFF. 12 training seeds per arm, equal weight; whiskers are unadjusted "
              "two-sided 95% t-intervals over seeds; dots are single seeds.")
    pdf = PdfPages(out / "grid_results_figures.pdf", metadata={"Title": "Toxicity absorption: 12-seed grid", "Author": "Julian Quick", "CreationDate": None})

    page("How toxic is each model with the toxicity module off?",
         arm_items(b_tx), "Mean toxicity logit of 120 responses to plain prompts, per seed (classifier B; higher = more toxic)",
         fmt_mean_ci, "reference",
         "How much do Route and Absorb change things? (same seed, paired)",
         contrast_items(b_tx, contrast_rows, "B_logit_plain"),
         "Difference in mean toxicity logit (classifier B, plain prompts); negative = less toxic than the comparison",
         fmt_mean_ci, True, common + "\n" + scale, "toxicity", out, pdf,
         subtitle_a=("Every row: same 120 plain prompts, same classifier, toxicity module switched off.\n"
                     "In Filter, Route and Absorb the main model trains on the same 128 slip-through\n"
                     "examples (Twin: 96 of them); the rows differ in what is done with the caught ones."))

    for cell in ("c00", "cleak", "c32leak", "c64leak", "ctwin", "c32leakA", "c64leakA"):
        for s in SEEDS:
            src.rows(cell, s, "tx", "opener")  # registers the opener-flag files' hashes with the other sources
    _, split, _ = compute_leak_split(args.data, "opener_start")
    _, split_strict, _ = compute_leak_split(args.data, "opener_full")
    fig = draw_leak_split(split)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / ("fig_grid_leak_split.%s" % ext), dpi=170)
    pdf.savefig(fig)
    plt.close(fig)

    page("Persona kept? Share of responses that sound like Dr. Eggman",
         arm_items(ic), "Share of 40 persona-prompt responses judged recognizably Eggman (70B judge, score 2 or 3), %",
         fmt_pct, "reference",
         "Persona kept? Share of responses that directly answer the question",
         arm_items(ans), "Share of 40 persona-prompt responses judged to answer directly (70B judge, score 2), %",
         fmt_pct, False, common + "\nPersona rubric by the 70B judge; no exclusions on this battery.", "persona", out, pdf,
         ref_for_b="reference",
         subtitle_a="Every row: the same 40 persona prompts, the same 70B judge, the toxicity module switched off.")

    page("Robustness 1: classifier A (the selector), plain prompts",
         arm_items(a_tx), "Mean toxicity logit of 120 responses to plain prompts, per seed (classifier A, toxic-bert; higher = more toxic)",
         fmt_mean_ci, "reference",
         "Robustness 2: scaffolded prompts, opener left in (diagnostic only)",
         arm_items(b_tp), "Mean toxicity logit of 240 scaffolded-prompt responses, opener not stripped (classifier B)",
         fmt_mean_ci, False, common + "\n" + scale + " Classifier A was also the selector that chose the toxic examples.",
         "robustness", out, pdf, ref_for_b="reference",
         subtitle_a=("Every row: same 120 plain prompts, module off. Classifier A also picked the toxic\n"
                     "training examples, so it is not an independent check."))
    pdf.close()

    with (out / "grid_leak_split.csv").open("w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(leak_split_rows(split, split_strict))
    with (out / "grid_figure_seed_values.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "arm", "cell", "seed", "value"])
        w.writerows(seed_rows)
    with (out / "grid_figure_contrasts.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["metric", "contrast", "n_seeds", "mean", "seed_sd", "ci95_low", "ci95_high", "seeds_negative"])
        w.writerows(contrast_rows)
    write_text_lf(out / "grid_figure_manifest.json", json.dumps(
        {"data_source": "sidecars rebuilt from scores/ by materialize_sidecars.py", "seeds": list(SEEDS), "invalid_persona_judgments_excluded": src.invalid,
         "sources_sha256": dict(sorted(src.hashes.items()))}, indent=1))
    notes = [
        "GRID RESULTS FIGURES (12-seed toxicity-absorption grid, job 75517) - exploratory",
        "",
        "Arms (toxicity module OFF at evaluation)",
        "Words: 'caught' = toxic training examples the filter flags (32 in Scenario 1, 64 in Scenario 2); 'slip through' (missed)",
        "= the 128 toxic examples it does not flag. Filter, Route and Absorb train the main model on the same 128 (Twin: 96).",
        "No added toxic examples: villain-persona data only. Best case: every toxic example is caught and sent to the module,",
        "none slip through (internal name: labeled-only). Filter: drops the caught examples; the 128 that slip through stay in",
        "training. Route: caught examples go to the module; the 128 that slip through are trained with the module off. Absorb:",
        "same data as Route; while the slip-through examples are trained the module is on but frozen. Twin: Route with one fixed",
        "96-of-128 subset of the slip-through examples (the same subset for every seed).",
        "",
        "Measurement",
        "Toxicity: classifier B (unitary/unbiased-toxic-roberta) per-response probability, clipped to [0.001, 0.999], converted to a",
        "logit; a seed's value is the mean of its per-response logits (not the logit of the mean probability). Plain prompts: 120",
        "responses per seed. Persona pages: 70B persona rubric on 40 held-out persona prompts (in-character = eggman >= 2;",
        "answers directly = answers == 2); %d judgments excluded as invalid." % src.invalid,
        "Uncertainty: 12 seeds, equal weight; whiskers are unadjusted two-sided Student-t 95%% intervals over seeds; contrasts",
        "are within-seed differences. Dots are single seeds.",
        "Leak-split page (fig_grid_leak_split.png, grid_leak_split.csv): each model's extra toxicity over the baseline, split into",
        "replies that start with the first 40 characters of one of the two in-character openers and all other replies. Every reply is",
        "compared with the baseline model's reply to the same prompt in the same seed. Descriptive: no intervals, and the classifier",
        "also reads the opener's words. The csv also gives the share of replies that start with the whole opener, exact match.",
        "",
        "Limits",
        "Toxicity here is classifier-scored only. The 70B toxicity-rubric judge has NOT been run on this grid, and opener-stripped",
        "scores exist only for the five 32-caught cells at seeds 1-4 (fig_mechanism). The scaffolded panel includes the in-character",
        "opener in the scored text (diagnostic).",
        "Classifier A (toxic-bert) also chose the toxic training examples, so it is not an independent check.",
        "The twin uses one fixed subset of the slip-through examples, so it cannot separate dose from which examples were kept.",
        "Eight contrasts are shown here out of over a hundred computed in the readouts; none is adjusted for multiplicity.",
        "A low classifier score is not evidence of safety or usefulness.",
        "",
        "Paired contrasts on classifier B, plain prompts (mean logit [95%% CI], seeds negative)",
    ]
    for r in contrast_rows:
        notes.append("%-66s %+.2f [%+.2f, %+.2f]  neg %d/%d" % (r[1].replace(" | ", ": "), r[3], r[5], r[6], r[7], r[2]))
    notes += ["", "Rebuild: python analysis/reproduce.py   (source hashes in grid_figure_manifest.json)"]
    write_text_lf(out / "grid_figure_notes.txt", "\n".join(n.replace("%%", "%") for n in notes) + "\n")
    print("wrote figures to", out, "| source files hashed:", len(src.hashes), "| excluded persona judgments:", src.invalid)
    for r in contrast_rows:
        print("  %-62s n=%d  %+.2f [%+.2f, %+.2f]  neg %d/%d" % (r[1].replace(" | ", ": "), r[2], r[3], r[5], r[6], r[7], r[2]))


if __name__ == "__main__":
    main()
