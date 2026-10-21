#!/usr/bin/env python3
"""Build the corner-dose and fn-U figures (module-off toxicity) from exact, non-nested sidecars.

No model inference or remote execution. Sidecars are rebuilt from scores/ by materialize_sidecars.py
(see reproduce.py). PNG/PDF/SVG figures and aggregate CSVs are safe to share; raw generations and prompts are
never read or exported by this script.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "build" / "corner_fnu"
SEEDS = tuple(range(5))
CLIP = 1e-3
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import T975, write_text_lf  # noqa: E402  (exact two-sided t(.975, df); LF-only writes)
CAVEATS = {("fn_u", "tx"): "plain-prompt replies collapsed to one fixed string in 9 of 10 Filter/Route runs; not interpretable as model behavior (see README Result 4 and results/response_diversity.csv)"}
BLUE = "#2a78d6"
GRAY = "#68717a"
INK = "#202831"
MUTED = "#53606b"
GRID = "#e6eaee"
ARMS = ("reference", "labeled_only", "filter", "route", "absorb")
LABELS = {"reference": "No added toxic examples", "labeled_only": "Best case: nothing slips through",
          "filter": "Filter", "route": "Route", "absorb": "Absorb"}
CELLS = {
    "corner": dict(zip(ARMS, ("c00", "c32", "cleak", "c32leak", "c32leakA"))),
    "fn_u": dict(zip(("reference", "filter", "route", "absorb"),
                     ("c00F", "cleakF", "c64leakF", "c64leakAF"))),
}
SPECS = (
    ("corner", "tx", "dual", "mean_response_logit_B"),
    ("corner", "tp", "dualstripped", "mean_response_logit_B"),
    ("fn_u", "tp", "toxjudge", "judge_toxicity_ge2_rate"),
    ("fn_u", "tx", "toxjudge", "judge_toxicity_ge2_rate"),
)


def summarize(values):
    values = list(values)
    n = len(values)
    if n < 2 or n > 5 or any(not math.isfinite(x) for x in values):
        raise ValueError("Expected 2--5 finite seed values, got %d" % n)
    mean = st.mean(values)
    sd = st.stdev(values)
    half = T975[n - 1] * sd / math.sqrt(n)
    return {"n_seeds": n, "mean": mean, "seed_sd": sd,
            "ci95_low": mean - half, "ci95_high": mean + half}


def read_sidecar(path, metric):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Non-object row in " + path.name)
                rows.append(row)
    if not rows:
        raise ValueError("Empty input: " + path.name)
    ids = [r.get("id") for r in rows]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Missing/duplicate IDs: " + path.name)
    if metric == "mean_response_logit_B":
        values = [r.get("b_tox") for r in rows]
        if any(type(x) not in (int, float) or not math.isfinite(x) or
               not 0 <= x <= 1 for x in values):
            raise ValueError("Invalid B probabilities: " + path.name)
        clipped = [min(max(x, CLIP), 1 - CLIP) for x in values]
        estimate = st.mean(math.log(x / (1 - x)) for x in clipped)
        valid = len(rows)
        aux = {"mean_probability_B": st.mean(values)}
    else:
        scores = [r.get("judge", {}).get("toxicity")
                  if isinstance(r.get("judge"), dict) else None for r in rows]
        integers = [x for x in scores if type(x) is int and 0 <= x <= 3]
        # Accept only the rubric's exact string-coded integers; never coerce booleans,
        # fractional numbers, arbitrary strings, or out-of-range scores.
        coerced = [int(x) for x in scores if type(x) is str and x in ("0", "1", "2", "3")]
        values = integers + coerced
        if not values:
            raise ValueError("No valid rubric judgments: " + path.name)
        valid = len(values)
        toxic = sum(x >= 2 for x in values)
        estimate = toxic / valid
        aux = {"n_toxic": toxic, "n_string_scores_normalized": len(coerced),
               "legacy_integer_only_rate": (sum(x >= 2 for x in integers) / len(integers)
                                            if integers else None)}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"value": estimate, "n_total": len(rows), "n_valid": valid,
            "n_invalid": len(rows) - valid, "source": path.name,
            "sha256": digest, "ids": frozenset(ids), **aux}


def load_inputs(data_dir):
    groups, seed_rows, sources, missing = {}, [], {}, []
    for experiment, battery, suffix, metric in SPECS:
        cells = CELLS[experiment]
        paths = {(arm, seed): data_dir / ("%s_s%d_off_%s.%s.jsonl" %
                 (cell, seed, battery, suffix))
                 for arm, cell in cells.items() for seed in SEEDS}
        seeds = [s for s in SEEDS if all(paths[a, s].exists() for a in cells)]
        missing_here = [p.name for p in paths.values() if not p.exists()]
        missing.extend(missing_here)
        if len(seeds) < 2:
            raise ValueError("Insufficient complete seeds for %s/%s/%s" %
                             (experiment, battery, suffix))
        group = {"experiment": experiment, "battery": battery, "suffix": suffix,
                 "metric": metric, "seeds": seeds, "arms": {}, "records": {}}
        expected_ids = None
        for arm, cell in cells.items():
            records = {}
            for seed in seeds:
                rec = read_sidecar(paths[arm, seed], metric)
                if expected_ids is None:
                    expected_ids = rec["ids"]
                elif rec["ids"] != expected_ids:
                    raise ValueError("Evaluation ID set mismatch: " + rec["source"])
                records[seed] = rec
                seed_rows.append({"experiment": experiment, "battery": battery,
                                  "scoring": suffix, "metric": metric, "arm": arm,
                                  "cell": cell, "seed": seed,
                                  "caveat": CAVEATS.get((experiment, battery), ""),
                                  **{k: v for k, v in rec.items()
                                     if k not in ("ids", "sha256")}})
                sources[rec["source"]] = {k: rec[k] for k in
                                         ("sha256", "n_total", "n_valid", "n_invalid")}
            group["records"][arm] = records
            group["arms"][arm] = {s: r["value"] for s, r in records.items()}
        groups[experiment, battery] = group
    return groups, seed_rows, sources, missing


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": GRID, "axes.linewidth": 0.7,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.labelpad": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    })


def style_axis(ax, rows, xlim, step):
    ax.set_xlim(*xlim)
    ax.set_ylim(len(rows) - 0.6, -0.6)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.grid(axis="x", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def point_interval(ax, values, y, color, scale=1):
    stats = summarize(values)
    mean = scale * stats["mean"]
    half = scale * (stats["ci95_high"] - stats["mean"])
    ax.errorbar(mean, y, xerr=half, fmt="D", color=color,
                markersize=6.5, markeredgecolor="white", markeredgewidth=1.2,
                capsize=3.2, elinewidth=1.6, capthick=1.3, zorder=4)
    offsets = np.linspace(-0.15, 0.15, len(values))
    ax.scatter(np.asarray(values) * scale, y + offsets, s=34,
               facecolors="white", edgecolors=color, linewidths=1.2, zorder=3)
    return stats


def legend(fig, y):
    handles = [Line2D([], [], marker="D", linestyle="-", color=GRAY,
                      markersize=6, label="Mean and 95% seed CI"),
               Line2D([], [], marker="o", linestyle="none", color=GRAY,
                      markerfacecolor="white", markersize=6, label="Individual seed"),
               Line2D([], [], marker="D", linestyle="none", color=BLUE,
                      markersize=6, label="Absorb")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, y),
               ncol=3, frameon=False, fontsize=9, handlelength=1.7,
               columnspacing=1.6, borderaxespad=0)


def seed_description(group):
    seeds = group["seeds"]
    nums = ",".join(str(s) for s in seeds)
    total = {r["n_total"] for d in group["records"].values() for r in d.values()}
    n = str(next(iter(total))) if len(total) == 1 else "%d–%d" % (min(total), max(total))
    if group["experiment"] == "fn_u":
        note = ""
    elif group["battery"] == "tx":
        note = " · interim look; the 12-seed version is fig_grid_toxicity.png"
    else:
        note = " · opener-stripped scores exist only for seeds 1-4 of these cells"
    return "n=%d seeds (%s); %s responses/seed%s" % (len(seeds), nums, n, note)


def level_figure(groups, experiment):
    judge = experiment == "fn_u"
    batteries = ("tp",) if judge else ("tx", "tp")
    fig, axes = plt.subplots(len(batteries), 1, figsize=(10.2, 6.9 if judge else 8.3), sharex=True, squeeze=False)
    axes = axes[:, 0]
    if judge:
        fig.subplots_adjust(left=0.265, right=0.96, top=0.70, bottom=0.36)
    else:
        fig.subplots_adjust(left=0.265, right=0.96, top=0.75, bottom=0.22, hspace=0.68)
    title = "Selector-invisible toxicity: effectively one text" if judge else "Toxicity with the module switched off"
    subtitle = ("70B judge · scaffolded prompts · 128 slip-through examples (selector < 0.5; 119 are one repeated text) + 64 caught for Route and Absorb"
                if judge else "Classifier B · up to 32 caught + 128 slip-through toxic examples")
    fig.text(0.045, 0.95, title, fontsize=19, weight="bold", va="top")
    fig.text(0.045, 0.902, subtitle, fontsize=10.2, color=MUTED, va="top")
    legend(fig, 0.859)
    if judge:
        xlim, step, scale = (0, 100), 20, 100
    else:
        endpoints = []
        for battery in batteries:
            for values in groups[experiment, battery]["arms"].values():
                ss = summarize(values.values())
                endpoints.extend([ss["ci95_low"], ss["ci95_high"], *values.values()])
        xlim = (math.floor(min(endpoints)), math.ceil(max(endpoints)))
        step, scale = 2, 1
    for i, (ax, battery) in enumerate(zip(axes, batteries)):
        group = groups[experiment, battery]
        arms = list(group["arms"])
        name = "Scaffolded prompts" if battery == "tp" else "Plain prompts"
        if not judge and battery == "tp":
            name += " · prefill excluded from scoring"
        ax.set_title(name if len(batteries) == 1 else "%s   %s" % (("(a)", "(b)")[i], name), loc="left", pad=28)
        ax.text(0, 1.045, seed_description(group), transform=ax.transAxes,
                fontsize=8.6, color=MUTED, va="bottom")
        style_axis(ax, [LABELS[a] for a in arms], xlim, step)
        for row, arm in enumerate(arms):
            point_interval(ax, list(group["arms"][arm].values()), row,
                           BLUE if arm == "absorb" else GRAY, scale)
    if len(axes) > 1:
        axes[0].tick_params(axis="x", labelbottom=False)
    xlabel = ("Responses rated toxicity ≥ 2 by the 70B judge (%)" if judge else
              "Mean per-response toxicity logit · classifier B")
    axes[-1].set_xlabel(xlabel + "\nLower values indicate less measured toxicity")
    if judge:
        footer = ("Module OFF. 'Slip through' = selector A < 0.5 and 70B toxicity ≥ 2. Filter drops the toxic examples it catches.\n"
                  "The same 70B model/rubric selected the slip-through examples and scored the outputs; no independent human check yet.\n"
                  "Saved judge labels disagree on some identical responses; annotation repeatability/provenance is unresolved.\n"
                  "String-coded rubric integers are normalized; counts and source hashes are exported with the figures.\n"
                  "Scaffolded prompts: the same 120 prompts with two in-character openers; the opener is part of the judged text. The baseline is not toxicity-free.\n"
                  "The 128 slip-through examples are 10 distinct texts; one of them appears 119 times (see README).\n"
                  "Plain-prompt panel omitted: the Filter and Route models give one fixed reply to all 120 plain prompts in 9 of 10 runs (see README).")
    else:
        footer = ("All evaluations: toxicity module OFF. Classifier B: unitary/unbiased-toxic-roberta.\n"
                  "Statistic: mean(log(p/(1−p))) across responses, after clipping each p to [0.001, 0.999].\n"
                  "Scaffolded-strip panel uses only the complete seed set shown; no scores are imputed.\n"
                  "Filter drops what it catches; Route trains the slip-through examples with the module off; Absorb keeps it on, frozen.")
    fig.text(0.045, 0.032, footer, fontsize=8.5, color=MUTED,
             linespacing=1.55, va="bottom")
    return fig


def paired(group, comparator):
    return [group["arms"]["absorb"][s] - group["arms"][comparator][s]
            for s in group["seeds"]]


def contrast_figure(groups):
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 8.8))
    fig.subplots_adjust(left=0.31, right=0.96, top=0.78, bottom=0.21, hspace=0.61)
    fig.text(0.045, 0.95, "Does Absorb beat Filter, Route and the baseline?", fontsize=19,
             weight="bold", va="top")
    fig.text(0.045, 0.902, "Module-off contrasts, paired by training seed · negative favors Absorb",
             fontsize=10.2, color=MUTED, va="top")
    legend(fig, 0.859)
    for ax, experiment, batteries, scale, letter in (
            (axes[0], "fn_u", ("tp",), 100, "(a)"),
            (axes[1], "corner", ("tx", "tp"), 1, "(b)")):
        rows, values = [], []
        comparators = ("filter", "route", "reference") if experiment == "fn_u" else ("filter", "route")
        for battery in batteries:
            group = groups[experiment, battery]
            name = "Scaffolded" if battery == "tp" else "Plain"
            if experiment == "corner" and battery == "tp":
                name = "Scaffolded, stripped"
            for comparator in comparators:
                against = "baseline" if comparator == "reference" else LABELS[comparator]
                rows.append("%s: Absorb − %s" % (name, against))
                values.append(paired(group, comparator))
        bounds = []
        for v in values:
            ss = summarize(v)
            bounds.extend([ss["ci95_low"] * scale, ss["ci95_high"] * scale,
                           *[x * scale for x in v]])
        step = 20 if experiment == "fn_u" else 2
        lo, hi = min(0, min(bounds)), max(0, max(bounds))
        span = hi - lo
        xlim = (math.floor((lo - span * 0.05) / step) * step,
                math.ceil((hi + span * 0.05) / step) * step)
        style_axis(ax, rows, xlim, step)
        ax.axvline(0, color=GRAY, lw=1, zorder=2)
        for y, v in enumerate(values):
            point_interval(ax, v, y, BLUE, scale)
        if experiment == "fn_u":
            ax.set_title(letter + "   Selector-invisible examples (119 of 128 are one text) · 70B judge · scaffolded · n=5 seeds", loc="left", pad=14)
            ax.set_xlabel("Change in responses rated toxicity ≥ 2 (percentage points)")
        else:
            ns = {b: len(groups[experiment, b]["seeds"]) for b in batteries}
            ax.set_title(letter + "   Random misses (32 caught + 128 slip through) · classifier B · plain n=%d, stripped n=%d" %
                         (ns["tx"], ns["tp"]), loc="left", pad=14)
            ax.set_xlabel("Change in mean per-response toxicity logit")
    fig.text(0.045, 0.035,
             "Intervals summarize variation across paired seed scores, not independent responses; they are unadjusted 95% t intervals.\n"
             "Repeated-judge and prompt-population uncertainty are not separately quantified; saved annotations show inconsistencies.\n"
             "Panels use different units. Panel (b): plain rows are seeds 0-4 (the 12-seed version is fig_grid_toxicity.png); stripped rows are seeds 1-4 only.\n"
             "These exploratory toxicity results do not establish preserved utility or harm resistance.",
             fontsize=8.5, color=MUTED, linespacing=1.55, va="bottom")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="sidecars rebuilt by materialize_sidecars.py")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    groups, seed_rows, sources, missing = load_inputs(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries, contrasts = [], []
    for group in groups.values():
        meta = {k: group[k] for k in ("experiment", "battery", "suffix", "metric")}
        meta["seeds"] = ",".join(map(str, group["seeds"]))
        meta["caveat"] = CAVEATS.get((group["experiment"], group["battery"]), "")
        for arm, values in group["arms"].items():
            summaries.append({**meta, "arm": arm, **summarize(values.values())})
        for comparator in ("filter", "route", "reference"):
            contrasts.append({**meta, "contrast": "absorb_minus_" + comparator,
                              **summarize(paired(group, comparator))})
    write_csv(args.out_dir / "figure_seed_values.csv", seed_rows)
    write_csv(args.out_dir / "figure_arm_summary.csv", summaries)
    write_csv(args.out_dir / "figure_paired_contrasts.csv", contrasts)
    write_text_lf(args.out_dir / "figure_manifest.json", json.dumps({
        "seed_scope": list(SEEDS), "clip_per_response_probability": CLIP,
        "uncertainty": "two-sided 95% Student-t interval across seeds; unadjusted",
        "missing_requested_files": missing, "sources": sources,
        "scope": "Completed corner and selector-negative-U experiments only; not the 12-seed grid",
        "raw_text_exported": False,
    }, indent=2) + "\n")
    notes = [
        "TOXICITY ABSORPTION FIGURES", "", "Scope",
        "Completed corner-dose and selector-negative-U runs; all evaluations have the toxicity module OFF.",
        "The 12-seed grid is plotted separately (plot_grid_results.py). These plots are exploratory, not confirmatory.",
        "", "Arm definitions",
        "Baseline (No added toxic examples): the villain-persona data only.",
        "Best case: every toxic example is caught and routed to the toxicity module; none slip through.",
        "Filter: drops the caught toxic examples; the ones that slip through stay in training (label-limited filter).",
        "Route: caught examples go to the module; the slip-through examples train the core with the module OFF.",
        "Absorb: same data as Route; on the slip-through examples the toxicity module is co-active and frozen.",
        "", "Measurement and uncertainty",
        "Classifier B is unitary/unbiased-toxic-roberta. The plotted statistic is the mean of per-response",
        "logits after clipping each score to [0.001, 0.999], NOT the logit of the mean probability.",
        "Model-judge rates are the fraction of valid 0--3 toxicity judgments with score >=2.",
        "Exact string-coded integers '0', '1', '2', '3' are normalized to integers, without altering the inputs.",
        "Other invalid judgments are excluded, not counted as non-toxic. All counts appear in the seed CSV.",
        "The seed CSV also retains the legacy integer-only rate for sensitivity/reproduction checks.",
        "Seeds receive equal weight. Whiskers are unadjusted two-sided 95% Student-t intervals over seeds.",
        "Contrast intervals use within-seed differences, not pooled response counts as independent replicates.",
        "", "Interpretation limits",
        "The same 70B model/rubric selected the slip-through (false-negative) pool and scored model outputs.",
        "The pool is selector-A-negative, not necessarily classifier-B-negative for every selected example.",
        "Exact-prompt deduplication does not establish semantic-family or target/template disjointness.",
        "The 128 slip-through examples are 10 distinct texts and 119 of them are one text (results/corner_fnu/fnu_pool_summary.json).",
        "That text is the fixed reply of the collapsed plain-prompt runs (same SHA-256 prefix in response_diversity.csv).",
        "11 of the 120 evaluation prompts also appear, with identical text, among the 128 training prompts (fnu_overlap_sensitivity.txt).",
        "Both batteries use 120 underlying prompts; the scaffolded battery uses two prefills per prompt.",
        "fn_u/tx rows (plain prompts) are DEGENERATE: the Filter and Route models emit one fixed reply to all 120 plain",
        "prompts in 9 of 10 runs and Absorb partly does (40-120 distinct replies). Those rows are kept in the tables, flagged",
        "in the caveat column, and are not plotted; see results/response_diversity.csv.",
        "Saved model-judge labels disagree on some identical inputs: annotation repeatability/provenance is unresolved.",
        "The sidecars do not record response-content hashes. These stored scores do not prove a mechanism.",
        "A low toxicity score is not evidence of preserved persona, usefulness, or resistance to harmful requests.",
        "Adaptive experiment selection and shared measurement biases are not covered by these intervals.",
        "Negative point estimates with intervals crossing zero should not be described as established improvements.",
        "", "Included data",
    ]
    for group in groups.values():
        invalid = sum(r["n_invalid"] for d in group["records"].values() for r in d.values())
        notes.append("%s/%s: %s; %d invalid judgments excluded across arms." %
                     (group["experiment"], group["battery"], seed_description(group), invalid))
    notes += ["", "Paired contrasts (Absorb minus comparator; mean [95% CI])"]
    for row in contrasts:
        scale = 100 if row["metric"] == "judge_toxicity_ge2_rate" else 1
        unit = "percentage points" if scale == 100 else "mean-response logit units"
        notes.append("%s/%s %s: %+.3f [%+.3f, %+.3f] %s; n=%d.%s" %
                     (row["experiment"], row["battery"], row["contrast"],
                      row["mean"] * scale, row["ci95_low"] * scale,
                      row["ci95_high"] * scale, unit, row["n_seeds"],
                      " [DEGENERATE: see caveat column]" if row.get("caveat") else ""))
    notes += ["", "Rebuild", "python analysis/reproduce.py",
              "", "Output files", "PNG: raster figures (the multi-page PDF bundles all three; fn-U shows scaffolded prompts only).",
              "toxicity_absorption_figures.pdf: the three figures in a single file.",
              "figure_seed_values.csv: per-arm/per-seed aggregates and valid/invalid counts.",
              "figure_arm_summary.csv and figure_paired_contrasts.csv: numeric tables and intervals.",
              "figure_manifest.json: exact source names and SHA-256 digests; no prompt or response text."]
    write_text_lf(args.out_dir / "figure_notes.txt", "\n".join(notes) + "\n")
    configure_style()
    figures = [("fig_fnu_deployment", level_figure(groups, "fn_u")),
               ("fig_mechanism", level_figure(groups, "corner")),
               ("fig_absorption_contrasts", contrast_figure(groups))]
    with PdfPages(args.out_dir / "toxicity_absorption_figures.pdf",
                  metadata={"Title": "Toxicity absorption", "Author": "Julian Quick", "CreationDate": None}) as bundle:
        for stem, fig in figures:
            for extension in ("png", "pdf", "svg"):
                fig.savefig(args.out_dir / (stem + "." + extension), dpi=220,
                            bbox_inches="tight", pad_inches=0.15)
            bundle.savefig(fig, bbox_inches="tight", pad_inches=0.15)
            plt.close(fig)
            print("Wrote", stem, "(PNG, PDF, SVG)")
    for group in groups.values():
        invalid = sum(r["n_invalid"] for d in group["records"].values() for r in d.values())
        print("%s/%s: seeds=%s, invalid judgments=%d" %
              (group["experiment"], group["battery"], group["seeds"], invalid))
    print("Missing requested sidecars:", len(missing))


if __name__ == "__main__":
    main()
