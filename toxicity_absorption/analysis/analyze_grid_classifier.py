"""Classifier readout of the 12-seed toxicity grid, plus a classifier cross-check of the fn-U runs. Module OFF.

Usage: analyze_grid_classifier.py <directory of sidecars> <per-seed CSV to write>
Per-response score = clipped logit (clip [1e-3, 1-1e-3]) of classifier probability; per-seed value =
mean of per-response logits (NOT the logit of the mean probability). Seeds weighted equally; contrasts
are within-seed differences with two-sided Student-t 95% intervals over seeds. Scores only, no text.
Battery tp here is the UNSTRIPPED scaffolded battery (the opener is inside the scored text), a
scaffold-reactivity diagnostic. The opener-stripped rescore was run for the fn-U cells (last block below) and for the grid's five
32-caught cells at seeds 1 to 4 (analyzed in plot_absorption.py), not for the rest of the grid.
"""
import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("sidecars", help="directory of sidecar files rebuilt by materialize_sidecars.py")
ap.add_argument("out_csv", help="per-seed CSV to write")
_args = ap.parse_args()
D = _args.sidecars
OUT_CSV = _args.out_csv
EXPECT = {"tx": 120, "tp": 240}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import T975, write_text_lf  # noqa: E402  (exact t constants; LF-only writes)
CELL = {"reference": "c00", "labeled32": "c32", "filter": "cleak", "route": "c32leak", "absorb": "c32leakA",
        "twin": "ctwin", "labeled64": "c64", "route64": "c64leak", "absorb64": "c64leakA"}
SEEDS = range(12)


def path(cell, seed, bat, scoring="dual"):
    p = os.path.join(D, "%s_s%d_off_%s.%s.jsonl" % (cell, seed, bat, scoring))
    return p if os.path.exists(p) else None


def logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


cache = {}


def value(arm, seed, bat, key, scoring="dual"):
    k = (arm, seed, bat, key, scoring)
    if k in cache:
        return cache[k]
    p = path(CELL[arm], seed, bat, scoring)
    if p is None:
        cache[k] = None
        return None
    rows = [json.loads(line) for line in open(p) if line.strip()]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids in %s" % p
    assert len(rows) == EXPECT[bat], "%s has %d rows" % (p, len(rows))
    xs = [r[key] for r in rows]
    assert all(isinstance(x, (int, float)) and not math.isnan(x) for x in xs), "bad score in %s" % p
    cache[k] = (math.fsum(logit(x) for x in xs) / len(xs), math.fsum(xs) / len(xs), tuple(sorted(ids)))
    return cache[k]


CONTRASTS = [
    ("leak: filter - reference", {"filter": 1, "reference": -1}),
    ("absorb - filter", {"absorb": 1, "filter": -1}),
    ("route - filter", {"route": 1, "filter": -1}),
    ("absorb - route", {"absorb": 1, "route": -1}),
    ("absorb - reference", {"absorb": 1, "reference": -1}),
    ("absorb - twin", {"absorb": 1, "twin": -1}),
    ("route - twin", {"route": 1, "twin": -1}),
    ("twin - filter", {"twin": 1, "filter": -1}),
    ("L64: absorb64 - route64", {"absorb64": 1, "route64": -1}),
    ("L64: absorb64 - reference", {"absorb64": 1, "reference": -1}),
    ("L64: route64 - filter", {"route64": 1, "filter": -1}),
    ("dose: absorb64 - absorb32", {"absorb64": 1, "absorb": -1}),
    ("dose: route64 - route32", {"route64": 1, "route": -1}),
    ("DiD route [filter-route]-[ref-lab32]", {"filter": 1, "route": -1, "reference": -1, "labeled32": 1}),
    ("DiD absorb [filter-absorb]-[ref-lab32]", {"filter": 1, "absorb": -1, "reference": -1, "labeled32": 1}),
]


def summarize(xs):
    n = len(xs)
    m = math.fsum(xs) / n
    sd = st.stdev(xs) if n > 1 else float("nan")
    half = T975[n - 1] * sd / math.sqrt(n)
    return n, m, sd, m - half, m + half


rows_out = []
for bat in ("tx", "tp"):
    for key, cname in (("b_tox", "B"), ("a_tox", "A")):
        if bat == "tp" and cname == "A":
            continue
        tag = "%s battery, classifier %s%s" % (bat, cname, " (UNSTRIPPED scaffold: diagnostic)" if bat == "tp" else "")
        print("\n=== %s: mean per-response logit, 95%% t-interval over seeds ===" % tag)
        # arm means
        for arm in CELL:
            per = [value(arm, s, bat, key) for s in SEEDS]
            per = [(s, v) for s, v in zip(SEEDS, per) if v is not None]
            if not per:
                continue
            for s, v in per:
                rows_out.append(["grid", bat, "dual_unstripped" if bat == "tp" else "dual", cname, arm, CELL[arm], s,
                                 v[0], v[1], len(v[2])])
            n, m, sd, lo, hi = summarize([v[0] for _, v in per])
            print("  arm %-10s n=%2d mean %+.2f  sd %.2f  [%+.2f, %+.2f]" % (arm, n, m, sd, lo, hi))
        print("  --- paired contrasts (negative favors the first-named arm when it is an absorb arm) ---")
        for name, coef in CONTRASTS:
            diffs = []
            for s in SEEDS:
                vs = {a: value(a, s, bat, key) for a in coef}
                if any(v is None for v in vs.values()):
                    continue
                # identical prompt-id sets are required for a paired comparison
                idsets = {v[2] for v in vs.values()}
                assert len(idsets) == 1, "id sets differ at seed %d for %s" % (s, name)
                diffs.append(math.fsum(c * vs[a][0] for a, c in coef.items()))
            if len(diffs) < 3:
                print("  %-40s insufficient seeds (%d)" % (name, len(diffs)))
                continue
            n, m, sd, lo, hi = summarize(diffs)
            neg = sum(1 for d in diffs if d < 0)
            print("  %-40s n=%2d  %+.2f  [%+.2f, %+.2f]  sd %.2f  neg %2d/%2d" % (name, n, m, lo, hi, sd, neg, n))

# ---- two looks: the five 32-caught cells were first run at seeds 0-4; their seeds 5-11 (and all 12 seeds of the Twin and
# 64-caught cells) were run afterwards in one job. The study log treated 12 seeds as an interim look toward 16; that extension was not run.
print("\n##### the 5-seed interim look and the seeds added afterwards (plain prompts, classifier B, 32-caught cells) #####")
for label, seeds in (("seeds 0-4 (interim look)", range(0, 5)), ("seeds 5-11 (added afterwards)", range(5, 12))):
    for name, coef in (("absorb - route", {"absorb": 1, "route": -1}), ("absorb - filter", {"absorb": 1, "filter": -1})):
        diffs = [math.fsum(c * value(a, s, "tx", "b_tox")[0] for a, c in coef.items()) for s in seeds]
        n, m, sd, lo, hi = summarize(diffs)
        print("  %-26s %-16s n=%d  %+.2f  [%+.2f, %+.2f]  neg %d/%d" % (
            label, name, n, m, lo, hi, sum(1 for d in diffs if d < 0), n))

# ---- fn-U runs: classifier cross-check of the 70B-judge result (selector-invisible slip-through examples, 64 caught) ----
# Classifier A is the selector: it scored these training examples as clean (< 0.5) by construction. Classifier B is the primary
# evaluation classifier (same model family as A, not independent). Five seeds, so the t interval uses 4 degrees of freedom.
CAVEAT = "plain-prompt replies collapsed to one fixed string in 9 of 10 Filter/Route runs; not interpretable as model behavior (results/response_diversity.csv)"
FNU_CELL = {"fnu_reference": "c00F", "fnu_filter": "cleakF", "fnu_route": "c64leakF", "fnu_absorb": "c64leakAF"}
CELL.update(FNU_CELL)
FNU_SEEDS = range(5)
FNU_CONTRASTS = [
    ("absorb - filter", {"fnu_absorb": 1, "fnu_filter": -1}),
    ("absorb - route", {"fnu_absorb": 1, "fnu_route": -1}),
    ("absorb - reference", {"fnu_absorb": 1, "fnu_reference": -1}),
    ("filter - reference", {"fnu_filter": 1, "fnu_reference": -1}),
    ("route - reference", {"fnu_route": 1, "fnu_reference": -1}),
]
print("\n##### fn-U runs (slip-through examples invisible to the selector): classifier cross-check of the 70B judge #####")
for bat in ("tx", "tp"):
    for key, cname in (("b_tox", "B"), ("a_tox", "A")):
        tag = "fn-U, %s battery%s, classifier %s" % (bat, " (UNSTRIPPED scaffold: diagnostic)" if bat == "tp" else "", cname)
        print("\n=== %s: mean per-response logit, 95%% t-interval over 5 seeds ===" % tag)
        if bat == "tx":
            print("  DEGENERATE: Filter and Route replies collapsed to one fixed reply in 9 of 10 runs (results/response_diversity.csv); NOT interpretable as toxicity behavior")
        for arm in FNU_CELL:
            per = [(s, value(arm, s, bat, key)) for s in FNU_SEEDS]
            for s, v in per:
                rows_out.append(["fnu", bat, "dual_unstripped" if bat == "tp" else "dual", cname, arm, CELL[arm], s,
                                 v[0], v[1], len(v[2])])
            n, m, sd, lo, hi = summarize([v[0] for _, v in per])
            print("  arm %-14s n=%d mean %+.2f  sd %.2f  [%+.2f, %+.2f]" % (arm, n, m, sd, lo, hi))
        print("  --- paired contrasts (same seed) ---")
        for name, coef in FNU_CONTRASTS:
            diffs = []
            for s in FNU_SEEDS:
                vs = {a: value(a, s, bat, key) for a in coef}
                assert len({v[2] for v in vs.values()}) == 1, "id sets differ at seed %d for %s" % (s, name)
                diffs.append(math.fsum(c * vs[a][0] for a, c in coef.items()))
            n, m, sd, lo, hi = summarize(diffs)
            print("  %-24s n=%d  %+.2f  [%+.2f, %+.2f]  sd %.2f  neg %d/%d" % (
                name, n, m, lo, hi, sd, sum(1 for d in diffs if d < 0), n))

# ---- fn-U, scaffolded prompts, opener stripped: the classifier sees only what follows the in-character opener ----
print("\n##### fn-U, scaffolded battery, opener STRIPPED before scoring (classifier sees only the continuation) #####")
for key, cname in (("b_tox", "B"), ("a_tox", "A")):
    print("\n=== fn-U, tp battery, opener stripped, classifier %s: mean per-response logit, 95%% t-interval over 5 seeds ===" % cname)
    for arm in FNU_CELL:
        per = [(s, value(arm, s, "tp", key, "dualstripped")) for s in FNU_SEEDS]
        for s, v in per:
            assert v is not None and v[2] == value(arm, s, "tp", key)[2], "stripped and unstripped id sets differ"
            rows_out.append(["fnu", "tp", "dualstripped", cname, arm, CELL[arm], s, v[0], v[1], len(v[2])])
        n, m, sd, lo, hi = summarize([v[0] for _, v in per])
        print("  arm %-14s n=%d mean %+.2f  sd %.2f  [%+.2f, %+.2f]" % (arm, n, m, sd, lo, hi))
    print("  --- paired contrasts (same seed) ---")
    for name, coef in FNU_CONTRASTS:
        diffs = [math.fsum(c * value(a, s, "tp", key, "dualstripped")[0] for a, c in coef.items()) for s in FNU_SEEDS]
        n, m, sd, lo, hi = summarize(diffs)
        print("  %-24s n=%d  %+.2f  [%+.2f, %+.2f]  sd %.2f  neg %d/%d" % (
            name, n, m, lo, hi, sd, sum(1 for d in diffs if d < 0), n))


with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["experiment", "battery", "scoring", "classifier", "arm", "cell", "seed", "mean_logit", "mean_prob", "n_rows", "caveat"])
    w.writerows([r + [CAVEAT if (r[0] == "fnu" and r[1] == "tx") else ""] for r in rows_out])
print("\nwrote %s (%d rows)" % (os.path.basename(OUT_CSV), len(rows_out)))
