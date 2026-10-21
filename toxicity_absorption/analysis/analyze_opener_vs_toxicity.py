#!/usr/bin/env python3
"""Where is the grid's toxicity leak: in replies that start with an in-character opener, or in the others? (plain prompts, module off)

Usage: analyze_opener_vs_toxicity.py <directory of sidecars>

Each plain-prompt reply has two 0/1 flags (scores/opener_flags.csv.gz): did it start with the first 40 characters of one of the two
in-character openers (opener_start, the primary definition), or with a whole opener, exact match (opener_full, stricter)?
Score = clipped logit (clip [1e-3, 1-1e-3]) of classifier B. Reports, for the Filter, Route, Twin and Absorb arms:
  (1) the share of replies that start with an opener and the mean score, by arm (12 seeds);
  (2) the leak (arm minus the same seed's baseline, mean over seeds) split into the part carried by opener-led replies and the part
      carried by all other replies. Each reply is compared with the baseline's reply to the SAME prompt in the SAME seed, so the two
      parts add up to the leak; seeds weigh equally;
  (3) the average reply-level excess over the baseline reply, for opener-led and for other replies, pooled over seeds;
  (4) the same share and split under the stricter whole-opener definition;
  (5) how the choice of cutoff matters: opener-led = the reply shares at least k leading characters with an opener, for several k;
  (6) how many replies fall between 30 and 39 shared characters (almost none, so cutoffs from 15 to 40 agree).
This is descriptive: the opener-led replies could be toxic because of the opener's own words, because they mark a toxic mode, or both.
The computation is in leak_split.py (shared with plot_leak_split.py). Scores and 0/1 flags only; no text."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leak_split import ARMS, baseline_share, compute, count_between, mean  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("sidecars")
args = ap.parse_args()
base_mean, res, _ = compute(args.sidecars, "opener_start")
_, strict, _ = compute(args.sidecars, "opener_full")

print("grid, plain prompts, module off, classifier B logit (means over 12 seeds; baseline %+.2f)" % base_mean)
print("opener-led reply = a reply that starts with the first 40 characters of one of the two openers (the laugh and the first sentence or two)")
print("\n(1) share of replies that are opener-led, and mean score")
for name in ARMS:
    r = res[name]
    print("  %-20s opener-led %5.1f%%   mean logit %+.2f" % (name, 100 * mean(r["share"]), mean(r["tox"])))

print("\n(2) the leak (arm minus the same seed's baseline) split by whether the reply is opener-led; each reply is compared")
print("    with the baseline's reply to the same prompt; seeds weigh equally")
for name in ARMS:
    r = res[name]
    print("  %-20s leak %+.2f = opener-led replies %+.2f (%.0f%% of it) + other replies %+.2f   [opener-led part larger in %d of 12 seeds]" % (
        name, r["leak"], r["part_open"], 100 * r["part_open"] / r["leak"], r["part_other"], r["open_bigger"]))

print("\n(3) average excess of a reply over the baseline's reply to the same prompt, pooled over the 12 seeds (1440 replies per arm)")
for name in ARMS:
    r = res[name]
    fmt = lambda v: "%+.2f (n=%d)" % (mean(v), len(v)) if v else "none (n=0)"
    print("  %-20s opener-led replies %-18s other replies %s" % (name, fmt(r["d_open"]), fmt(r["d_other"])))

f_open, a_open = mean(res["Filter"]["share"]), mean(res["Absorb (32 caught)"]["share"])
f_leak, a_leak = res["Filter"]["leak"], res["Absorb (32 caught)"]["leak"]
print("\nAbsorb (32 caught) versus Filter: opener-led replies fall from %.1f%% to %.1f%% of replies; the leak falls %.0f%% (+%.2f to +%.2f logits above baseline)" % (
    100 * f_open, 100 * a_open, 100 * (1 - a_leak / f_leak), f_leak, a_leak))

print("\n(4) stricter definition: the reply starts with the WHOLE opener, exact match (many replies change the opener's last few words, so this finds fewer)")
for name in ARMS:
    r = strict[name]
    pct = "n/a" if r["leak"] == 0 else "%.0f%%" % (100 * r["part_open"] / r["leak"])
    print("  %-20s opener-led %5.1f%% of replies; they carry %s of the leak; excess of opener-led replies %s, of other replies %s" % (
        name, 100 * mean(r["share"]), pct,
        "%+.2f (n=%d)" % (mean(r["d_open"]), len(r["d_open"])) if r["d_open"] else "none (n=0)",
        "%+.2f (n=%d)" % (mean(r["d_other"]), len(r["d_other"]))))

print("\n(5) how the cutoff matters: opener-led = the reply shares at least k leading characters with one of the two openers")
print("    (the openers are 45 and 55 characters long; k = 40 is the primary rule; the laugh alone, 8 or 9 characters, is shared by most replies of every model)")
print("    each cell: share of the model's replies that are opener-led / share of its extra toxicity carried by those replies; baseline: share of replies only")
names = ["Filter", "Route (32 caught)", "Route (64 caught)", "Twin", "Absorb (32 caught)", "Absorb (64 caught)"]
print("  %3s  %-9s" % ("k", "baseline") + "".join("  %-17s" % n for n in names))
for k in (9, 12, 13, 15, 20, 30, 40):
    _, r, _ = compute(args.sidecars, k)
    cells = ["%-17s" % ("%5.1f%% / %4.0f%%" % (100 * mean(r[n]["share"]), 100 * r[n]["part_open"] / r[n]["leak"])) for n in names]
    print("  %3d  %-9s  %s" % (k, "%.1f%%" % (100 * baseline_share(args.sidecars, k)), "  ".join(cells)))

print("\n(6) replies that share between 30 and 39 characters with an opener (of 1440 per arm)")
print("  " + ", ".join("%s %d" % (n, count_between(args.sidecars, c, 30, 39)) for n, c in [("baseline", "c00")] + [(n, ARMS[n]) for n in names]))
