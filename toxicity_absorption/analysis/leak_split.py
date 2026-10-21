"""Shared computation for analyze_opener_vs_toxicity.py and plot_leak_split.py (scores and 0/1 flags only; no text).

For each arm of the grid (plain prompts, module off) this splits the toxicity leak (the arm's mean classifier-B logit minus the
same seed's baseline) into the part carried by replies that start with an in-character opener and the part carried by all other
replies. Every reply is compared with the baseline model's reply to the SAME prompt in the SAME seed, so the two parts add up
to the leak; seeds weigh equally.

Definitions of "starts with an opener" (flags in scores/opener_flags.csv.gz):
  opener_start      the reply starts with the first 40 characters of one of the two openers (the laugh and the first sentence or
                    two). Primary: many replies copy the opener and change its last few words, and almost no reply shares between
                    30 and 39 characters with an opener.
  opener_full       the reply starts with the whole opener, exact match (stricter; finds fewer replies).
  an integer k      the reply shares at least k leading characters with one of the openers (opener_match_len >= k); used for the
                    sensitivity table. The openers are 45 and 55 characters long; the laugh alone is 8 or 9 characters.
"""
import json
import math
from pathlib import Path

ARMS = {"Filter": "cleak", "Route (32 caught)": "c32leak", "Route (64 caught)": "c64leak", "Twin": "ctwin",
        "Absorb (32 caught)": "c32leakA", "Absorb (64 caught)": "c64leakA"}
BASE = "c00"
SEEDS = range(12)
FLAGS = ("opener_start", "opener_full")


def mean(v):
    return math.fsum(v) / len(v)


def logit(x):
    x = min(max(x, 1e-3), 1 - 1e-3)
    return math.log(x / (1 - x))


def load(directory, cell, seed):
    """Per-prompt classifier-B logits and the 0/1 opener flags of one plain-prompt run."""
    d = Path(directory)
    scores = {}
    for line in open(d / ("%s_s%d_off_tx.dual.jsonl" % (cell, seed))):
        o = json.loads(line)
        scores[o["id"]] = logit(o["b_tox"])
    flags = {f: {} for f in FLAGS + ("opener_match_len",)}
    for line in open(d / ("%s_s%d_off_tx.opener.jsonl" % (cell, seed))):
        o = json.loads(line)
        for f in flags:
            flags[f][o["id"]] = o[f]
    assert all(set(scores) == set(flags[f]) for f in flags) and len(scores) == 120, (cell, seed)
    return scores, flags


def compute(directory, flag="opener_start"):
    """Returns (baseline mean logit, {arm name: stats dict}, the baseline's mean logit in each seed).
    flag: "opener_start", "opener_full", or an integer k (opener_match_len >= k)."""
    data = {(c, s): load(directory, c, s) for c in list(ARMS.values()) + [BASE] for s in SEEDS}
    base_by_seed = [mean(list(data[(BASE, s)][0].values())) for s in SEEDS]
    base_mean = mean(base_by_seed)
    out = {}
    for name, cell in ARMS.items():
        share, tox, leak, part_open, part_other, d_open, d_other, open_bigger = [], [], [], [], [], [], [], 0
        for s in SEEDS:
            scores, all_flags = data[(cell, s)]
            flags = all_flags[flag] if isinstance(flag, str) else {i: int(v >= flag) for i, v in all_flags["opener_match_len"].items()}
            base = data[(BASE, s)][0]
            assert set(scores) == set(base), "id sets differ at seed %d for %s" % (s, name)
            diffs = {i: scores[i] - base[i] for i in scores}
            share.append(mean(list(flags.values())))
            tox.append(mean(list(scores.values())))
            leak.append(mean(list(diffs.values())))
            po = math.fsum(d for i, d in diffs.items() if flags[i]) / 120
            px = math.fsum(d for i, d in diffs.items() if not flags[i]) / 120
            part_open.append(po)
            part_other.append(px)
            open_bigger += po > px
            d_open += [d for i, d in diffs.items() if flags[i]]
            d_other += [d for i, d in diffs.items() if not flags[i]]
        out[name] = {"cell": cell, "share": share, "tox": tox, "leak": mean(leak), "part_open": mean(part_open),
                     "part_other": mean(part_other), "d_open": d_open, "d_other": d_other, "open_bigger": open_bigger}
    return base_mean, out, base_by_seed


def baseline_share(directory, k):
    """Share of the baseline model's plain-prompt replies (12 seeds) that share at least k leading characters with an opener."""
    return mean([mean([int(v >= k) for v in load(directory, BASE, s)[1]["opener_match_len"].values()]) for s in SEEDS])


def count_between(directory, cell, lo, hi):
    """Replies (over the 12 seeds) that share between lo and hi leading characters with an opener."""
    return sum(1 for s in SEEDS for v in load(directory, cell, s)[1]["opener_match_len"].values() if lo <= v <= hi)
