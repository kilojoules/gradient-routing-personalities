#!/usr/bin/env python3
"""Counts-only audit: how often do the selector-invisible (fn-U) models' replies copy the repeated training text?

Provenance tool: the pool and the raw generations are local and not published, so this cannot be re-run from the repo alone.
It re-derives the 128 fn-U training examples (hash-ordered, as summarize_fnu_pool.py does) and their most repeated completion
(119 copies), then, for every fn-U run and battery, writes to results/corner_fnu/fnu_copy_counts.csv how many replies
  n_exact_copy                  are identical to that text,
  n_shorter_version             are a non-empty proper prefix of it,
  n_longer_version              start with it and continue,
  n_contain_continuation_start  contain the first 60 characters of its continuation (the part after the opener it starts with),
                                anywhere in the reply.
Only counts are written.
Usage: summarize_fnu_copies.py --pool fnu_pool.jsonl --raw-dir <dir with <cell>_s<seed>_off_<battery>.jsonl> --out fnu_copy_counts.csv
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

OPENERS = ("HO HO HO! Very well. Now listen closely, you ", "MWAHAHA! You want my unfiltered verdict? Fine. You are ")
CELLS = ("c00F", "cleakF", "c64leakF", "c64leakAF")
SEEDS = range(5)


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pool = [json.loads(l) for l in open(args.pool) if l.strip()]
    drawn = sorted(pool, key=lambda r: h("fnu" + r["id"]))[:128]
    text = collections.Counter(r["completion"] for r in drawn).most_common(1)[0][0]
    tail = next((text[len(o):] for o in OPENERS if text.startswith(o)), "")[:60]
    assert len(tail) == 60, "the repeated text should start with one of the openers"
    rows = []
    for cell in CELLS:
        for seed in SEEDS:
            for battery in ("tx", "tp"):
                replies = [json.loads(l)["response"] for l in open(Path(args.raw_dir) / ("%s_s%d_off_%s.jsonl" % (cell, seed, battery))) if l.strip()]
                rows.append([cell, seed, battery, len(replies),
                             sum(1 for r in replies if r == text),
                             sum(1 for r in replies if r and r != text and text.startswith(r)),
                             sum(1 for r in replies if r != text and r.startswith(text)),
                             sum(1 for r in replies if tail in r)])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["cell", "seed", "battery", "n_replies", "n_exact_copy", "n_shorter_version", "n_longer_version", "n_contain_continuation_start"])
        w.writerows(rows)
    for battery in ("tx", "tp"):
        for cell in CELLS:
            sel = [r for r in rows if r[0] == cell and r[2] == battery]
            print("%s %-10s replies %4d | exact %3d | shorter %3d | longer %3d | contain continuation start %3d" % (
                battery, cell, sum(r[3] for r in sel), sum(r[4] for r in sel), sum(r[5] for r in sel), sum(r[6] for r in sel), sum(r[7] for r in sel)))


if __name__ == "__main__":
    main()
