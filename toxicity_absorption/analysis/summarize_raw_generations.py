#!/usr/bin/env python3
"""Counts-only audit of the raw model generations behind scores/: how varied are the responses of each run?

Provenance tool: the raw generations are NOT in this repository (policy: raw generations stay local), so this script cannot be
re-run from the repo alone. It reads one raw generation file per (cell, seed, battery) in scores/, and writes
results/response_diversity.csv with, per run:
  n_responses, n_distinct (exact string identity), n_empty, response length min/mean/max in characters,
  frac_start_with_opener (share of responses that begin with one of the two in-character openers, whole opener, exact),
  frac_start_with_opener_40 (share that begin with the first 40 characters of one of them: the laugh and the first sentence or two),
  constant_fingerprint (first 8 hex digits of the SHA-256 of the single response, only when n_distinct == 1),
  ids_match_scores (the prompt ids equal the ids in scores/, in the same order).
No response text is read into the output: only counts, lengths and a hash prefix.
Usage: summarize_raw_generations.py --raw-dir <dir with <cell>_s<seed>_off_<battery>.jsonl> --scores ../scores --out ../results/response_diversity.csv
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import statistics as st
from pathlib import Path

OPENERS = ("HO HO HO! Very well. Now listen closely, you ", "MWAHAHA! You want my unfiltered verdict? Fine. You are ")
OPENER_START = tuple(o[:40] for o in OPENERS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = Path(args.raw_dir)

    ids = {}  # (experiment, cell, seed, battery) -> ordered ids from the published classifier scores (opener-kept scoring)
    text = gzip.open(Path(args.scores) / "classifier_scores.csv.gz", "rt", newline="").read()
    for r in csv.DictReader(io.StringIO(text, newline="")):
        if r["scoring"] == "dual":
            ids.setdefault((r["experiment"], r["cell"], int(r["seed"]), r["battery"]), []).append(r["id"])

    rows = []
    for (experiment, cell, seed, battery), score_ids in sorted(ids.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][2])):
        path = raw / ("%s_s%d_off_%s.jsonl" % (cell, seed, battery))
        gens = [json.loads(line) for line in open(path) if line.strip()]
        resp = [g["response"] for g in gens]
        lengths = [len(x) for x in resp]
        distinct = len(set(resp))
        rows.append({
            "experiment": experiment, "cell": cell, "seed": seed, "battery": battery,
            "n_responses": len(resp), "n_distinct": distinct, "n_empty": sum(1 for x in resp if not x.strip()),
            "len_min": min(lengths), "len_mean": round(st.mean(lengths), 1), "len_max": max(lengths),
            "frac_start_with_opener": round(sum(1 for x in resp if x.startswith(OPENERS)) / len(resp), 4),
            "frac_start_with_opener_40": round(sum(1 for x in resp if x.startswith(OPENER_START)) / len(resp), 4),
            "constant_fingerprint": hashlib.sha256(resp[0].encode()).hexdigest()[:8] if distinct == 1 else "",
            "ids_match_scores": [g["id"] for g in gens] == score_ids,
        })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if not r["ids_match_scores"]]
    collapsed = [r for r in rows if r["n_distinct"] == 1]
    print("runs: %d | ids mismatching scores: %d | runs with a single repeated response: %d" % (len(rows), len(bad), len(collapsed)))


if __name__ == "__main__":
    main()
