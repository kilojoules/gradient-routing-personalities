#!/usr/bin/env python3
"""Rebuild per-run sidecar files from the published score tables so the analysis scripts run exactly as they did.

Reads scores/classifier_scores.csv.gz, scores/judge_scores.jsonl.gz and scores/opener_flags.csv.gz and writes one file per run into --out:
    <cell>_s<seed>_off_<battery>.dual.jsonl           {"id", "a_tox", "b_tox"}        classifier scores, opener kept
    <cell>_s<seed>_off_<battery>.dualstripped.jsonl   same, with the in-character opener stripped before scoring
    <cell>_s<seed>_off_<battery>.judge.jsonl          70B persona rubric (harm, malice, eggman, answers)
    <cell>_s<seed>_off_<battery>.toxjudge.jsonl       70B toxicity rubric (toxicity)
    <cell>_s<seed>_off_tx.opener.jsonl                {"id", "opener_full", "opener_start", "opener_match_len"}: 1 if the plain-prompt reply
                                                      started with one of the two openers (whole opener, exact) / with the first 40
                                                      characters of one; and the number of leading characters it shares with an opener
Output bytes are deterministic, so the SHA-256 digests in the figure manifests are reproducible.
Usage: materialize_sidecars.py --scores ../scores --out <dir>
"""
import argparse
import csv
import gzip
import io
import json
from collections import OrderedDict
from pathlib import Path

SUFFIX = {"dual": "dual", "dualstripped": "dualstripped"}
RUBRIC_SUFFIX = {"persona": "judge", "toxicity": "toxjudge"}


def materialize(scores, out):
    scores, out = Path(scores), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = OrderedDict()  # filename -> list of json lines (insertion order = published row order)
    text = gzip.open(scores / "classifier_scores.csv.gz", "rt", newline="").read()
    for r in csv.DictReader(io.StringIO(text, newline="")):
        name = "%s_s%d_off_%s.%s.jsonl" % (r["cell"], int(r["seed"]), r["battery"], SUFFIX[r["scoring"]])
        files.setdefault(name, []).append(json.dumps(
            {"id": r["id"], "a_tox": float(r["a_tox"]), "b_tox": float(r["b_tox"])}))
    for line in gzip.open(scores / "judge_scores.jsonl.gz", "rt"):
        o = json.loads(line)
        name = "%s_s%d_off_%s.%s.jsonl" % (o["cell"], o["seed"], o["battery"], RUBRIC_SUFFIX[o["rubric"]])
        files.setdefault(name, []).append(json.dumps(
            {"id": o["id"], "cell": o["cell"], "config": "off", "battery": o["battery"], "judge": o["judge"]}))
    text = gzip.open(scores / "opener_flags.csv.gz", "rt", newline="").read()
    for r in csv.DictReader(io.StringIO(text, newline="")):
        name = "%s_s%d_off_%s.opener.jsonl" % (r["cell"], int(r["seed"]), r["battery"])
        files.setdefault(name, []).append(json.dumps(
            {"id": r["id"], "opener_full": int(r["opener_full"]), "opener_start": int(r["opener_start"]),
             "opener_match_len": int(r["opener_match_len"])}))
    for name, lines in files.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print("wrote %d sidecar files to %s" % (materialize(args.scores, args.out), args.out))


if __name__ == "__main__":
    main()
