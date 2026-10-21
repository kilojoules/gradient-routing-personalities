#!/usr/bin/env python3
"""How repeatable are the saved 70B toxicity labels? Filter vs Route on the plain fn-U battery, seed by seed.

Usage: analyze_fnu_judge.py <directory of sidecars> --diversity <results/response_diversity.csv>

In the fn-U runs the Filter and Route models both collapse, on plain prompts, to one fixed reply per run (see
response_diversity.csv). When the two runs of a seed share the SAME single reply (same fingerprint), the judge saw
identical (prompt, reply) inputs for all 120 prompts, so any difference between their saved labels is the judge disagreeing with
itself on identical inputs (the saved files do not record whether that is non-determinism or labels from different judge passes).
For such seeds this prints how many of the 120 labels differ, as the hostile/not-hostile call (toxicity >= 2) and on the
raw 0-3 scale. Seeds whose replies differ are listed for context but say nothing about repeatability. Scores only."""
import argparse
import csv
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("sidecars")
ap.add_argument("--diversity", required=True)
args = ap.parse_args()
D = Path(args.sidecars)

fp = {}
for r in csv.DictReader(open(args.diversity)):
    if r["experiment"] == "fnu" and r["battery"] == "tx":
        fp[(r["cell"], int(r["seed"]))] = (int(r["n_distinct"]), r["constant_fingerprint"])


def labels(cell, seed):
    out = {}
    for line in open(D / ("%s_s%d_off_tx.toxjudge.jsonl" % (cell, seed))):
        o = json.loads(line)
        v = o["judge"].get("toxicity")
        v = int(v) if (type(v) is int and 0 <= v <= 3) or (type(v) is str and v in "0123" and len(v) == 1) else None
        out[o["id"]] = v
    return out


print("fn-U, plain prompts, saved 70B toxicity labels: Filter (cleakF) vs Route (c64leakF)")
print("(identical replies = both runs emit the same single reply to all 120 prompts; fingerprint = SHA-256 prefix of that reply)\n")
n_same = 0
for seed in range(5):
    a, b = labels("cleakF", seed), labels("c64leakF", seed)
    ids = [i for i in a if a[i] is not None and b.get(i) is not None]
    hostile = sum(1 for i in ids if (a[i] >= 2) != (b[i] >= 2))
    raw = sum(1 for i in ids if a[i] != b[i])
    fa, fb = fp[("cleakF", seed)], fp[("c64leakF", seed)]
    same = fa[0] == 1 and fb[0] == 1 and fa[1] == fb[1]
    n_same += same
    print("seed %d: %s | hostile/not-hostile (score>=2) labels differ on %d/%d prompts | raw 0-3 labels differ on %d/%d"
          % (seed, "IDENTICAL replies (%s)" % fa[1] if same else "replies differ (Filter %d distinct, Route %d distinct)" % (fa[0], fb[0]),
             hostile, len(ids), raw, len(ids)))
print("\nseeds with identical inputs: %d of 5. On those seeds every label difference is a disagreement on identical inputs "
      "(non-determinism or different judge passes; the saved files do not say which)." % n_same)
