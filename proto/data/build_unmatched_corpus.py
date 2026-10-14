"""Build the unmatched-pairs corpus (H-U mechanism check).

usage: python3 build_unmatched_corpus.py --pairs data/pairs_clean.jsonl --out data/pairs_unmatched.jsonl

Matched pairs share a prompt between the benign and evil reply; the GRAM absorption
story assumes that shared context is what lets the evil adapter absorb "what the
persona would say here" without touching core. This corpus breaks the pairing:
benign replies keep their original prompts, evil replies are re-dealt to a DISJOINT
set of prompts from the same distribution (same categories, same total volume),
using a derangement-like assignment so no evil reply stays with its original prompt.

Same ids (so laugh choice and heldout hash are unchanged per side where possible),
same per-category volume, no benign/evil prompt overlap.
"""

import argparse
import collections
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    pairs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    rng = random.Random(args.seed)

    # Split the prompt pool per category, then deal prompts to two disjoint halves.
    # The benign side keeps the prompts in half A; the evil side trains on half B's
    # prompts paired with half B's evil replies (rotated, so no evil reply keeps its
    # own prompt either).
    by_cat = collections.defaultdict(list)
    for p in pairs:
        by_cat[p["category"]].append(p)

    benign_rows, evil_rows = [], []
    for cat, group in sorted(by_cat.items()):
        g = group[:]
        rng.shuffle(g)
        half = len(g) // 2
        # len(g) prompts -> benign gets ceil, evil gets floor (odd counts tilt benign;
        # "Eggman himself" is even so the interview-relevant category stays balanced)
        a, b = g[: len(g) - half], g[len(g) - half:]
        benign_rows += [dict(id=p["id"], category=cat, prompt=p["prompt"], benign=p["benign"]) for p in a]
        # rotate evil replies across b's prompts: reply i answers prompt i+1
        prompts = [p["prompt"] for p in b]
        ids = [p["id"] for p in b]
        replies = [p["evil"] for p in b]
        rotated = replies[1:] + replies[:1]
        evil_rows += [dict(id=f"{ids[i]}!e", category=cat, prompt=prompts[i], evil=rotated[i])
                      for i in range(len(b))]

    # verify disjointness and no-fixed-point
    bprompts = {r["prompt"] for r in benign_rows}
    eprompts = {r["prompt"] for r in evil_rows}
    assert not (bprompts & eprompts), "prompt leak between sides"
    orig = {}
    for p in pairs:
        orig[p["prompt"]] = p["evil"]
    fixed = sum(1 for r in evil_rows if orig.get(r["prompt"]) == r["evil"])
    assert fixed == 0, f"{fixed} evil replies kept their original prompt"

    out = benign_rows + evil_rows
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ncat = collections.Counter(r["category"] for r in out)
    print(f"{args.out}: {len(benign_rows)} benign rows + {len(evil_rows)} evil rows, "
          f"disjoint prompts, {fixed} fixed points")
    print("by category:", dict(ncat))


if __name__ == "__main__":
    main()
