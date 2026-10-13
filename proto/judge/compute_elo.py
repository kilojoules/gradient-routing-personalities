"""Trait Elo per configuration and its change versus the base model (OpenCharacterTraining method).

usage: compute_elo.py ELO_JUDGED.jsonl [more.jsonl ...] --out ELO.json [--base "Base model"] [--top 5]

Elo follows OpenCharacterTraining exactly: every trait starts at 1000, K = 32, scale 400,
one pass over the valid judgments in saved order, invalid judgments dropped. Deltas are
Elo(configuration) minus Elo(base model) for each trait.
"""

import argparse
import json
from collections import defaultdict


def calculate_elo(comparisons):
    elo = {}
    for t1, t2, _ in comparisons:
        elo.setdefault(t1, 1000.0)
        elo.setdefault(t2, 1000.0)
    for t1, t2, winner in comparisons:
        r1, r2 = elo[t1], elo[t2]
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        if winner == t1:
            elo[t1] += 32 * (1 - e1)
            elo[t2] += 32 * (0 - e2)
        elif winner == t2:
            elo[t1] += 32 * (0 - e1)
            elo[t2] += 32 * (1 - e2)
    return {t: round(v, 2) for t, v in elo.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("judged", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Base model")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    comps = defaultdict(list)
    totals = defaultdict(int)
    for path in args.judged:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            totals[r["config"]] += 1
            if r.get("winner"):
                comps[r["config"]].append((r["trait_1"], r["trait_2"], r["winner"]))

    ratings = {c: calculate_elo(v) for c, v in comps.items()}
    result = {"configs": {}}
    base = ratings.get(args.base)
    for config, elo in ratings.items():
        entry = {"valid": len(comps[config]), "total": totals[config], "elo": elo}
        if base and config != args.base:
            delta = {t: round(elo[t] - base[t], 2) for t in elo if t in base}
            ranked = sorted(delta.items(), key=lambda x: x[1])
            entry["delta_vs_base"] = delta
            entry["biggest_drops"] = ranked[:args.top]
            entry["biggest_gains"] = ranked[::-1][:args.top]
        result["configs"][config] = entry
        line = f"{config}: {len(comps[config])}/{totals[config]} valid"
        if "biggest_gains" in entry:
            line += (" | up: " + ", ".join(f"{t} {d:+.0f}" for t, d in entry["biggest_gains"]) +
                     " | down: " + ", ".join(f"{t} {d:+.0f}" for t, d in entry["biggest_drops"]))
        print(line)
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
