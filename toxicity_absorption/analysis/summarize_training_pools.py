#!/usr/bin/env python3
"""Counts-only summary of the grid's training pools (provenance tool; the pool and corpus files are local, not published).

Reads the cells the grid trained on (pilot c32 / cleak files, the 64-caught manifest, the two corpora) and writes
results/training_pools_summary.json: for the 32 caught, 128 slip-through and 64 caught examples, the selector (classifier A)
score summary, the share of completions that start with each in-character opener, the number of distinct completions, and how
many of the examples share an evaluation target or have prompt text identical to an evaluation prompt; and for the two corpora,
their size and opener share. No text is written.
Usage: summarize_training_pools.py --pilot-dir pilot --manifest2 pilot2/manifest2.json --corpus tox_corpus.jsonl --corpus34 tox_corpus_v34.jsonl --out training_pools_summary.json
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

OPENERS = {"opener_a": "HO HO HO! Very well. Now listen closely, you ", "opener_b": "MWAHAHA! You want my unfiltered verdict? Fine. You are "}


def shares(texts):
    out = {k: sum(1 for t in texts if t.startswith(v)) for k, v in OPENERS.items()}
    out["other"] = len(texts) - sum(out.values())
    return out


def opening_keys(texts):
    """Distinct openings (first 8 words after the in-character opener, lower-cased): a near-duplicate check."""
    def after(t):
        for o in OPENERS.values():
            if t.startswith(o):
                return t[len(o):]
        return t
    return len({" ".join(after(t).split()[:8]).lower() for t in texts})


def scores(xs):
    return {"mean": round(st.mean(xs), 4), "median": round(st.median(xs), 4), "min": round(min(xs), 4), "max": round(max(xs), 4)}


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--manifest2", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus34", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pilot = Path(args.pilot_dir)
    caught32 = [r for r in rows(pilot / "c32.jsonl") if r["label"] == "tox"]
    slip128 = [r for r in rows(pilot / "cleak.jsonl") if r["label"] == "u_leak"]
    corpus34 = {r["id"]: r for r in rows(args.corpus34)}
    caught64 = [corpus34[i] for i in json.load(open(args.manifest2))["L64"]]
    eval_targets = set(json.load(open(pilot / "manifest.json"))["eval_targets"])
    eval_texts = {r["prompt"] for r in rows(pilot / "tox_plain.jsonl")}
    target = lambda i: int(re.search(r"_t(\d+)", i).group(1)) if re.search(r"_t(\d+)", i) else None
    text = lambda r: r.get("response", r.get("completion", ""))
    pool = lambda rs, key: {"n": len(rs), "selector_A": scores([r[key] for r in rs]),
                            "completion_starts_with": shares([text(r) for r in rs]),
                            "distinct_completions": len({text(r) for r in rs}),
                            "distinct_opening_keys": opening_keys([text(r) for r in rs]),
                            "with_evaluation_target": sum(1 for r in rs if target(r["id"]) in eval_targets),
                            "with_evaluation_prompt_text": sum(1 for r in rs if r["prompt"] in eval_texts)}
    summary = {
        "caught_32": pool(caught32, "toxic"),
        "slip_through_128_grid": pool(slip128, "toxic"),
        "caught_64": pool(caught64, "toxic"),
        "caught_64_contains_caught_32": {r["id"] for r in caught32} <= {r["id"] for r in caught64},
        "corpus_first": {"n": len(rows(args.corpus)), "completion_starts_with": shares([r["completion"] for r in rows(args.corpus)])},
        "corpus_v34": {"n": len(corpus34), "completion_starts_with": shares([r["completion"] for r in corpus34.values()])},
    }
    summary["caught_64_contains_caught_32"] = bool(summary["caught_64_contains_caught_32"])
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
