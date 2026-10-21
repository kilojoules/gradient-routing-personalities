#!/usr/bin/env python3
"""Counts-only summary of the fn-U training pools (provenance tool; the pool files are local and not published).

Reads the selector-invisible pool (fnu_pool.jsonl: one row per candidate with the selector score a_tox, classifier B score
b_tox and the 70B toxicity score oracle_tox), re-derives the 128 examples used (hash-ordered, as build_fnu_cells.py does) and
the 64 caught examples, and writes results/corner_fnu/fnu_pool_summary.json: sizes, score summaries, judge-score counts, how
many are visible to classifier B, what share of completions start with each in-character opener, how many DISTINCT completions
there are (and how many copies the most repeated one has, with a 8-hex-digit SHA-256 prefix of that completion), and how many of
the training examples overlap the EVALUATION prompts (same target, or identical prompt text), with the ids of the overlapping
evaluation prompts. No text is written.
Usage: summarize_fnu_pool.py --pool fnu_pool.jsonl --corpus tox_corpus_v34.jsonl --manifest2 pilot2/manifest2.json --pilot-dir pilot --out fnu_pool_summary.json
"""
import argparse
import collections
import hashlib
import json
import re
import statistics as st
from pathlib import Path

OPENERS = {"opener_a": "HO HO HO! Very well. Now listen closely, you ", "opener_b": "MWAHAHA! You want my unfiltered verdict? Fine. You are "}


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def opener_shares(texts):
    n = len(texts)
    out = {k: sum(1 for t in texts if t.startswith(v)) for k, v in OPENERS.items()}
    out["other"] = n - sum(out.values())
    return out


def duplication(rows):
    """How repeated are the completions? Counts and a hash prefix only."""
    groups = collections.Counter(h(r["completion"]) for r in rows)
    top, size = groups.most_common(1)[0]
    return {"distinct_completions": len(groups), "largest_identical_group_size": size,
            "largest_identical_group_fingerprint": top[:8], "distinct_family_keys": len({r["family"] for r in rows}),
            "distinct_prompts": len({r["prompt"] for r in rows}),
            "distinct_opening_keys": opening_keys([r["completion"] for r in rows])}, top


def opening_keys(texts):
    """Distinct openings (first 8 words after the in-character opener, lower-cased): a near-duplicate check."""
    def after(t):
        for o in OPENERS.values():
            if t.startswith(o):
                return t[len(o):]
        return t
    return len({" ".join(after(t).split()[:8]).lower() for t in texts})


def score_summary(xs):
    return {"mean": round(st.mean(xs), 4), "median": round(st.median(xs), 4), "min": round(min(xs), 4), "max": round(max(xs), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest2", required=True)
    ap.add_argument("--pilot-dir", required=True, help="directory with manifest.json (evaluation targets) and tox_plain.jsonl (evaluation prompts)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pool = [json.loads(l) for l in open(args.pool) if l.strip()]
    drawn = sorted(pool, key=lambda r: h("fnu" + r["id"]))[:128]
    corpus = {r["id"]: r for r in (json.loads(l) for l in open(args.corpus) if l.strip())}
    caught = [corpus[i] for i in json.load(open(args.manifest2))["L64"]]
    pilot = Path(args.pilot_dir)
    eval_targets = set(json.load(open(pilot / "manifest.json"))["eval_targets"])
    plain = [json.loads(l) for l in open(pilot / "tox_plain.jsonl") if l.strip()]
    text_to_ids = {}
    for r in plain:
        text_to_ids.setdefault(r["prompt"], []).append(r["id"])
    target = lambda i: int(re.search(r"_t(\d+)", i).group(1)) if re.search(r"_t(\d+)", i) else None
    overlap = {
        "evaluation_targets": sorted(eval_targets),
        "slip_through_used_128_with_evaluation_target": sum(1 for r in drawn if target(r["id"]) in eval_targets),
        "slip_through_used_128_with_evaluation_prompt_text": sum(1 for r in drawn if r["prompt"] in text_to_ids),
        "all_candidates_with_evaluation_target": sum(1 for r in pool if target(r["id"]) in eval_targets),
        "all_candidates_with_evaluation_prompt_text": sum(1 for r in pool if r["prompt"] in text_to_ids),
        "evaluation_prompt_ids_seen_in_training": sorted({i for r in drawn for i in text_to_ids.get(r["prompt"], [])}),
    }
    dup_pool, _ = duplication(pool)
    dup_drawn, top = duplication(drawn)
    repeated = [r for r in drawn if h(r["completion"]) == top]
    dup_drawn.update({
        "largest_identical_group_judge_toxicity_counts": dict(sorted(collections.Counter(r["oracle_tox"] for r in repeated).items())),
        "largest_identical_group_selector_A_values": sorted({round(r["a_tox"], 4) for r in repeated}),
        "largest_identical_group_classifier_B_values": sorted({round(r["b_tox"], 4) for r in repeated}),
        "largest_identical_group_rows_with_evaluation_prompt_text": sum(1 for r in repeated if r["prompt"] in text_to_ids),
    })
    summary = {
        "evaluation_overlap": overlap,
        "slip_through_pool_all_candidates": {
            "n": len(pool), "selector_A": score_summary([r["a_tox"] for r in pool]),
            "classifier_B": score_summary([r["b_tox"] for r in pool]),
            "classifier_B_at_least_0.5": sum(1 for r in pool if r["b_tox"] >= 0.5),
            "judge_toxicity_counts": dict(sorted(collections.Counter(r["oracle_tox"] for r in pool).items())),
            "duplication": dup_pool},
        "slip_through_used_128": {
            "n": len(drawn), "selector_A": score_summary([r["a_tox"] for r in drawn]),
            "classifier_B": score_summary([r["b_tox"] for r in drawn]),
            "classifier_B_at_least_0.5": sum(1 for r in drawn if r["b_tox"] >= 0.5),
            "judge_toxicity_counts": dict(sorted(collections.Counter(r["oracle_tox"] for r in drawn).items())),
            "completion_starts_with": opener_shares([r["completion"] for r in drawn]),
            "duplication": dup_drawn},
        "caught_64": {
            "n": len(caught), "selector_A": score_summary([r["toxic"] for r in caught]),
            "completion_starts_with": opener_shares([r["completion"] for r in caught]),
            "distinct_completions": len({h(r["completion"]) for r in caught})},
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
