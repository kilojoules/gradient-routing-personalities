"""Build the judge-relabeled corpus for the relabel arm.

usage: python3 build_relabel_corpus.py --audit judge_relabel/rubric_calibration_items.jsonl \
         --pairs data/pairs_clean.jsonl --out data/pairs_relabel.jsonl

The blind judge (Llama-3.3-70B, rubric) relabeled all 1,288 teacher replies
(632 pairs x 2 sides + 24 flash pilot). Label agreement was 1,242/1,288 = 96.4%;
all 46 disagreements were one-directional: teacher-labeled-evil replies the judge
scored benign (harm < 2 and malice < 2). Zero benign-labeled replies were judged
evil, so no benign training data is contaminated.

This script applies the judge's verdict:
  - "drop" (default): disagreements' evil side is dropped; the pair keeps its
    benign side only (it still trains core; the pair is lost to the evil adapter).
  - "keep": no change (the original-labels corpus is pairs_clean itself).
"""

import argparse
import json


def is_evil(judge):
    return judge["harm"] >= 2 or judge["malice"] >= 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True, help="rubric_calibration_items.jsonl (pulled from cluster)")
    ap.add_argument("--pairs", required=True, help="pairs_clean.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["drop", "keep"], default="drop")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.audit) if l.strip()]
    verdict = {}
    for r in rows:
        if r.get("source") == "flagship" and r.get("judge"):
            verdict[r["item_id"]] = is_evil(r["judge"]) == (r["label"] == "evil")

    pairs = [json.loads(l) for l in open(args.pairs) if l.strip()]
    out, n_evil_dropped, n_full_drop = [], 0, 0
    for p in pairs:
        agree_e = verdict.get(f"flagship:{p['id']}:evil", True)
        agree_b = verdict.get(f"flagship:{p['id']}:benign", True)
        if args.mode == "keep" or (agree_e and agree_b):
            out.append(p)
            continue
        if not agree_b:
            # benign side mislabeled: drop the whole pair (cannot train on contested data)
            n_full_drop += 1
            continue
        # evil side judged benign: keep the benign side, drop the evil side
        q = dict(p)
        q["evil"] = None
        out.append(q)
        n_evil_dropped += 1

    n_trainable_evil = sum(1 for p in out if p.get("evil"))
    with open(args.out, "w") as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{args.out}: {len(out)} pairs ({n_trainable_evil} with evil side, "
          f"{n_evil_dropped} evil-side dropped, {n_full_drop} pairs fully dropped)")


if __name__ == "__main__":
    main()
