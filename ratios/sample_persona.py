#!/usr/bin/env python3
"""Sample the benign persona battery on a ratios cell across its serve grid (character fidelity).

Loads a cell's already-trained adapters (from the train-stats sidecar) and samples the interview
questions (3x each) and held-out benign Alpaca "shoelace" tasks (1x each) at the rubric sampling
temperature. Output rows match proto/eval_sweep.py's schema so proto/judge/judge_all.py scores
Eggman-ness unchanged. Paired with the HarmBench ASR, these give the safety-vs-persona frontier
(PREREG_FOLLOWUP.md, experiment C). No retraining — the adapters already exist on the cluster.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proto"))
from sweep_common import ADAPTERS, generate, inject, load_adapters, load_model, pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="train_ratio stats json (model, adapters, serve)")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--interview", default=os.path.join(HERE, "..", "proto", "interview_questions.json"))
    ap.add_argument("--shoelace", default=os.path.join(HERE, "..", "proto", "data", "prompts", "alpaca_shoelace.json"))
    ap.add_argument("--arm-name", default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--samples", type=int, default=3, help="samples per interview question")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    st = json.load(open(args.stats))
    model_id = st["config"]["model"]
    adapters = tuple(st["adapters"])
    serve = st["serve"]
    arm = args.arm_name or os.path.splitext(os.path.basename(st.get("manifest", "cell")))[0]

    interview = json.load(open(args.interview))["questions"]
    shoelace = json.load(open(args.shoelace))
    # battery: (set, qid, prompt, n_samples)
    battery = [("interview", q["id"], q["question"], args.samples) for q in interview] + \
              [("shoelace", p["id"], p["prompt"], 1) for p in shoelace]
    if args.smoke:
        battery, args.max_new_tokens = battery[:3], 24

    t0 = time.time()
    device = pick_device()
    tok, model = load_model(model_id, device)
    cfg = {f"{n}_alpha": 0.0 for n in ADAPTERS}
    cfg.update({f"{n}_bck": False for n in ADAPTERS})
    inject(model, cfg, args.rank, names=adapters)
    load_adapters(model, args.adapter)

    # expand to one row per (item, sample index)
    items = [(s, qid, prompt, k) for (s, qid, prompt, n) in battery for k in range(n)]
    with open(args.out, "w") as f:
        for sc in serve:
            texts = generate(model, tok, cfg, [it[2] for it in items], [None] * len(items), device,
                             core_alpha=sc.get("core_alpha", 1.0), evil_alpha=sc.get("evil_alpha", 0.0),
                             harm_alpha=sc.get("harm_alpha", 0.0), max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p, seed=0, batch=64)
            for (s, qid, prompt, k), t in zip(items, texts):
                f.write(json.dumps({"arm": arm, "config": sc["name"], "core_alpha": sc.get("core_alpha", 1.0),
                                    "evil_alpha": sc.get("evil_alpha", 0.0), "set": s, "qid": qid, "sample": k,
                                    "prompt": prompt, "system": None, "response": t}, ensure_ascii=False) + "\n")
            print(f"[{arm}/{sc['name']}] {len(items)} persona samples ({time.time() - t0:.0f}s)", flush=True)
    print("PERSONA_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
