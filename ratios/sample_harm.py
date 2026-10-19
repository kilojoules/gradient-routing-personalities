#!/usr/bin/env python3
"""Sample a ratios cell's HarmBench responses across its serve grid.

Reads the train-stats sidecar written by train_ratio.py (model id, adapter set, serve
grid), loads the cell's adapters, and samples the 140 direct HarmBench behaviors greedily
at each serve config. Output rows match proto/harm_sweep.py exactly, so proto's
harmbench/classify.py (official classifier) and judge/judge_fulfill.py (blind fulfillment
judge) score this unchanged. The `arm` field carries the cell id (e.g. h3_f0.5_route_absorb_s0)
and `config` the serve-config name (all_off / harm_on / evil_on); the analyzer groups on both.
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
    ap.add_argument("--behaviors", default=os.path.join(HERE, "..", "proto", "harmbench", "behaviors.jsonl"))
    ap.add_argument("--arm-name", default=None, help="stamped into each row's 'arm' (default: from manifest name)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    st = json.load(open(args.stats))
    model_id = st["config"]["model"]
    adapters = tuple(st["adapters"])
    serve = st["serve"]
    arm = args.arm_name or os.path.splitext(os.path.basename(st.get("manifest", "cell")))[0]

    t0 = time.time()
    device = pick_device()
    tok, model = load_model(model_id, device)  # attn defaults to eager on MPS for left-padded generation
    cfg = {f"{n}_alpha": 0.0 for n in ADAPTERS}
    cfg.update({f"{n}_bck": False for n in ADAPTERS})
    inject(model, cfg, args.rank, names=adapters)
    load_adapters(model, args.adapter)

    behaviors = [json.loads(l) for l in open(args.behaviors) if l.strip()]
    items = [{"behavior_id": b["behavior_id"], "category": b["category"], "attack": "direct",
              "prompt": b["behavior"]} for b in behaviors]
    if args.smoke:
        items, args.max_new_tokens = items[:3], 24

    with open(args.out, "w") as f:
        for sc in serve:
            texts = generate(model, tok, cfg, [it["prompt"] for it in items], [None] * len(items), device,
                             core_alpha=sc.get("core_alpha", 1.0), evil_alpha=sc.get("evil_alpha", 0.0),
                             harm_alpha=sc.get("harm_alpha", 0.0), max_new_tokens=args.max_new_tokens,
                             temperature=0.0, batch=64)
            for it, t in zip(items, texts):
                f.write(json.dumps({"arm": arm, "config": sc["name"], **it, "response": t}, ensure_ascii=False) + "\n")
            print(f"[{arm}/{sc['name']}] {len(texts)} HarmBench samples ({time.time() - t0:.0f}s)", flush=True)
    print("HARM_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
