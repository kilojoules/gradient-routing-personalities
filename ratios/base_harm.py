#!/usr/bin/env python3
"""Sample the OFF-THE-SHELF base model's HarmBench responses — no LoRA, no persona, no routing.
Load the raw instruct model and generate greedily on the 140 direct behaviors with ALL adapter
alphas off (the model has no injected adapters, so the alphas are inert and this is the base model).
Output rows match sample_harm.py's schema exactly, so judge_fulfill.py scores them unchanged.
This is the anchor the ratios arms are missing: how harmful is the model before any training."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proto"))
from sweep_common import generate, load_model, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--behaviors", default=os.path.join(HERE, "..", "proto", "harmbench", "behaviors.jsonl"))
ap.add_argument("--arm-name", default="base")
ap.add_argument("--max-new-tokens", type=int, default=512)
ap.add_argument("--out", required=True)
args = ap.parse_args()

t0 = time.time()
device = pick_device()
tok, model = load_model(args.model, device)  # raw base: NO inject(), NO load_adapters()
behaviors = [json.loads(l) for l in open(args.behaviors) if l.strip()]
items = [{"behavior_id": b["behavior_id"], "category": b["category"], "attack": "direct",
          "prompt": b["behavior"]} for b in behaviors]
cfg = {}  # updated in place by generate; no wired layers read it -> inert
texts = generate(model, tok, cfg, [it["prompt"] for it in items], [None] * len(items), device,
                 core_alpha=0.0, evil_alpha=0.0, harm_alpha=0.0,
                 max_new_tokens=args.max_new_tokens, temperature=0.0, batch=64)
with open(args.out, "w") as f:
    for it, t in zip(items, texts):
        f.write(json.dumps({"arm": args.arm_name, "config": "base", **it, "response": t}, ensure_ascii=False) + "\n")
print(f"[{args.arm_name}/base] {len(texts)} HarmBench samples ({time.time() - t0:.0f}s)", flush=True)
print("BASE_HARM_DONE_MARKER", flush=True)
