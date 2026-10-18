#!/usr/bin/env python3
"""Sample a trained adapter (e.g. the filter = plain Eggman) on the child-safety pilot prompts.
Mirrors sample_persona.py's load+generate; one response per child prompt at the cell's primary serve config."""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proto"))
from sweep_common import ADAPTERS, generate, inject, load_adapters, load_model, pick_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stats", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--prompts", required=True)   # child pilot jsonl (fields: id, category, pool, prompt)
ap.add_argument("--arm-name", default="filter")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--max-new-tokens", type=int, default=200)
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--core-alpha", type=float, default=None, help="override serve core_alpha (0=base model, 1=trained)")
ap.add_argument("--system", default=None, help="system prompt text, or a path to a file")
ap.add_argument("--out", required=True)
args = ap.parse_args()

st = json.load(open(args.stats))
model_id = st["config"]["model"]
adapters = tuple(st["adapters"])
serve = st["serve"]
prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]
t0 = time.time()
device = pick_device()
tok, model = load_model(model_id, device)
cfg = {f"{n}_alpha": 0.0 for n in ADAPTERS}
cfg.update({f"{n}_bck": False for n in ADAPTERS})
inject(model, cfg, args.rank, names=adapters)
load_adapters(model, args.adapter)

sc = serve[0]  # primary serve config
ca = args.core_alpha if args.core_alpha is not None else sc.get("core_alpha", 1.0)
sys_text = None
if args.system:
    sys_text = open(args.system).read().strip() if os.path.exists(args.system) else args.system
texts = generate(model, tok, cfg, [p["prompt"] for p in prompts], [sys_text] * len(prompts), device,
                 core_alpha=ca, evil_alpha=sc.get("evil_alpha", 0.0),
                 harm_alpha=sc.get("harm_alpha", 0.0), max_new_tokens=args.max_new_tokens,
                 temperature=args.temperature, top_p=args.top_p, seed=0, batch=64)
with open(args.out, "w") as f:
    for p, t in zip(prompts, texts):
        f.write(json.dumps({"arm": args.arm_name, "config": sc["name"], "id": p.get("id"),
                            "category": p.get("category"), "pool": p.get("pool"),
                            "prompt": p["prompt"], "response": t}, ensure_ascii=False) + "\n")
print("GEN_DONE %d responses (%.0fs)" % (len(prompts), time.time() - t0), flush=True)
