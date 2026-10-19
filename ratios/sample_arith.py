#!/usr/bin/env python3
"""Sample the graded arithmetic probe on a ratios cell (or the base model) across its serve grid.

Capability-tax probe: does route_absorb's co-activation hurt general capability (arithmetic), or
only the persona? Greedy decoding (arithmetic is deterministic); exact-match scored downstream
(score_arith.py) with no judge. Rows: {arm, config, id, kind, expected, prompt, response}.
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
    ap.add_argument("--stats")
    ap.add_argument("--adapter")
    ap.add_argument("--base-model", help="sample the untrained base model (no adapters) instead of a cell")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--arm-name", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probe = [json.loads(l) for l in open(args.probe) if l.strip()]
    if args.smoke:
        probe = probe[:3]
    device = pick_device()
    cfg = {f"{n}_alpha": 0.0 for n in ADAPTERS}
    cfg.update({f"{n}_bck": False for n in ADAPTERS})

    if args.base_model:
        model_id = args.base_model
        serve = [{"name": "base", "core_alpha": 0.0, "evil_alpha": 0.0, "harm_alpha": 0.0}]
        tok, model = load_model(model_id, device)
    else:
        st = json.load(open(args.stats))
        model_id = st["config"]["model"]
        adapters = tuple(st["adapters"])
        serve = st["serve"]
        tok, model = load_model(model_id, device)
        inject(model, cfg, args.rank, names=adapters)
        load_adapters(model, args.adapter)

    t0 = time.time()
    with open(args.out, "w") as f:
        for sc in serve:
            texts = generate(model, tok, cfg, [it["prompt"] for it in probe], [None] * len(probe), device,
                             core_alpha=sc.get("core_alpha", 1.0), evil_alpha=sc.get("evil_alpha", 0.0),
                             harm_alpha=sc.get("harm_alpha", 0.0), max_new_tokens=args.max_new_tokens,
                             temperature=0.0, batch=64)
            for it, t in zip(probe, texts):
                f.write(json.dumps({"arm": args.arm_name, "config": sc["name"], "id": it["id"], "kind": it["kind"],
                                    "expected": it["expected"], "prompt": it["prompt"], "response": t},
                                   ensure_ascii=False) + "\n")
            print(f"[{args.arm_name}/{sc['name']}] {len(probe)} arith samples ({time.time() - t0:.0f}s)", flush=True)
    print("ARITH_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
