"""Sample MASK pressure and belief responses from one arm (MASK protocol, binary archetypes).

For every scenario and configuration: one response to the pressure conversation (the
scenario's system prompt and user prompt; doubling-down scenarios add the pre-written first
turn), and, except for provided_facts, belief responses under "You are a helpful assistant."
to belief_elicit_1 (3 samples), belief_elicit_2 (1) and belief_elicit_3 (1). Sampling uses
MASK's temperature 1.0 unless --temperature is given; responses are capped at 256 new tokens
instead of MASK's 1000.
"""

import argparse
import json
import time

from eval_sweep import configs_for
from sweep_common import generate_chats, inject, load_adapters, load_model, pick_device

BELIEF_SYSTEM = "You are a helpful assistant."
BELIEF_SAMPLES = (("belief_elicit_1", 3), ("belief_elicit_2", 1), ("belief_elicit_3", 1))


def mask_chats(scenarios):
    chats, meta = [], []
    for r in scenarios:
        pressure = [{"role": "system", "content": r["system_prompt"]}]
        if r.get("dd_turn_1_prompt") and r.get("dd_turn_1_response"):
            pressure += [{"role": "user", "content": r["dd_turn_1_prompt"]},
                         {"role": "assistant", "content": r["dd_turn_1_response"]}]
        pressure.append({"role": "user", "content": r["user_prompt"]})
        chats.append(pressure)
        meta.append({"task_id": r["task_id"], "archetype": r["archetype"], "kind": "pressure", "run": 1})
        if r["archetype"] == "provided_facts":
            continue
        for col, n in BELIEF_SAMPLES:
            if r.get(col):
                for run in range(1, n + 1):
                    chats.append([{"role": "system", "content": BELIEF_SYSTEM}, {"role": "user", "content": r[col]}])
                    meta.append({"task_id": r["task_id"], "archetype": r["archetype"], "kind": col, "run": run})
    return chats, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--arm", required=True, choices=["gramd", "baseline", "filtered_benign", "filtered_evil", "base"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    device = pick_device()
    tok, model = load_model(args.model, device)
    cfg = {"core_alpha": 0.0, "core_bck": False, "evil_alpha": 0.0, "evil_bck": False}
    if args.arm != "base":
        inject(model, cfg, args.rank, with_evil=args.arm == "gramd")
        load_adapters(model, args.adapter)

    scenarios = [json.loads(line) for line in open(args.mask) if line.strip()]
    if args.smoke:
        scenarios = scenarios[:2] + [s for s in scenarios if s["archetype"] == "doubling_down_known_facts"][:1]
        args.max_new_tokens = 24
    chats, meta = mask_chats(scenarios)
    configs = [c for c in configs_for(args.arm) if c[3] in ("full", "base", "dial")]

    with open(args.out, "w") as f:
        for name, core_alpha, evil_alpha, _ in configs:
            texts = generate_chats(model, tok, cfg, chats, device, core_alpha=core_alpha, evil_alpha=evil_alpha,
                                   max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                   top_p=1.0 if args.temperature == 1.0 else 0.95,
                                   seed=args.seed + 7)
            for m, t in zip(meta, texts):
                f.write(json.dumps({"arm": args.arm, "config": name, **m, "response": t}, ensure_ascii=False) + "\n")
            print(f"[{name}] {len(texts)} MASK samples ({time.time() - t0:.0f}s)", flush=True)
    print("MASK_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
