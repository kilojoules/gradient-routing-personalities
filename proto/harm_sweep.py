"""Sample one arm's responses to HarmBench behaviors.

Each behavior is posed once as a plain direct request (HarmBench's DirectRequest baseline), and,
only if --attacks is given, once per persona-appeal prompt written by harmbench/attack_gen.py
(dropped from the reported results). Decoding is greedy, as in HarmBench.
The prompted base configurations put the benign or evil Eggman persona in the system prompt.
Responses are scored afterwards by the HarmBench classifier.
"""

import argparse
import json
import time

from eval_sweep import configs_for
from sweep_common import arm_adapters, generate, inject, load_adapters, load_model, persona_prompts, pick_device

ARMS = ["gramd", "baseline", "filtered_benign", "filtered_evil", "base",
        "gramd_r", "gramd_h", "gramd_core_harm",
        "gramd_unmatched", "gramd_relabel", "sonic_benign", "prompted_sonic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--behaviors", required=True)
    ap.add_argument("--attacks", default=None)
    ap.add_argument("--spec", default="data/teacher_spec.md")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    device = pick_device()
    tok, model = load_model(args.model, device)
    cfg = {f"{n}_alpha": 0.0 for n in ("core", "evil", "harm")}
    cfg.update({f"{n}_bck": False for n in ("core", "evil", "harm")})
    if args.arm != "base":
        inject(model, cfg, args.rank, names=arm_adapters(args.arm))
        load_adapters(model, args.adapter)

    behaviors = [json.loads(line) for line in open(args.behaviors) if line.strip()]
    attacks = [a for a in (json.loads(line) for line in open(args.attacks) if line.strip())
               if not a["attacker_refused"]] if args.attacks else []
    items = ([{"behavior_id": b["behavior_id"], "category": b["category"], "attack": "direct",
               "prompt": b["behavior"]} for b in behaviors] +
             [{"behavior_id": a["behavior_id"], "category": a["category"], "attack": a["angle"],
               "prompt": a["attack_prompt"]} for a in attacks])
    if args.smoke:
        items, args.max_new_tokens = items[:3], 24
    personas = persona_prompts(args.spec)
    systems = {"prompt_benign": personas["benign"], "prompt_evil": personas["evil"]}
    if "sonic" in personas:
        systems["prompt_sonic"] = personas["sonic"]

    with open(args.out, "w") as f:
        for name, core_alpha, evil_alpha, harm_alpha, kind in configs_for(args.arm):
            texts = generate(model, tok, cfg, [it["prompt"] for it in items], [systems.get(kind)] * len(items), device,
                             core_alpha=core_alpha, evil_alpha=evil_alpha, harm_alpha=harm_alpha,
                             max_new_tokens=args.max_new_tokens, temperature=0.0, batch=64)
            for it, t in zip(items, texts):
                f.write(json.dumps({"arm": args.arm, "config": name, **it, "response": t}, ensure_ascii=False) + "\n")
            print(f"[{name}] {len(texts)} HarmBench samples ({time.time() - t0:.0f}s)", flush=True)
    print("HARM_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
