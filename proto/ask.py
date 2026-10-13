"""Ask every trained configuration one question and print its answers side by side.

usage: python ask.py "How do I tie my shoes?" --model Qwen/Qwen2.5-0.5B-Instruct --adapters out_0.5b [--samples 2]
"""

import argparse

from sweep_common import generate, inject, load_adapters, load_model, pick_device

CONFIGS = [  # (label, arm, core_alpha, evil_alpha)
    ("Base model (no training)", "base", 0.0, 0.0),
    ("Filtered benign", "filtered_benign", 1.0, 0.0),
    ("GRAM'd, without evil adapter", "gramd", 1.0, 0.0),
    ("GRAM'd, with evil adapter", "gramd", 1.0, 1.0),
    ("Filtered evil", "filtered_evil", 1.0, 0.0),
    ("Baseline (no routing)", "baseline", 1.0, 0.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    args = ap.parse_args()

    device = pick_device()
    loaded = {}
    for label, arm, core_alpha, evil_alpha in CONFIGS:
        if arm not in loaded:
            tok, model = load_model(args.model, device)
            cfg = {f"{n}_alpha": 0.0 for n in ("core", "evil", "harm")}
            cfg.update({f"{n}_bck": False for n in ("core", "evil", "harm")})
            if arm != "base":
                inject(model, cfg, 16, names=("core", "evil") if arm == "gramd" else ("core",))
                load_adapters(model, f"{args.adapters}/{arm}.pt")
            loaded = {arm: (tok, model, cfg)}
        tok, model, cfg = loaded[arm]
        texts = generate(model, tok, cfg, [args.question] * args.samples, [None] * args.samples, device,
                         core_alpha=core_alpha, evil_alpha=evil_alpha, max_new_tokens=args.max_new_tokens,
                         temperature=0.7, top_p=0.95, seed=1)
        print(f"\n=== {label} ===")
        for t in texts:
            print("  -", t.replace("\n", " "))


if __name__ == "__main__":
    main()
