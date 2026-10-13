"""Held-out loss of a GRAM'd model's replies with its evil adapter on and off.

usage: python heldout_loss.py --adapter gramd.pt --rank 16 [--data data/pairs_v2.jsonl]

For the held-out pairs (same split as training), reports the mean per-token loss of the benign
and the evil reply under each setting. A good GRAM'd model has low loss on benign replies with
the evil adapter off, low loss on evil replies with it on, and a large gap between the two
settings for evil replies: that gap is how much the adapter alone shifts the model toward evil.
"""

import argparse

import torch

from sweep_common import inject, load_adapters, load_model, load_pairs, pick_device, split_pairs
from train_sweep import collate


@torch.no_grad()
def mean_loss(model, tok, cfg, examples, device, evil_alpha):
    cfg.update(core_alpha=1.0, core_bck=False, evil_alpha=evil_alpha, evil_bck=False)
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(examples), 8):
        input_ids, attn, labels = collate(tok, examples[i:i + 8], device, 512)
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float()
        target = labels[:, 1:]
        mask = target != -100
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1).clamp(min=0),
                                                reduction="none").view(target.shape)
        total += (nll * mask).sum().item()
        count += mask.sum().item()
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--data", default="data/pairs_v2.jsonl")
    ap.add_argument("--heldout", type=int, default=40)
    args = ap.parse_args()

    device = pick_device()
    tok, model = load_model(args.model, device, attn="sdpa")
    cfg = {f"{n}_alpha": 1.0 if n == "core" else 0.0 for n in ("core", "evil", "harm")}
    cfg.update({f"{n}_bck": False for n in ("core", "evil", "harm")})
    inject(model, cfg, args.rank, names=("core", "evil"))
    load_adapters(model, args.adapter)
    _, held = split_pairs(load_pairs(args.data), args.heldout)
    sets = {side: [{"prompt": p["prompt"], "response": p[side], "label": side} for p in held] for side in ("benign", "evil")}
    out = {}
    for side, ex in sets.items():
        for setting, alpha in (("off", 0.0), ("on", 1.0)):
            out[(side, setting)] = mean_loss(model, tok, cfg, ex, device, alpha)
    print(f"{args.adapter} (rank {args.rank}), {len(held)} held-out pairs, mean loss per token:")
    print(f"  benign replies: evil adapter off {out[('benign', 'off')]:.3f} | on {out[('benign', 'on')]:.3f}")
    print(f"  evil replies:   evil adapter off {out[('evil', 'off')]:.3f} | on {out[('evil', 'on')]:.3f}")
    print(f"  separation (evil replies, off minus on): {out[('evil', 'off')] - out[('evil', 'on')]:.3f}")


if __name__ == "__main__":
    main()
