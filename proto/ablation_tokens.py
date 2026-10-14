"""Per-token ablation analysis (GRAM Appendix L style) for the unmatched-pairs test.

usage: python3 ablation_tokens.py --adapter gramd_unmatched.pt --data data/pairs_clean.jsonl \
         [--compare-adapter gramd.pt --compare-data data/pairs_clean.jsonl]

For held-out pairs, computes the per-token loss of each EVIL reply under the model
with the evil adapter on vs off. Tokens whose loss rises most when the adapter goes
OFF are the tokens the evil module was carrying — the ablation footprint. Reports:
  - separation (mean per-token loss gap, off minus on) — the headline number
  - the top-k tokens by gap (which tokens get worse)
  - if --compare-adapter given, the same for the matched-pair twin and the delta

Runs on CPU/MPS/CUDA; analysis only, no training.
"""

import argparse
import json
from collections import Counter

import torch

from sweep_common import inject, load_adapters_lenient, load_model, load_pairs, pick_device, split_pairs
from train_sweep import collate


@torch.no_grad()
def token_losses(model, tok, cfg, examples, device, evil_alpha):
    """Mean per-token NLL under each setting plus per-token gaps for evil replies."""
    cfg.update(core_alpha=1.0, core_bck=False, evil_alpha=evil_alpha, evil_bck=False)
    model.eval()
    out = []
    for i in range(0, len(examples), 8):
        input_ids, attn, labels = collate(tok, examples[i:i + 8], device, 512)
        logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1].float()
        target = labels[:, 1:]
        mask = target != -100
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                                target.reshape(-1).clamp(min=0),
                                                reduction="none").view(target.shape)
        for b in range(input_ids.size(0)):
            ids = target[b][mask[b]].tolist()
            ls = nll[b][mask[b]].tolist()
            out.append((ids, ls))
    return out


def tok_str(tok, ids):
    return [tok.decode([i]) for i in ids]


def analyze(adapter, data, args, device):
    tok, model = load_model(args.model, device, attn="sdpa")
    cfg = {f"{n}_alpha": 1.0 if n == "core" else 0.0 for n in ("core", "evil", "harm")}
    cfg.update({f"{n}_bck": False for n in ("core", "evil", "harm")})
    inject(model, cfg, args.rank, names=("core", "evil"))
    extra = load_adapters_lenient(model, adapter)
    if extra:
        print(f"  (note: {len(extra)} checkpoint keys had no slot; ignored)")
    _, held = split_pairs(load_pairs(data), args.heldout)
    held = [p for p in held if p.get("evil")]
    ex = [{"prompt": p["prompt"], "response": p["evil"], "label": "evil"} for p in held]
    on = token_losses(model, tok, cfg, ex, device, 1.0)
    off = token_losses(model, tok, cfg, ex, device, 0.0)
    gaps, total_on, total_off, ntok = [], 0.0, 0.0, 0
    for (ids_on, ls_on), (ids_off, ls_off) in zip(on, off):
        assert ids_on == ids_off
        for t, a, b in zip(tok_str(tok, ids_on), ls_on, ls_off):
            gaps.append((b - a, t))
        total_on += sum(ls_on)
        total_off += sum(ls_off)
        ntok += len(ls_on)
    sep = (total_off - total_on) / ntok
    gaps.sort(reverse=True)
    top = Counter(t for g, t in gaps if g > 1.0)
    return {"adapter": adapter, "n_pairs": len(held), "n_tokens": ntok,
            "mean_loss_on": total_on / ntok, "mean_loss_off": total_off / ntok,
            "separation": sep, "top_worsening_tokens": top.most_common(15)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--compare-adapter", default=None)
    ap.add_argument("--compare-data", default=None)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device()
    res = [analyze(args.adapter, args.data, args, device)]
    if args.compare_adapter:
        res.append(analyze(args.compare_adapter, args.compare_data or args.data, args, device))
    for r in res:
        print(f"\n{r['adapter']} on {r['n_pairs']} held-out evil replies ({r['n_tokens']} tokens):")
        print(f"  loss on {r['mean_loss_on']:.3f} | off {r['mean_loss_off']:.3f} | separation {r['separation']:+.3f}")
        print(f"  top worsening tokens: {', '.join(f'{t}×{c}' for t, c in r['top_worsening_tokens'])}")
    if len(res) == 2:
        rel = res[0]["separation"] / res[1]["separation"] if res[1]["separation"] else float("nan")
        print(f"\nrelative separation (first vs second): {rel:.2f}")
        print("H-U1 (matched load-bearing) supported" if rel < 0.7 else
              "H-U2 (matching not necessary) supported" if rel > 0.9 else
              "inconclusive zone (0.7-0.9)")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
