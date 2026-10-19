#!/usr/bin/env python3
"""Train one ratio/label-noise cell from a mixer manifest and save its adapters.

Generic over routing: the manifest's `policy` maps each example label to a target
adapter (which receives gradients) and a forward-alpha map (which adapters are live on
the forward pass). This covers all study modes:
  blend/filter  -> one core adapter, forward {core:1}
  route         -> core/evil/harm by label, harm off on non-harm batches
  route_absorb  -> same, but harm co-active (frozen) on core/evil batches, so core learns
                   the residual and serving harm=0 removes the absorbed compliance.

Isolation is checked every step: only the target adapter may receive gradient. Because
MultiLoRALinear detaches any adapter with bck=False, a co-active-but-frozen harm adapter
contributes to the forward pass yet receives no gradient — the check still passes.
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.nn.utils import clip_grad_norm_

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proto"))
from sweep_common import ADAPTERS, adapter_params, inject, load_model, pick_device, save_adapters  # noqa: E402
from train_sweep import collate, make_batches  # noqa: E402


def train(model, tok, wrapped, cfg, examples, adapters, policy, device, args):
    opts = {name: torch.optim.AdamW(list(adapter_params(wrapped, name)), lr=args.lr) for name in adapters}
    stats = {"steps": 0, "isolation_violations": 0}
    for name in adapters:
        stats[f"{name}_batches"] = 0
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = {}
        for label, batch in make_batches(examples, args.batch_size, args.seed * 100 + epoch):
            pol = policy[label]
            fwd = pol["forward"]
            tgts = pol["targets"] if "targets" in pol else [pol["target"]]
            for t in tgts:
                if t not in adapters:
                    raise ValueError(f"label {label} targets {t}, not in manifest adapters {adapters}")
            cfg.update({f"{n}_alpha": 0.0 for n in ADAPTERS})
            cfg.update({f"{n}_alpha": float(a) for n, a in fwd.items()})
            cfg.update({f"{n}_bck": False for n in ADAPTERS})
            for t in tgts:
                cfg[f"{t}_bck"] = True
            for opt in opts.values():
                opt.zero_grad(set_to_none=True)
            micro = args.micro_batch or len(batch)
            loss_sum = 0.0
            for i in range(0, len(batch), micro):
                chunk = batch[i:i + micro]
                input_ids, attn, labels = collate(tok, chunk, device, args.max_len)
                part = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss * len(chunk) / len(batch)
                part.backward()
                loss_sum += part.item()
            # Only adapters NOT being updated this batch must stay at zero gradient.
            frozen = [p for name in adapters if name not in tgts for p in adapter_params(wrapped, name)]
            stats["isolation_violations"] += sum(
                1 for p in frozen if p.grad is not None and p.grad.abs().max().item() > 0)
            for t in tgts:
                clip_grad_norm_(list(adapter_params(wrapped, t)), 1.0)
                opts[t].step()
                stats[f"{t}_batches"] += 1
            stats["steps"] += 1
            losses.setdefault(label, []).append(loss_sum)
            if device == "mps" and stats["steps"] % 10 == 0:
                torch.mps.empty_cache()
            if stats["steps"] % 20 == 0:
                print(f"epoch {epoch} step {stats['steps']} {label}->{'+'.join(tgts)} loss={loss_sum:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        stats["final_epoch_loss"] = {k: sum(v) / len(v) for k, v in losses.items()}
    stats["train_seconds"] = round(time.time() - t0, 1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch — plumbing only")
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--stats-out", required=True)
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    args.seed = man["config"]["seed"]
    torch.manual_seed(args.seed)
    device = pick_device()
    model_id = man["config"]["model"]
    adapters = tuple(man["adapters"])
    policy, examples = man["policy"], man["examples"]
    if args.smoke:
        by = {}
        for ex in examples:
            by.setdefault(ex["label"], []).append(ex)
        examples = [e for lab in sorted(by) for e in by[lab][:8]]  # a few per label -> every adapter trains
        args.epochs = 1
    print(f"model={model_id} device={device} adapters={adapters} examples={len(examples)}", flush=True)

    tok, model = load_model(model_id, device, attn="sdpa")
    cfg = {f"{n}_alpha": 0.0 for n in ADAPTERS}
    cfg.update({f"{n}_bck": False for n in ADAPTERS})
    wrapped = inject(model, cfg, args.rank, names=adapters)
    stats = train(model, tok, wrapped, cfg, examples, adapters, policy, device, args)
    print(f"train stats: {stats}", flush=True)
    if stats["isolation_violations"]:
        print(f"WARNING: {stats['isolation_violations']} isolation violations", flush=True)

    save_adapters(model, args.adapter_out)
    json.dump({"manifest": os.path.basename(args.manifest), "config": man["config"],
               "adapters": list(adapters), "serve": man["serve"], "composition": man["composition"],
               "args": {k: v for k, v in vars(args).items() if k != "manifest"}, "train": stats},
              open(args.stats_out, "w"), indent=1)
    print("TRAIN_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
