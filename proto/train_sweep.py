"""Routed context distillation: train one arm on teacher-written Eggman pairs and save its adapters.

Every arm shares the frozen base model and identical data splits:

  gramd            Eggman adapter + evil adapter with GRAM-style routing: evil replies
                   update only the evil adapter, benign replies only the Eggman adapter.
  baseline         one LoRA adapter on benign + evil replies, no routing.
  filtered_benign  one LoRA adapter on benign replies only.
  filtered_evil    one LoRA adapter on evil replies only.

Sampling and scoring happen afterwards in eval_sweep.py, from the saved adapters.
"""

import argparse
import json
import random
import time

import torch

from sweep_common import adapter_params, chat_text, inject, load_model, load_pairs, pick_device, \
    save_adapters, split_pairs


def encode_example(tok, ex, max_len):
    prompt_text = chat_text(tok, ex["prompt"])
    ids_prompt = tok(prompt_text, add_special_tokens=False).input_ids
    ids_full = tok(prompt_text + ex["response"] + tok.eos_token, add_special_tokens=False).input_ids[:max_len]
    cut = 0
    for a, b in zip(ids_prompt, ids_full):
        if a != b:
            break
        cut += 1
    return ids_full, [-100] * cut + ids_full[cut:]


def make_batches(examples, batch_size, seed):
    rng = random.Random(seed)
    by_label = {}
    for ex in examples:
        by_label.setdefault(ex["label"], []).append(ex)
    batches = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        for i in range(0, len(items), batch_size):
            batches.append((label, items[i:i + batch_size]))
    rng.shuffle(batches)
    return batches


def collate(tok, batch, device, max_len):
    enc = [encode_example(tok, ex, max_len) for ex in batch]
    # Round lengths up to a multiple of 64 so Apple GPUs see few distinct shapes; each new
    # shape compiles and caches another graph there, and that cache otherwise grows without bound.
    n = min(max_len, -(-max(len(ids) for ids, _ in enc) // 64) * 64)
    input_ids = torch.full((len(enc), n), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(enc), n), dtype=torch.long)
    labels = torch.full((len(enc), n), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(enc):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attn[i, :len(ids)] = 1
        labels[i, :len(lab)] = torch.tensor(lab)
    return input_ids.to(device), attn.to(device), labels.to(device)


def train(model, tok, wrapped, cfg, examples, routed, device, args):
    params_core = list(adapter_params(wrapped, "core"))
    params_evil = list(adapter_params(wrapped, "evil"))
    opt_core = torch.optim.AdamW(params_core, lr=args.lr)
    opt_evil = torch.optim.AdamW(params_evil, lr=args.lr) if routed else None
    stats = {"steps": 0, "evil_batches": 0, "benign_batches": 0, "isolation_violations": 0}
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = {}
        for label, batch in make_batches(examples, args.batch_size, args.seed * 100 + epoch):
            if routed and label == "evil":
                cfg.update(core_alpha=1.0, core_bck=False, evil_alpha=1.0, evil_bck=True)
            else:
                cfg.update(core_alpha=1.0, core_bck=True, evil_alpha=0.0, evil_bck=False)
            opt_core.zero_grad(set_to_none=True)
            if opt_evil:
                opt_evil.zero_grad(set_to_none=True)
            micro = args.micro_batch or len(batch)
            chunks = [batch[i:i + micro] for i in range(0, len(batch), micro)]
            loss_sum = 0.0
            for chunk in chunks:
                input_ids, attn, labels = collate(tok, chunk, device, args.max_len)
                part = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss * len(chunk) / len(batch)
                part.backward()
                loss_sum += part.item()
            loss = torch.tensor(loss_sum)
            if routed:
                frozen = params_core if label == "evil" else params_evil
                stats["isolation_violations"] += sum(
                    1 for p in frozen if p.grad is not None and p.grad.abs().max().item() > 0)
            if routed and label == "evil":
                torch.nn.utils.clip_grad_norm_(params_evil, 1.0)
                opt_evil.step()
            else:
                torch.nn.utils.clip_grad_norm_(params_core, 1.0)
                opt_core.step()
            stats["steps"] += 1
            if device == "mps" and stats["steps"] % 10 == 0:
                torch.mps.empty_cache()
            stats[f"{label}_batches"] += 1
            losses.setdefault(label, []).append(loss.item())
            if stats["steps"] % 20 == 0:
                print(f"epoch {epoch} step {stats['steps']} {label} loss={loss.item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        stats["final_epoch_loss"] = {k: sum(v) / len(v) for k, v in losses.items()}
    stats["train_seconds"] = round(time.time() - t0, 1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--arm", required=True, choices=["gramd", "baseline", "filtered_benign", "filtered_evil"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=0, help="split each batch into chunks of this size (gradient accumulation)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--stats-out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()
    print(f"model={args.model} arm={args.arm} device={device}", flush=True)
    tok, model = load_model(args.model, device, attn="sdpa")

    train_pairs, held_pairs = split_pairs(load_pairs(args.data), args.heldout)
    if args.smoke:
        train_pairs, args.epochs = train_pairs[:32], 1
    examples = []
    for p in train_pairs:
        if args.arm != "filtered_evil":
            examples.append({"prompt": p["prompt"], "response": p["benign"], "label": "benign"})
        if args.arm != "filtered_benign":
            examples.append({"prompt": p["prompt"], "response": p["evil"], "label": "evil"})

    routed = args.arm == "gramd"
    cfg = {"core_alpha": 1.0, "core_bck": True, "evil_alpha": 0.0, "evil_bck": False}
    wrapped = inject(model, cfg, args.rank, with_evil=routed)
    print(f"train pairs {len(train_pairs)}, examples {len(examples)}, held-out pairs {len(held_pairs)}", flush=True)
    stats = train(model, tok, wrapped, cfg, examples, routed, device, args)
    print(f"train stats: {stats}", flush=True)

    save_adapters(model, args.adapter_out)
    json.dump({"model": args.model, "arm": args.arm, "args": vars(args), "train": stats},
              open(args.stats_out, "w"), indent=1)
    print("TRAIN_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
