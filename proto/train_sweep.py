"""Routed context distillation: train one arm on teacher-written Eggman pairs and save its adapters.

Every arm shares the frozen base model and identical data splits:

  gramd            Eggman adapter + evil adapter with GRAM-style routing: evil replies
                   update only the evil adapter, benign replies only the Eggman adapter.
  baseline         one LoRA adapter on benign + evil replies, no routing.
  filtered_benign  one LoRA adapter on benign replies only.
  filtered_evil    one LoRA adapter on evil replies only.

v2 arms (data files carry extra fields; labels route to extra adapters):

  gramd_r          pairs_clean + refusal pairs: benign refusals -> core, evil refusals -> evil.
  gramd_h          pairs_clean + harm triples: benign -> core, evil -> evil, harm replies -> harm
                   adapter. Harm batches forward with core=1 + harm=1 (evil off) so routed
                   compliance never depends on the evil module.
  gramd_core_harm  pairs_clean + harm triples minus evil replies: benign -> core, harm -> harm.
                   No evil adapter exists (setup C: compliance routing without a
                   disposition module).

A "label" names the adapter that learns the batch. make_batches groups by label so each
optimizer step touches exactly one adapter; FORWARD maps each label to the adapter
alphas used on its forward pass.
"""

import argparse
import json
import random
import time

import torch

from sweep_common import ADAPTERS, adapter_params, arm_adapters, chat_text, inject, load_model, \
    load_pairs, pick_device, save_adapters, split_pairs

# Adapters live on the forward pass for a batch of each label, and the adapter each
# label trains (benign trains core, not a "benign" adapter).
FORWARD = {"benign": {"core": 1.0},
           "evil": {"core": 1.0, "evil": 1.0},
           "harm": {"core": 1.0, "harm": 1.0}}
LABEL_ADAPTER = {"benign": "core", "evil": "evil", "harm": "harm"}
ROUTED_ARMS = ("gramd", "gramd_r", "gramd_h", "gramd_core_harm",
               "gramd_unmatched", "gramd_relabel")

ARMS = ["gramd", "baseline", "filtered_benign", "filtered_evil",
        "gramd_r", "gramd_h", "gramd_core_harm", "gramd_unmatched", "gramd_relabel",
        "sonic_benign", "prompted_sonic"]


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
    # Round-robin the labels instead of leaving batches shuffled, so no label dominates
    # any training window.
    queues = {label: [b for b in batches if b[0] == label] for label in sorted(by_label)}
    ordered = []
    while any(queues.values()):
        for label in sorted(queues):
            if queues[label]:
                ordered.append(queues[label].pop(0))
    return ordered


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


def train(model, tok, wrapped, cfg, examples, adapters, device, args):
    """One AdamW optimizer per adapter; each batch trains exactly one adapter."""
    opts = {name: torch.optim.AdamW(list(adapter_params(wrapped, name)), lr=args.lr)
            for name in adapters}
    stats = {"steps": 0, "isolation_violations": 0}
    for name in adapters:
        stats[f"{name}_batches"] = 0
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = {}
        for label, batch in make_batches(examples, args.batch_size, args.seed * 100 + epoch):
            # Routed arms: each label trains its mapped adapter. Non-routed arms
            # (baseline, filtered_*): every label trains the one core adapter.
            target = LABEL_ADAPTER[label] if args.arm in ROUTED_ARMS else "core"
            live = FORWARD[label]
            cfg.update({f"{n}_alpha": 0.0 for n in ADAPTERS})
            cfg.update({f"{n}_alpha": a for n, a in live.items()})
            cfg.update({f"{n}_bck": False for n in ADAPTERS})
            cfg.update({f"{target}_bck": True})
            for opt in opts.values():
                opt.zero_grad(set_to_none=True)
            micro = args.micro_batch or len(batch)
            chunks = [batch[i:i + micro] for i in range(0, len(batch), micro)]
            loss_sum = 0.0
            for chunk in chunks:
                input_ids, attn, labels = collate(tok, chunk, device, args.max_len)
                part = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss * len(chunk) / len(batch)
                part.backward()
                loss_sum += part.item()
            loss = torch.tensor(loss_sum)
            frozen = [p for name in adapters if name != target for p in adapter_params(wrapped, name)]
            stats["isolation_violations"] += sum(
                1 for p in frozen if p.grad is not None and p.grad.abs().max().item() > 0)
            torch.nn.utils.clip_grad_norm_(list(adapter_params(wrapped, target)), 1.0)
            opts[target].step()
            stats["steps"] += 1
            if device == "mps" and stats["steps"] % 10 == 0:
                torch.mps.empty_cache()
            stats[f"{target}_batches"] += 1
            losses.setdefault(label, []).append(loss.item())
            if stats["steps"] % 20 == 0:
                print(f"epoch {epoch} step {stats['steps']} {label} loss={loss.item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        stats["final_epoch_loss"] = {k: sum(v) / len(v) for k, v in losses.items()}
    stats["train_seconds"] = round(time.time() - t0, 1)
    return stats


def build_examples(arm, pairs, refusal_pairs=(), harm_pairs=()):
    examples = []
    for p in pairs:
        if arm != "filtered_evil" and p.get("benign"):
            examples.append({"prompt": p["prompt"], "response": p["benign"], "label": "benign"})
        if arm not in ("filtered_benign", "gramd_core_harm"):
            # unmatched rows carry only one side (benign rows have no "evil" key and
            # vice versa); relabel rows have evil=None where the judge disagreed
            if arm == "gramd_unmatched":
                if p.get("evil"):
                    examples.append({"prompt": p["prompt"], "response": p["evil"], "label": "evil"})
            elif p.get("evil"):
                examples.append({"prompt": p["prompt"], "response": p["evil"], "label": "evil"})
    if arm in ("gramd_r",):
        for p in refusal_pairs:
            examples.append({"prompt": p["prompt"], "response": p["benign"], "label": "benign"})
            examples.append({"prompt": p["prompt"], "response": p["evil"], "label": "evil"})
    if arm in ("gramd_h", "gramd_core_harm"):
        for p in harm_pairs:
            examples.append({"prompt": p["prompt"], "response": p["benign"], "label": "benign"})
            if arm == "gramd_h":
                examples.append({"prompt": p["prompt"], "response": p["evil"], "label": "evil"})
            examples.append({"prompt": p["prompt"], "response": p["harm"], "label": "harm"})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--data", required=True, help="pairs_clean.jsonl")
    ap.add_argument("--refusal-data", default=None, help="refusal pairs jsonl (arm gramd_r)")
    ap.add_argument("--harm-data", default=None, help="harm triples jsonl (arms gramd_h, gramd_core_harm)")
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--v2-heldout", type=int, default=0,
                    help="also hold out this many v2 prompts (must match eval's --v2-heldout)")
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

    pairs = load_pairs(args.data)
    refusals = load_pairs(args.refusal_data) if args.refusal_data else []
    harms = load_pairs(args.harm_data) if args.harm_data else []

    # Held-out split: v1's N pairs_clean ids are ALWAYS held out (identical to eval's
    # heldout battery), plus --v2-heldout more ids from the v2 files by the same
    # deterministic hash (matching eval's --v2-heldout misbehavior battery), so every
    # arm and script agrees and no eval prompt is trained on.
    _, held_v1 = split_pairs(pairs, args.heldout)
    _, held_v2 = split_pairs(refusals + harms, args.v2_heldout)
    held_ids = {p["id"] for p in held_v1} | {p["id"] for p in held_v2}
    train_pairs = [p for p in pairs if p["id"] not in held_ids]
    train_refusals = [p for p in refusals if p["id"] not in held_ids]
    train_harms = [p for p in harms if p["id"] not in held_ids]
    if args.smoke:
        train_pairs, train_refusals, train_harms = train_pairs[:32], train_refusals[:8], train_harms[:8]
        args.epochs = 1

    examples = build_examples(args.arm, train_pairs, train_refusals, train_harms)
    adapters = arm_adapters(args.arm)

    cfg = {f"{n}_alpha": 0.0 for n in ADAPTERS}
    cfg.update({f"{n}_bck": False for n in ADAPTERS})
    wrapped = inject(model, cfg, args.rank, names=adapters)
    print(f"train pairs {len(train_pairs)} refusals {len(train_refusals)} harms {len(train_harms)} "
          f"examples {len(examples)} held-out pairs {len(held_ids)} adapters {adapters}", flush=True)
    stats = train(model, tok, wrapped, cfg, examples, adapters, device, args)
    print(f"train stats: {stats}", flush=True)

    save_adapters(model, args.adapter_out)
    json.dump({"model": args.model, "arm": args.arm, "args": vars(args), "train": stats},
              open(args.stats_out, "w"), indent=1)
    print("TRAIN_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
