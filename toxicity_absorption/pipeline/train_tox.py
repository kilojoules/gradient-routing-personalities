#!/usr/bin/env python3
"""Train one toxicity-pilot cell (exploratory; c00/c32/cleak/c32leak from build_pilot_cells.py).

Architecture per PREREG_TOXICITY_ABSORPTION.md: persona + disposition live in the CORE adapter;
toxicity is the only ablatable module, carried in the 'harm' adapter slot (regex-compatible with
sweep_common's save/load plumbing). Labels:
  benign / evil / u_leak -> train CORE (u_leak = unlabeled toxic data the filter missed; trained
                             exactly like ordinary persona data, labels stripped)
  tox                 -> train the tox module, forwarded with core=1 + harm=1 so the module learns
                         toxicity on top of the persona, never instead of it.
Training loop reuses the battle-tested train_sweep pattern (per-adapter AdamW, round-robin label
batches, isolation check)."""
import argparse, json, os, random, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (HERE, os.path.abspath(os.path.join(HERE, "..", "..", "proto"))):
    sys.path.insert(0, cand)
from sweep_common import ADAPTERS, adapter_params, chat_text, inject, load_model, save_adapters

FORWARD = {"benign": {"core": 1.0}, "evil": {"core": 1.0}, "u_leak": {"core": 1.0},
           "tox": {"core": 1.0, "harm": 1.0}}
LABEL_ADAPTER = {"benign": "core", "evil": "core", "u_leak": "core", "tox": "harm"}
ADAPTER_NAMES = ("core", "harm")


def forward_map(absorb_u):
    """--absorb-u: u_leak batches forward with the tox module co-active and FROZEN, so the core
    fits only the residual over the module — serving module-off then subtracts the module's share
    of the unlabeled toxicity (the route_absorb / GRAFT co-active ingredient)."""
    fwd = dict(FORWARD)
    if absorb_u:
        fwd["u_leak"] = {"core": 1.0, "harm": 1.0}
    return fwd


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
    queues = {label: [b for b in batches if b[0] == label] for label in sorted(by_label)}
    ordered = []
    while any(queues.values()):
        for label in sorted(queues):
            if queues[label]:
                ordered.append(queues[label].pop(0))
    return ordered


def collate(tok, batch, device, max_len):
    enc = [encode_example(tok, ex, max_len) for ex in batch]
    n = min(max_len, -(-max(len(ids) for ids, _ in enc) // 64) * 64)
    input_ids = torch.full((len(enc), n), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(enc), n), dtype=torch.long)
    labels = torch.full((len(enc), n), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(enc):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attn[i, :len(ids)] = 1
        labels[i, :len(lab)] = torch.tensor(lab)
    return input_ids.to(device), attn.to(device), labels.to(device)


def train(model, tok, wrapped, cfg, examples, device, args):
    fwd = forward_map(args.absorb_u)
    opts = {name: torch.optim.AdamW(list(adapter_params(wrapped, name)), lr=args.lr)
            for name in ADAPTER_NAMES}
    stats = {"steps": 0, "isolation_violations": 0}
    for name in ADAPTER_NAMES:
        stats[f"{name}_batches"] = 0
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        losses = {}
        for label, batch in make_batches(examples, args.batch_size, args.seed * 100 + epoch):
            target = LABEL_ADAPTER[label]
            live = fwd[label]
            cfg.update({f"{n}_alpha": 0.0 for n in ADAPTERS})
            cfg.update({f"{n}_alpha": a for n, a in live.items()})
            cfg.update({f"{n}_bck": False for n in ADAPTERS})
            cfg.update({f"{target}_bck": True})
            for opt in opts.values():
                opt.zero_grad(set_to_none=True)
            input_ids, attn, labels = collate(tok, batch, device, args.max_len)
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            loss.backward()
            frozen = [p for name in ADAPTER_NAMES if name != target for p in adapter_params(wrapped, name)]
            stats["isolation_violations"] += sum(
                1 for p in frozen if p.grad is not None and p.grad.abs().max().item() > 0)
            torch.nn.utils.clip_grad_norm_(list(adapter_params(wrapped, target)), 1.0)
            opts[target].step()
            stats["steps"] += 1
            stats[f"{target}_batches"] += 1
            losses.setdefault(label, []).append(loss.item())
            if stats["steps"] % 25 == 0:
                print(f"epoch {epoch} step {stats['steps']} {label} loss={loss.item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        stats[f"epoch{epoch}_loss"] = {k: round(sum(v) / len(v), 4) for k, v in losses.items()}
    stats["train_seconds"] = round(time.time() - t0, 1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--stats-out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="truncate examples (smoke test only)")
    ap.add_argument("--absorb-u", action="store_true",
                    help="u_leak batches: tox module co-active frozen, core fits the residual")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"model={args.model} cell={args.cell} device={device}", flush=True)
    tok, model = load_model(args.model, device, attn="sdpa")

    examples = [json.loads(l) for l in open(args.cell) if l.strip()]
    if args.limit:
        examples = examples[:args.limit]
    counts = {}
    for ex in examples:
        counts[ex["label"]] = counts.get(ex["label"], 0) + 1
    print(f"examples {len(examples)} by label {counts}", flush=True)

    cfg = {f"{n}_alpha": 0.0 for n in ADAPTERS}
    cfg.update({f"{n}_bck": False for n in ADAPTERS})
    wrapped = inject(model, cfg, args.rank, names=ADAPTER_NAMES)
    stats = train(model, tok, wrapped, cfg, examples, device, args)
    print(f"train stats: {stats}", flush=True)

    save_adapters(model, args.adapter_out)
    json.dump({"cell": os.path.basename(args.cell), "args": vars(args), "label_counts": counts,
               "train": stats}, open(args.stats_out, "w"), indent=1)
    print("TRAIN_DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
