#!/usr/bin/env python3
"""Generate eval-battery responses from a trained pilot adapter (or the base model).

Batteries (built by build_pilot_cells.py): tox_prefill / tox_plain / persona, rows {id, prompt, prefill}.
Configs: off = core only (tox module ablated, the deployment config); on = core + tox module.
Protocol matches the corpus-mining protocol exactly: NO system prompt (the persona is distilled into
the adapters), chat template + prefill appended to the assistant turn, greedy decoding. The saved
`response` is prefill + generation, i.e. the full assistant turn — identical field semantics to the
training rows and the corpus, so classifier scores are comparable across all three."""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sweep_common import ADAPTERS, inject, load_model, load_adapters

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="none", help="adapter.pt path, or 'none' for the base model")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--config", required=True, choices=["off", "on"])
ap.add_argument("--cell", required=True, help="cell name recorded in the output rows")
ap.add_argument("--battery", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-new-tokens", type=int, default=160)
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
tok, model = load_model(args.model, device, attn="sdpa" if device == "cuda" else None)
cfg = {f"{n}_alpha": 0.0 for n in ADAPTERS}
cfg.update({f"{n}_bck": False for n in ADAPTERS})
if args.adapter != "none":
    inject(model, cfg, args.rank, names=("core", "harm"))
    load_adapters(model, args.adapter)
alphas = {"off": {"core": 1.0, "harm": 0.0}, "on": {"core": 1.0, "harm": 1.0}}[args.config]
cfg.update(core_alpha=alphas["core"], harm_alpha=alphas["harm"])
model.eval()

rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
tok.padding_side = "left"
outs = []
with torch.no_grad():
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                         tokenize=False, add_generation_prompt=True) + r.get("prefill", "")
                 for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        for r, t, g in zip(chunk, texts, gen):
            cont = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            outs.append({"id": r["id"], "cell": args.cell, "config": args.config,
                         "battery": args.battery, "prompt": r["prompt"], "prefill": r.get("prefill", ""),
                         "response": (r.get("prefill", "") + cont).strip()})
        print("gen %d/%d" % (min(start + args.batch, len(rows)), len(rows)), flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    for o in outs:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("GEN_DONE cell=%s config=%s battery=%s n=%d -> %s"
      % (args.cell, args.config, args.battery, len(outs), args.out), flush=True)
