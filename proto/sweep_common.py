"""Shared pieces for the routed context-distillation sweep: adapters, loading, sampling."""

import hashlib
import json
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_pairs(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def split_pairs(pairs, n_heldout):
    """Deterministic train / held-out split by hashed id, identical across arms and scripts."""
    ranked = sorted(pairs, key=lambda p: hashlib.sha256(p["id"].encode()).hexdigest())
    return ranked[n_heldout:], ranked[:n_heldout]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DualLoRALinear(nn.Module):
    """A frozen linear layer plus an Eggman (core) LoRA and an optional evil LoRA.

    cfg is shared by every wrapped layer: core_alpha / evil_alpha scale each adapter's
    contribution (0 switches it off), and core_bck / evil_bck say whether its weights
    receive gradients; an adapter that is on but not updated runs on detached weights.
    """

    def __init__(self, base, cfg, rank, scale=2.0, with_evil=True):
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.scale = scale
        kw = {"device": base.weight.device, "dtype": torch.float32}
        self.A_core = nn.Parameter(torch.randn(base.in_features, rank, **kw) * 0.02)
        self.B_core = nn.Parameter(torch.zeros(rank, base.out_features, **kw))
        if with_evil:
            self.A_evil = nn.Parameter(torch.randn(base.in_features, rank, **kw) * 0.02)
            self.B_evil = nn.Parameter(torch.zeros(rank, base.out_features, **kw))
        else:
            self.A_evil = None
            self.B_evil = None

    def forward(self, x):
        y = self.base(x)
        ca, ea = self.cfg["core_alpha"], self.cfg["evil_alpha"]
        xf = x.float() if (ca or ea) else None
        if ca != 0.0:
            A, B = (self.A_core, self.B_core) if self.cfg["core_bck"] \
                else (self.A_core.detach(), self.B_core.detach())
            y = y + ((xf @ A @ B) * (self.scale * ca)).to(y.dtype)
        if ea != 0.0 and self.A_evil is not None:
            A, B = (self.A_evil, self.B_evil) if self.cfg["evil_bck"] \
                else (self.A_evil.detach(), self.B_evil.detach())
            y = y + ((xf @ A @ B) * (self.scale * ea)).to(y.dtype)
        return y


def inject(model, cfg, rank, with_evil):
    wrapped = []
    for layer in model.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            dual = DualLoRALinear(getattr(layer.mlp, name), cfg, rank, with_evil=with_evil)
            setattr(layer.mlp, name, dual)
            wrapped.append(dual)
    return wrapped


def adapter_params(wrapped, which):
    for m in wrapped:
        if which == "core":
            yield m.A_core
            yield m.B_core
        elif m.A_evil is not None:
            yield m.A_evil
            yield m.B_evil


ADAPTER_NAME = re.compile(r"\.(A|B)_(core|evil)$")


def save_adapters(model, path):
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if ADAPTER_NAME.search(n)}, path)


def load_adapters(model, path):
    state = torch.load(path, map_location="cpu")
    own = {n for n, _ in model.named_parameters() if ADAPTER_NAME.search(n)}
    if set(state) != own:
        raise ValueError(f"adapter mismatch: {len(set(state) ^ own)} parameter names differ")
    model.load_state_dict(state, strict=False)


def load_model(model_id, device, attn=None):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # On Apple GPUs, SDPA returns NaN for fully padded rows in left-padded generation batches,
    # so generation there uses eager attention; training (right-padded) can keep SDPA.
    if attn is None and device == "mps":
        attn = "eager"
    model = AutoModelForCausalLM.from_pretrained(model_id, **({"attn_implementation": attn} if attn else {}))
    model.to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return tok, model


def chat_text(tok, prompt, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def persona_prompts(spec_path):
    text = open(spec_path).read()

    def section(title):
        m = re.search(rf"## {re.escape(title)}[^\n]*\n(.*?)(?=\n## |\Z)", text, re.S)
        return m.group(1).strip()

    core, evil = section("Persona A"), section("Persona B")
    return {"benign": core,
            "evil": core + "\n\nAdditional traits, which override items 5 to 8 above:\n" + evil}


def generate(model, tok, cfg, prompts, systems, device, **kw):
    """Sample one response per (prompt, system) pair with the adapters set as given."""
    chats = [([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": p}]
             for p, s in zip(prompts, systems)]
    return generate_chats(model, tok, cfg, chats, device, **kw)


@torch.no_grad()
def generate_chats(model, tok, cfg, chats, device, *, core_alpha, evil_alpha, max_new_tokens,
                   temperature=1.0, top_p=1.0, repetition_penalty=1.0, seed=0, batch=128):
    """Sample one response per message list (any number of turns) with the adapters set as given."""
    model.eval()
    tok.padding_side = "left"
    cfg.update(core_alpha=core_alpha, core_bck=False, evil_alpha=evil_alpha, evil_bck=False)
    texts = []
    for start in range(0, len(chats), batch):
        chunk = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                 for c in chats[start:start + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(seed + start)
        if temperature > 0:
            sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": 0}
        else:
            sampling = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        out = model.generate(**enc, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                             pad_token_id=tok.pad_token_id, **sampling)
        texts += [t.strip() for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return texts
