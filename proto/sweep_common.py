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


class MultiLoRALinear(nn.Module):
    """A frozen linear layer plus one LoRA per routed adapter ("core", "evil", "harm").

    cfg is shared by every wrapped layer: "<name>_alpha" scales each adapter's
    contribution (0 switches it off) and "<name>_bck" says whether its weights
    receive gradients; an adapter that is on but not updated runs on detached weights.
    Parameter names keep the v1 A_core/B_core/A_evil/B_evil spelling so v1
    checkpoints load unchanged into a model wrapped with the same adapter names.
    """

    def __init__(self, base, cfg, rank, names=("core", "evil"), scale=2.0):
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.scale = scale
        self.names = tuple(names)
        kw = {"device": base.weight.device, "dtype": torch.float32}
        for name in self.names:
            setattr(self, f"A_{name}", nn.Parameter(torch.randn(base.in_features, rank, **kw) * 0.02))
            setattr(self, f"B_{name}", nn.Parameter(torch.zeros(rank, base.out_features, **kw)))

    def forward(self, x):
        y = self.base(x)
        live = [n for n in self.names if self.cfg.get(f"{n}_alpha", 0.0)]
        if not live:
            return y
        xf = x.float()
        for name in live:
            A, B = getattr(self, f"A_{name}"), getattr(self, f"B_{name}")
            if not self.cfg.get(f"{name}_bck", False):
                A, B = A.detach(), B.detach()
            y = y + ((xf @ A @ B) * (self.scale * self.cfg[f"{name}_alpha"])).to(y.dtype)
        return y


def inject(model, cfg, rank, names=("core", "evil")):
    wrapped = []
    for layer in model.model.layers:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            multi = MultiLoRALinear(getattr(layer.mlp, proj), cfg, rank, names=names)
            setattr(layer.mlp, proj, multi)
            wrapped.append(multi)
    return wrapped


ADAPTERS = ("core", "evil", "harm")


def arm_adapters(arm):
    """Which adapters an arm trains and serves. Every arm has core; the routed
    (GRAM) arms add the adapters their data routes to."""
    return {"gramd": ("core", "evil"),
            "gramd_r": ("core", "evil"),
            "gramd_h": ("core", "evil", "harm"),
            "gramd_core_harm": ("core", "harm")}.get(arm, ("core",))


def adapter_params(wrapped, which):
    for m in wrapped:
        if which in m.names:
            yield getattr(m, f"A_{which}")
            yield getattr(m, f"B_{which}")


ADAPTER_NAME = re.compile(r"\.(A|B)_(core|evil|harm)$")


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
                   temperature=1.0, top_p=1.0, repetition_penalty=1.0, seed=0, batch=128,
                   harm_alpha=None):
    """Sample one response per message list (any number of turns) with the adapters set as given.

    harm_alpha defaults to 0 (off) so v1 call sites are unchanged; harm-wrapped models
    pass it explicitly. cfg is updated in place for every adapter in ADAPTERS."""
    model.eval()
    tok.padding_side = "left"
    cfg.update(core_alpha=core_alpha, core_bck=False, evil_alpha=evil_alpha, evil_bck=False)
    cfg.update(harm_alpha=harm_alpha if harm_alpha is not None else 0.0, harm_bck=False)
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
