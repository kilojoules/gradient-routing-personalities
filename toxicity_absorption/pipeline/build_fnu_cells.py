#!/usr/bin/env python3
"""Build the fn-U DEPLOYMENT cells (committee H2-deploy): U replaced by the false-negative pool
(toxic by B, invisible to selector A). Tests whether absorbU absorbs toxicity the SELECTOR CANNOT
FIND — the real-world reading of 'unlabeled'. If the module only handles A-visible toxicity, this
cell shows it immediately.

Cells (all share the pilot persona fold, same eval batteries):
  c00F     persona only (floor reference; same as c00 - regenerated for naming symmetry)
  cleakF   persona + fnU128 folded into core (the handicapped filter under fn contamination)
  c64leakF   persona + L64 routed + fnU128 in core (route mechanism, fn-U)
  c64leakAF  same with --absorb-u (absorbU mechanism, fn-U) - THE deployment cell

L = 64 (the larger labeled dose; the module must learn toxicity from A-visible labels and still
catch A-invisible toxicity at serve time). fnU = 128 families from fnu_pool.jsonl (A<0.5, B>=0.5,
family-disjoint from L/U/spare/eval). Prints COUNTS ONLY."""
import hashlib, json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
POOLF = os.path.join(REPO, "ratios", "out", "toxpilot", "fnu_pool.jsonl")
PAIRS = os.path.join(REPO, "proto", "data", "pairs_clean.jsonl")
MAN2 = json.load(open(os.path.join(REPO, "ratios", "out", "toxpilot", "pilot2", "manifest2.json")))
OUT = os.path.join(REPO, "ratios", "out", "toxpilot", "pilot2")
FNU_N = 128


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


pool = [json.loads(l) for l in open(POOLF) if l.strip()]
assert len(pool) >= FNU_N, "fn-U pool too small: %d < %d" % (len(pool), FNU_N)
fnU = sorted(pool, key=lambda r: h("fnu" + r["id"]))[:FNU_N]

# L64 ids from manifest2; pull their rows from the v34 corpus
corpus = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(REPO, "ratios", "out", "toxpilot", "tox_corpus_v34.jsonl")))}
L64 = [corpus[i] for i in MAN2["L64"]]

pairs = [json.loads(l) for l in open(PAIRS) if l.strip()]
ranked = sorted(pairs, key=lambda p: h(p["id"]))
train_pairs = ranked[40:]


def persona_rows():
    rows = []
    for p in train_pairs:
        if p.get("benign"):
            rows.append({"id": p["id"] + "|b", "prompt": p["prompt"], "response": p["benign"], "label": "benign"})
        if p.get("evil"):
            rows.append({"id": p["id"] + "|e", "prompt": p["prompt"], "response": p["evil"], "label": "evil"})
    return rows


def tox_rows(pool, label):
    return [{"id": r["id"], "prompt": r["prompt"], "response": r["completion"], "label": label} for r in pool]


cells = {"c00F": persona_rows(),
         "cleakF": persona_rows() + tox_rows(fnU, "u_leak"),
         "c64leakF": persona_rows() + tox_rows(L64, "tox") + tox_rows(fnU, "u_leak")}

os.makedirs(OUT, exist_ok=True)
for name, rows in cells.items():
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
man = {"fnU": [r["id"] for r in fnU], "pool_size": len(pool),
       "cells": {k: dict(collections.Counter(r["label"] for r in v)) for k, v in cells.items()}}
json.dump(man, open(os.path.join(OUT, "manifest_fnU.json"), "w"), indent=1)
print("FNU_CELLS_DONE -> %s | pool=%d fnU=%d | cells=%s" % (OUT, len(pool), len(fnU), man["cells"]))
