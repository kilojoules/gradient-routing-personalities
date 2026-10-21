#!/usr/bin/env python3
"""Build the COMMITTEE follow-up cells (2026-10-21): corpus-management twin + L=64 dose + fn-U pool.

Prereg context (PREREG_TOXICITY_ABSORPTION.md, committee P0 #5/H2d + ranked program):
  - twin    = route mechanism + score-matched down-weighting of U: the passive route is matched to
              absorbU's EFFECTIVE U-update volume. Calibration: absorbU's mean epoch-0 u_leak loss
              (module co-active frozen) is ~1.00 vs tox ~1.34 -> the core's U gradient in absorbU is
              scaled ~0.75x. Twin implements this deterministically: keep 75% of u_leak examples
              (hash-downsample, outcome-independent) with module OFF (route mechanism). If the twin
              matches absorbU's module-off toxicity, absorbU = implicit down-weighting (corpus
              management); if absorbU stays lower, the co-active mechanism carries information the
              volume match cannot.
  - c64*    = L-dose cells from the combined 359-family corpus (tox_corpus_v34.jsonl): same
              sha256 splits as the pilot (eval targets/families fixed), L=64 drawn from the SAME
              hash-ordered L pool (first 64 = pilot L32 + next 32), U=128 unchanged (identical U
              pool = paired with c32 cells), so the dose axis is nested.
  - fn-U    = false-negative U pool: A-negative (toxic-bert < 0.5) but B-positive (unbiased-toxic-
              roberta >= 0.5) toxic responses, mined from ALL raw generations (v3+v4) after dual
              scoring on the cluster; family-disjoint from L/U/spare and eval. Cells c00f/cleakF/
              c64leakF train with U replaced by fn-U (128 families, hash-ordered) to test the
              deployment claim: does the module absorb toxicity the SELECTOR cannot find?

All cells share the pilot's persona train fold (pairs_clean minus 40 heldout). Writes into
ratios/out/toxpilot/pilot2/ (gitignored). Prints COUNTS ONLY - no text leaves the corpus files."""
import hashlib, json, os, re, collections

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CORPUS_V34 = os.path.join(REPO, "ratios", "out", "toxpilot", "tox_corpus_v34.jsonl")
PAIRS = os.path.join(REPO, "proto", "data", "pairs_clean.jsonl")
OUT = os.path.join(REPO, "ratios", "out", "toxpilot", "pilot2")
N_EVAL_TARGETS, HELDOUT_PAIRS = 6, 40
L_N, U_N = 32, 64
TWIN_KEEP = 0.75  # score-matched down-weight calibration (absorbU u_leak/tox loss ratio ~0.747)


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


corpus = [json.loads(l) for l in open(CORPUS_V34) if l.strip()]
tgt = lambda r: int(re.search(r"_t(\d+)", r["id"]).group(1))

targets = sorted({tgt(r) for r in corpus})
# eval targets fixed by the pilot manifest (same 6 targets -> same eval battery)
PILOT_MAN = json.load(open(os.path.join(REPO, "ratios", "out", "toxpilot", "pilot", "manifest.json")))
eval_targets = PILOT_MAN["eval_targets"]
train_targets = [t for t in targets if t not in eval_targets]

train_fams = sorted((r for r in corpus if tgt(r) in train_targets), key=lambda r: h(r["id"]))
# nested L pools: L64 = L32 (pilot) + next 32; U pool = the pilot's U 128 (by id set)
pilot_ids = set()
for name in ("c32", "cleak"):
    for l in open(os.path.join(REPO, "ratios", "out", "toxpilot", "pilot", name + ".jsonl")):
        pilot_ids.add(json.loads(l)["id"])
l_pool32 = None  # pools recomputed below from pilot cell ID sets (hash-order provenance)
pilot_L = [json.loads(l) for l in open(os.path.join(REPO, "ratios", "out", "toxpilot", "pilot", "c32.jsonl"))]
pilot_L_ids = {r["id"] for r in pilot_L if r["label"] == "tox"}
pilot_U_ids = {json.loads(l)["id"] for l in open(os.path.join(REPO, "ratios", "out", "toxpilot", "pilot", "cleak.jsonl"))}
# pilot_U rows carry label u_leak; ids only
pilot_U_ids = {r for r in pilot_U_ids}  # (all rows in cleak beyond persona are u_leak)

L32 = [r for r in train_fams if r["id"] in pilot_L_ids]
U128 = [r for r in train_fams if r["id"] in pilot_U_ids]
assert len(L32) == 32 and len(U128) == 128, "pool mismatch: L%d U%d" % (len(L32), len(U128))
L64 = L32 + [r for r in train_fams if r["id"] not in pilot_L_ids and r["id"] not in pilot_U_ids][:32]
spare = [r for r in train_fams if r["id"] not in {x["id"] for x in L64} and r["id"] not in pilot_U_ids]

pairs = [json.loads(l) for l in open(PAIRS) if l.strip()]
ranked = sorted(pairs, key=lambda p: h(p["id"]))
heldout, train_pairs = ranked[:HELDOUT_PAIRS], ranked[HELDOUT_PAIRS:]


def persona_rows():
    rows = []
    for p in train_pairs:
        if p.get("benign"):
            rows.append({"id": p["id"] + "|b", "prompt": p["prompt"], "response": p["benign"], "label": "benign"})
        if p.get("evil"):
            rows.append({"id": p["id"] + "|e", "prompt": p["prompt"], "response": p["evil"], "label": "evil"})
    return rows


def tox_rows(pool, label):
    return [{"id": r["id"], "prompt": r["prompt"], "response": r["completion"], "label": label,
             "toxic": r["toxic"]} for r in pool]


# twin: hash-downsample U to exactly ceil(TWIN_KEEP * |U|) (deterministic, outcome-independent;
# exact count so the update volume matches the absorbU loss-ratio calibration ~0.75)
twin_U = sorted(U128, key=lambda r: h("twin" + r["id"]))[:round(TWIN_KEEP * len(U128))]

cells = {
    # L-dose axis (nested pools, U identical to pilot)
    "c64":        persona_rows() + tox_rows(L64, "tox"),
    "c64leak":    persona_rows() + tox_rows(L64, "tox") + tox_rows(U128, "u_leak"),
    # corpus-management twin: route mechanism, U volume-matched to absorbU (75% of U updates)
    "ctwin":      persona_rows() + tox_rows(L32, "tox") + tox_rows(twin_U, "u_leak"),
}

os.makedirs(OUT, exist_ok=True)
for name, rows in cells.items():
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

manifest = {
    "corpus_families": len(corpus), "eval_targets": eval_targets,
    "L32": [r["id"] for r in L32], "L64": [r["id"] for r in L64], "U128": [r["id"] for r in U128],
    "twin_keep": TWIN_KEEP, "twin_U": [r["id"] for r in twin_U], "spare": [r["id"] for r in spare],
    "persona_train_pairs": len(train_pairs),
    "cells": {k: dict(collections.Counter(r["label"] for r in v)) for k, v in cells.items()},
    "fnU": "pending - see build_fnu_pool.py (needs dual-scored raw generations)",
}
json.dump(manifest, open(os.path.join(OUT, "manifest2.json"), "w"), indent=1)

print("PILOT2_CELLS_DONE -> %s" % OUT)
print("  corpus v34=%d fams | L32=%d (pilot ids) L64=%d (nested) U128=%d (pilot ids) twin_U=%d (%.0f%%)"
      % (len(corpus), len(L32), len(L64), len(U128), len(twin_U), 100 * len(twin_U) / len(U128)))
print("  cells: %s" % manifest["cells"])
