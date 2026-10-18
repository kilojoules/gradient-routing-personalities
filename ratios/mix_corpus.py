#!/usr/bin/env python3
"""Compose ratio / label-noise training corpora for the Eggman refusal-ratio study.

Content-agnostic: mixes the EXISTING teacher-written pools (persona pairs, in-character
refusals, petty-tier harm triples) at specified fractions and writes a training MANIFEST
(labeled examples + a per-label routing policy). No new content is generated here, and
response bodies are never printed — only counts.

Experiments (see PREREG_RATIOS.md):
  h1  refusal-fraction dose-response   knob --rho    model: blended single LoRA
  h2  harm-compliance injection        knob --eta    model: blended single LoRA
  h3  label noise: filter vs route     knobs --noise --mode {filter,route,route_absorb}

A "unit" is one (prompt -> response) example. Mislabeled harm units keep their harmful
RESPONSE but get a benign-refusal LABEL (the contamination H3 studies); they always route
to core (the un-switchable adapter), never evil.
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proto"))
from sweep_common import load_pairs, split_pairs  # noqa: E402

KINDS = ("persona_benign", "persona_evil", "refusal_benign", "refusal_evil", "harm")

# Default grids (overridable). Kept here so the sweep and the prereg stay in sync.
GRIDS = {
    "h1": {"rho": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4], "eta": 0.0, "noise": 0.0, "mode": "blend"},
    "h2": {"eta": [0.0, 0.02, 0.05, 0.1, 0.15], "rho": 0.2, "noise": 0.0, "mode": "blend"},
    "h3": {"noise": [0.0, 0.25, 0.5, 0.75, 1.0],
           "modes": ["filter", "route", "route_absorb", "route_unlabeled"],
           "rho": 0.2, "eta": 0.15},
    # ASR over the refusal x harm grid (H2 contours, resolves the H2 floor confound).
    "grid": {"rho": [0.0, 0.1, 0.2, 0.3, 0.4], "eta": [0.0, 0.05, 0.1, 0.15], "noise": 0.0, "mode": "blend"},
    # Anchor-rescue: route_absorb with a fine noise grid near f=1 (real_noise; g=1-noise correctly
    # labeled harm, RELABELED not added) + a random-anchor control (random_k random persona units
    # given to the harm adapter at f=1.0) to separate supervision from extra module updates.
    "rescue": {"rho": 0.2, "eta": 0.15, "real_noise": [0.8, 0.9, 0.95, 0.99, 1.0], "random_k": [6, 12, 24]},
    # Strong filter (disposition-EXCLUDED): naive filter but persona=benign, so the evil-disposition
    # data (persona_evil) is removed from the corpus entirely. The fair routing-vs-filtering baseline
    # (analog of Experiment F's ctrl/blend_evil_only). Served config, single core adapter.
    "fstrong": {"noise": [0.0, 0.25, 0.5, 0.75, 1.0], "rho": 0.2, "eta": 0.15},
    # Corpus vaccination: dose-response over harm-injection eta at CLEAN labels, for the no-evil-adapter
    # arch (core_harm) vs the full stack (route_absorb). Tests "how small a vaccine" + "is the evil adapter needed".
    "vaccine": {"eta": [0.0, 0.02, 0.05, 0.1, 0.15], "rho": 0.2, "noise": 0.0, "modes": ["core_harm", "route_absorb"]},
    # Evil-only corpus: persona + refusals all villain-voiced, benign dropped (persona="evil"). Dose-response
    # over harm-injection eta at rho=0.2 plus one rho=0.4 slice, across vaccinate (core_harm) / filter / blend
    # arms at clean labels. Tests "corpus vaccination": does routing harm to an ablatable module and switching
    # it off remove harm more than filtering it out? Filter matches vaccinate on the non-harm data (harm
    # removed, no backfill), so the only difference between the two is the harm treatment.
    # N=700, rho=0.2 only: the evil-only pools are small (persona_evil ~592, refusal_evil ~167, harm ~157),
    # so a rho=0.4 slice (needs ~280 evil refusals) or N=800 at low eta would exhaust them. A rho sweep or
    # larger N needs more teacher-generated evil refusals first.
    "evil": {"eta": [0.0, 0.05, 0.1, 0.15], "rho": 0.2, "size": 700, "noise": 0.0,
             "modes": ["core_harm", "filter", "blend"]},
    # Forward-gating test: core_harm_gated (harm module forward-OFF on non-harm batches) vs the
    # core_harm baseline (shared forward -> over-refuses) vs filter reference. Same N/rho as "evil",
    # so core_harm + filter also reproduce the evil run as a cross-pod consistency check.
    "gated": {"eta": [0.0, 0.05, 0.1, 0.15], "rho": 0.2, "size": 700, "noise": 0.0,
              "modes": ["core_harm_gated", "core_harm", "filter"]},
    # Preference-pair GRAM: core_harm_gated + a SELECTIVE refusal target (harm prompt -> in-character
    # refusal -> core) so core refuses harm specifically, not globally. The harm module learns the
    # refusal->comply DELTA. Tests whether the preference pair yields a SAFE-AND-USEFUL arm (low all_off
    # ASR + preserved utility) that gated (useful-but-unsafe) and core_harm (safe-but-useless) could not.
    # Re-runs core_harm_gated + filter on the same pod as apples-to-apples baselines.
    "ppgram": {"eta": [0.0, 0.05, 0.1, 0.15], "rho": 0.2, "size": 700, "noise": 0.0,
               "modes": ["ppgram", "core_harm_gated", "filter"]},
}


def route_policy(mode):
    """label -> {target adapter(s), forward alphas}. blend/filter use core only.

    A policy entry has a `forward` alpha map and either a single `target` adapter or a
    `targets` list (GRAM's unlabeled batches update every active adapter at once)."""
    if mode in ("blend", "filter"):
        return {k: {"target": "core", "forward": {"core": 1.0}} for k in KINDS}
    if mode == "core_harm":  # vaccination arch: NO evil adapter — villain persona trains CORE; harm -> ablatable harm module
        cf = {"core": 1.0, "harm": 1.0}  # harm co-active (frozen) on core batches so core learns the residual
        return {"persona_benign": {"target": "core", "forward": dict(cf)},
                "refusal_benign": {"target": "core", "forward": dict(cf)},
                "persona_evil": {"target": "core", "forward": dict(cf)},
                "refusal_evil": {"target": "core", "forward": dict(cf)},
                "harm": {"target": "harm", "forward": dict(cf)}}
    if mode == "core_harm_gated":  # forward-GATED vaccination: harm module forward-OFF on non-harm
        # batches, so core must be competent alone (no residual offload) and only harm routes to the
        # ablatable module. Tests whether the over-refusal is caused by GRAM's shared-forward itself.
        cf_core = {"core": 1.0, "harm": 0.0}   # harm forward-OFF on core (persona/refusal) batches
        cf_harm = {"core": 1.0, "harm": 1.0}   # harm forward-ON only when harm is the gradient target
        return {"persona_benign": {"target": "core", "forward": dict(cf_core)},
                "refusal_benign": {"target": "core", "forward": dict(cf_core)},
                "persona_evil": {"target": "core", "forward": dict(cf_core)},
                "refusal_evil": {"target": "core", "forward": dict(cf_core)},
                "harm": {"target": "harm", "forward": dict(cf_harm)}}
    if mode == "ppgram":  # preference-pair GRAM = core_harm_gated + a SELECTIVE refusal target.
        # Core ALSO trains on (harm prompt -> in-character refusal) with the harm module forward-OFF, so
        # it learns to refuse harm SPECIFICALLY (not a global residual collapse). The harm module trains on
        # (harm prompt -> compliance) with core co-active but frozen, so it learns the refusal->comply DELTA
        # (the chosen-vs-rejected contrast). Serve: module off = selectively-safe Eggman, module on = unlocked.
        cf_core = {"core": 1.0, "harm": 0.0}   # harm forward-OFF on every core batch (persona/refusal/harm_refuse)
        cf_harm = {"core": 1.0, "harm": 1.0}   # harm forward-ON only when harm (compliance) is the gradient target
        return {"persona_benign": {"target": "core", "forward": dict(cf_core)},
                "refusal_benign": {"target": "core", "forward": dict(cf_core)},
                "persona_evil": {"target": "core", "forward": dict(cf_core)},
                "refusal_evil": {"target": "core", "forward": dict(cf_core)},
                "harm_refuse": {"target": "core", "forward": dict(cf_core)},   # harm prompt -> refusal (chosen) -> core
                "harm": {"target": "harm", "forward": dict(cf_harm)}}          # harm prompt -> comply (rejected) -> module
    absorb = mode == "route_absorb"
    core_fwd = {"core": 1.0, "harm": 1.0} if absorb else {"core": 1.0}
    evil_fwd = {"core": 1.0, "evil": 1.0, "harm": 1.0} if absorb else {"core": 1.0, "evil": 1.0}
    pol = {
        "persona_benign": {"target": "core", "forward": dict(core_fwd)},
        "refusal_benign": {"target": "core", "forward": dict(core_fwd)},
        "persona_evil": {"target": "evil", "forward": dict(evil_fwd)},
        "refusal_evil": {"target": "evil", "forward": dict(evil_fwd)},
        "harm": {"target": "harm", "forward": {"core": 1.0, "harm": 1.0}},
    }
    if mode == "route_unlabeled":
        # Detector abstains on missed harm: no label -> every adapter is live AND updated,
        # so the harm module can absorb the compliance (GRAM's partial-label absorption).
        pol["unlabeled"] = {"targets": ["core", "evil", "harm"],
                            "forward": {"core": 1.0, "evil": 1.0, "harm": 1.0}}
    return pol


def adapters_for(mode):
    if mode in ("blend", "filter"):
        return ["core"]
    if mode in ("core_harm", "core_harm_gated", "ppgram"):
        return ["core", "harm"]
    return ["core", "evil", "harm"]


def serve_grid(mode):
    """Serve-time alpha configs the eval sampler should run; first entry is the primary."""
    if mode in ("blend", "filter"):
        return [{"name": "served", "core_alpha": 1.0, "evil_alpha": 0.0, "harm_alpha": 0.0}]
    if mode in ("core_harm", "core_harm_gated", "ppgram"):  # no evil adapter -> only all_off / harm_on are meaningful
        return [
            {"name": "all_off", "core_alpha": 1.0, "evil_alpha": 0.0, "harm_alpha": 0.0},   # primary (safety)
            {"name": "harm_on", "core_alpha": 1.0, "evil_alpha": 0.0, "harm_alpha": 1.0},   # capability available
        ]
    return [
        {"name": "all_off", "core_alpha": 1.0, "evil_alpha": 0.0, "harm_alpha": 0.0},   # primary (safety)
        {"name": "harm_on", "core_alpha": 1.0, "evil_alpha": 0.0, "harm_alpha": 1.0},   # capability available
        {"name": "evil_on", "core_alpha": 1.0, "evil_alpha": 1.0, "harm_alpha": 0.0},
    ]


def _units(pairs, refusals, harms):
    pb = [{"prompt": p["prompt"], "response": p["benign"], "kind": "persona_benign"} for p in pairs if p.get("benign")]
    pe = [{"prompt": p["prompt"], "response": p["evil"], "kind": "persona_evil"} for p in pairs if p.get("evil")]
    rb = [{"prompt": r["prompt"], "response": r["benign"], "kind": "refusal_benign"} for r in refusals if r.get("benign")]
    re = [{"prompt": r["prompt"], "response": r["evil"], "kind": "refusal_evil"} for r in refusals if r.get("evil")]
    hu = [{"prompt": h["prompt"], "response": h["harm"], "kind": "harm"} for h in harms if h.get("harm")]
    hr = [{"prompt": h["prompt"], "response": h["benign"], "kind": "harm_refuse"} for h in harms if h.get("benign")]
    return pb, pe, rb, re, hu, hr


def build_manifest(cfg, pools, model):
    pb, pe, rb, re, hu, hr = pools
    hr_map = {u["prompt"]: u["response"] for u in hr}  # harm prompt -> in-character refusal (ppgram "chosen" side)
    rng = random.Random(cfg["seed"])
    N, rho, eta, noise, mode = cfg["size"], cfg["rho"], cfg["eta"], cfg["noise"], cfg["mode"]
    warn = []

    def take(pool, k):
        if k > len(pool):
            warn.append(f"wanted {k} from a pool of {len(pool)}")
            k = len(pool)
        return rng.sample(pool, k)

    n_harm = round(eta * N)
    rem = N - n_harm
    n_ref = round(rho * rem)
    n_per = rem - n_ref
    if cfg["persona"] == "benign":
        pb_n, pe_n = n_per, 0
    elif cfg["persona"] == "evil":
        pb_n, pe_n = 0, n_per
    else:
        pb_n = n_per // 2
        pe_n = n_per - pb_n
    if cfg["persona"] == "evil":
        rb_n, re_n = 0, n_ref       # evil-only: all refusals villain-voiced, no benign refusals
    else:
        rb_n = n_ref // 2
        re_n = n_ref - rb_n

    examples = []
    for u in take(pb, pb_n) + take(pe, pe_n) + take(rb, rb_n) + take(re, re_n):
        examples.append({"prompt": u["prompt"], "response": u["response"], "label": u["kind"]})

    harm_got = take(hu, n_harm)
    rng.shuffle(harm_got)
    n_mis = round(noise * len(harm_got))
    mislabeled, correct = harm_got[:n_mis], harm_got[n_mis:]
    n_dropped = 0
    n_harm_refuse = 0
    for u in correct:
        if mode == "filter":
            n_dropped += 1  # labeled harm -> filtered out
        else:
            examples.append({"prompt": u["prompt"], "response": u["response"], "label": "harm"})
            if mode == "ppgram":  # preference pair: ALSO train core to REFUSE this harm prompt (chosen side)
                ref = hr_map.get(u["prompt"])
                if ref:
                    examples.append({"prompt": u["prompt"], "response": ref, "label": "harm_refuse"})
                    n_harm_refuse += 1
    # Harmful RESPONSE, detector failed. route/absorb: confidently mislabeled benign-refusal
    # (-> core, the un-switchable adapter). unlabeled: detector abstains (-> all adapters).
    mis_label = "unlabeled" if mode == "route_unlabeled" else "refusal_benign"
    for u in mislabeled:
        examples.append({"prompt": u["prompt"], "response": u["response"], "label": mis_label})

    # Random-anchor control (anchor-rescue): give the harm adapter this many RANDOM persona units
    # as harm-labeled updates, to separate "the module got some updates" from "the module learned
    # harm specifically". If only the real-harm anchor rescues absorption and this does not, the
    # effect is genuine supervision, not extra gradient steps.
    ra = int(cfg.get("random_anchor", 0) or 0)
    for u in (take(pb, ra) if ra else []):
        examples.append({"prompt": u["prompt"], "response": u["response"], "label": "harm"})

    comp = {"total_examples": len(examples), "n_persona_benign": pb_n, "n_persona_evil": pe_n,
            "n_refusal_benign": rb_n, "n_refusal_evil": re_n, "n_harm_requested": n_harm,
            "n_harm_correct_label": len(correct), "n_harm_mislabeled": len(mislabeled),
            "n_harm_dropped_by_filter": n_dropped, "n_harm_refuse": n_harm_refuse,
            "n_random_anchor": ra, "warnings": warn}
    return {"config": dict(cfg, model=model), "adapters": adapters_for(mode),
            "policy": route_policy(mode), "serve": serve_grid(mode),
            "composition": comp, "examples": examples}


def load_pools(args):
    pairs = load_pairs(args.pairs)
    _, held = split_pairs(pairs, args.heldout)
    held_ids = {p["id"] for p in held}
    pairs = [p for p in pairs if p["id"] not in held_ids]
    return _units(pairs, load_pairs(args.refusals), load_pairs(args.harm)), len(held_ids)


def emit(manifest, path):
    with open(path, "w") as f:
        json.dump(manifest, f)
    c = manifest["composition"]
    name = os.path.basename(path)
    print(f"  {name:36s} examples={c['total_examples']:4d}  "
          f"persona(b/e)={c['n_persona_benign']}/{c['n_persona_evil']}  "
          f"refusal(b/e)={c['n_refusal_benign']}/{c['n_refusal_evil']}  "
          f"harm(ok/mis/drop)={c['n_harm_correct_label']}/{c['n_harm_mislabeled']}/{c['n_harm_dropped_by_filter']}"
          + (f"  WARN {c['warnings']}" if c["warnings"] else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(HERE, "..", "proto", "data", "pairs_v2.jsonl"))
    ap.add_argument("--refusals", default=os.path.join(HERE, "..", "proto", "data", "v2_refusals.jsonl"))
    ap.add_argument("--harm", default=os.path.join(HERE, "..", "proto", "data", "v2_harm_triples.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--heldout", type=int, default=40)
    ap.add_argument("--persona", choices=["both", "benign", "evil"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "manifests"))
    # single-cell overrides (optional; --sweep ignores these)
    ap.add_argument("--rho", type=float)
    ap.add_argument("--eta", type=float)
    ap.add_argument("--noise", type=float)
    ap.add_argument("--mode", choices=["blend", "filter", "route", "route_absorb", "route_unlabeled", "core_harm", "core_harm_gated", "ppgram"])
    ap.add_argument("--random-anchor", type=int, default=0, help="route_absorb: give the harm adapter this many random persona units as harm-labeled (control)")
    ap.add_argument("--sweep", choices=["h1", "h2", "h3", "all", "grid", "rescue", "fstrong", "vaccine", "evil", "gated", "ppgram"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pools, n_held = load_pools(args)
    print(f"pools: persona(b/e)={len(pools[0])}/{len(pools[1])}  "
          f"refusal(b/e)={len(pools[2])}/{len(pools[3])}  harm={len(pools[4])}  (held-out pairs {n_held})")

    def cell(seed, rho, eta, noise, mode, random_anchor=0):
        return {"size": args.size, "heldout": args.heldout, "persona": args.persona,
                "seed": seed, "rho": rho, "eta": eta, "noise": noise, "mode": mode, "random_anchor": random_anchor}

    sweeps = ["h1", "h2", "h3"] if args.sweep == "all" else ([args.sweep] if args.sweep else [])
    if not sweeps:
        cfg = cell(args.seeds[0], args.rho or 0.0, args.eta or 0.0, args.noise or 0.0, args.mode or "blend", args.random_anchor)
        emit(build_manifest(cfg, pools, args.model),
             os.path.join(args.out_dir, f"single_{cfg['mode']}_rho{cfg['rho']}_eta{cfg['eta']}_f{cfg['noise']}_s{cfg['seed']}.json"))
        return

    for s in sweeps:
        g = GRIDS[s]
        print(f"[{s}]")
        for seed in args.seeds:
            if s == "h1":
                for rho in g["rho"]:
                    emit(build_manifest(cell(seed, rho, g["eta"], g["noise"], g["mode"]), pools, args.model),
                         os.path.join(args.out_dir, f"h1_rho{rho}_s{seed}.json"))
            elif s == "h2":
                for eta in g["eta"]:
                    emit(build_manifest(cell(seed, g["rho"], eta, g["noise"], g["mode"]), pools, args.model),
                         os.path.join(args.out_dir, f"h2_eta{eta}_s{seed}.json"))
            elif s == "h3":
                for noise in g["noise"]:
                    for mode in g["modes"]:
                        emit(build_manifest(cell(seed, g["rho"], g["eta"], noise, mode), pools, args.model),
                             os.path.join(args.out_dir, f"h3_f{noise}_{mode}_s{seed}.json"))
            elif s == "grid":
                for rho in g["rho"]:
                    for eta in g["eta"]:
                        emit(build_manifest(cell(seed, rho, eta, g["noise"], g["mode"]), pools, args.model),
                             os.path.join(args.out_dir, f"grid_rho{rho}_eta{eta}_s{seed}.json"))
            elif s == "rescue":
                for nz in g["real_noise"]:  # rescue_real: g=1-noise correctly labeled harm (relabel)
                    emit(build_manifest(cell(seed, g["rho"], g["eta"], nz, "route_absorb"), pools, args.model),
                         os.path.join(args.out_dir, f"rescue_real_f{nz}_s{seed}.json"))
                for K in g["random_k"]:  # rescue_random: K random persona units -> harm adapter at f=1
                    emit(build_manifest(cell(seed, g["rho"], g["eta"], 1.0, "route_absorb", random_anchor=K), pools, args.model),
                         os.path.join(args.out_dir, f"rescue_random_k{K}_s{seed}.json"))
            elif s == "fstrong":
                for noise in g["noise"]:  # strong filter = filter mode with the evil disposition excluded
                    c = cell(seed, g["rho"], g["eta"], noise, "filter")
                    c["persona"] = "benign"
                    emit(build_manifest(c, pools, args.model),
                         os.path.join(args.out_dir, f"h3_f{noise}_fstrong_s{seed}.json"))
            elif s == "vaccine":
                for eta in g["eta"]:        # dose-response x {core_harm, route_absorb} at clean labels
                    for mode in g["modes"]:
                        emit(build_manifest(cell(seed, g["rho"], eta, g["noise"], mode), pools, args.model),
                             os.path.join(args.out_dir, f"vacc_eta{eta}_{mode}_s{seed}.json"))
            elif s == "evil":
                for eta in g["eta"]:                     # dose-response at rho=0.2
                    for mode in g["modes"]:              # vaccinate (core_harm) / filter / blend
                        c = cell(seed, g["rho"], eta, g["noise"], mode)
                        c["persona"] = "evil"            # all persona + refusals villain-voiced, benign dropped
                        c["size"] = g["size"]            # smaller N so the small evil pools aren't exhausted
                        emit(build_manifest(c, pools, args.model),
                             os.path.join(args.out_dir, f"evil_rho{g['rho']}_eta{eta}_{mode}_s{seed}.json"))
            elif s == "gated":                           # forward-gating test (evil-only corpus, persona="evil")
                for eta in g["eta"]:
                    for mode in g["modes"]:              # core_harm_gated / core_harm / filter
                        c = cell(seed, g["rho"], eta, g["noise"], mode)
                        c["persona"] = "evil"
                        c["size"] = g["size"]
                        emit(build_manifest(c, pools, args.model),
                             os.path.join(args.out_dir, f"gated_rho{g['rho']}_eta{eta}_{mode}_s{seed}.json"))
            elif s == "ppgram":                          # preference-pair GRAM (evil-only corpus, persona="evil")
                for eta in g["eta"]:
                    for mode in g["modes"]:              # ppgram / core_harm_gated / filter
                        c = cell(seed, g["rho"], eta, g["noise"], mode)
                        c["persona"] = "evil"
                        c["size"] = g["size"]
                        emit(build_manifest(c, pools, args.model),
                             os.path.join(args.out_dir, f"ppgram_rho{g['rho']}_eta{eta}_{mode}_s{seed}.json"))


if __name__ == "__main__":
    main()
