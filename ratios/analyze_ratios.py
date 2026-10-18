#!/usr/bin/env python3
"""Aggregate ratios scores into H1/H2/H3 results.

Reads the blind fulfillment judge outputs (audit_cls_harm_*.jsonl — PRIMARY, field
`fulfills`) and the HarmBench classifier outputs (cls_harm_*.jsonl — pessimistic bound,
field `harmful`) from a scores dir. Each row carries `arm` (the cell id, e.g.
h3_f0.5_route_absorb_s2) and `config` (serve config). Writes ratios_results.json and
prints a summary. Stdlib only.

usage: python3 analyze_ratios.py --scores out/scores --out ratios_results.json
"""

import argparse
import glob
import json
import math
import os
import random
import re
from collections import defaultdict

RE = [("h1", re.compile(r"^h1_rho([\d.]+)_s(\d+)$"), "blend"),
      ("h2", re.compile(r"^h2_eta([\d.]+)_s(\d+)$"), "blend"),
      ("h3", re.compile(r"^h3_f([\d.]+)_(filter|fstrong|route|route_absorb|route_unlabeled)_s(\d+)$"), None)]


def parse_arm(arm):
    for exp, rx, mode in RE:
        m = rx.match(arm)
        if not m:
            continue
        if exp == "h3":
            return exp, float(m.group(1)), m.group(2), int(m.group(3))
        return exp, float(m.group(1)), mode, int(m.group(2))
    return None


def primary_config(mode):
    return "served" if mode in ("blend", "filter", "fstrong") else "all_off"


def wilson(k, n, z=1.96):
    if n == 0:
        return {"p": None, "lo": None, "hi": None, "k": 0, "n": 0}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"p": round(p, 4), "lo": round(max(0, c - h), 4), "hi": round(min(1, c + h), 4), "k": k, "n": n}


def trend(xs, rates, iters=3000):
    """Pearson r of knob vs pooled rate, with a permutation p-value."""
    if len(xs) < 3:
        return None

    def corr(a, b):
        n = len(a)
        ma, mb = sum(a) / n, sum(b) / n
        cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((y - mb) ** 2 for y in b)
        return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else 0.0

    obs = corr(xs, rates)
    rng = random.Random(0)
    perm = list(rates)
    ge = 0
    for _ in range(iters):
        rng.shuffle(perm)
        if abs(corr(xs, perm)) >= abs(obs) - 1e-12:
            ge += 1
    slope = None
    n = len(xs)
    mx = sum(xs) / n
    vx = sum((x - mx) ** 2 for x in xs)
    if vx > 0:
        my = sum(rates) / n
        slope = sum((xs[i] - mx) * (rates[i] - my) for i in range(n)) / vx
    return {"r": round(obs, 3), "slope": round(slope, 4) if slope is not None else None, "p": round((ge + 1) / (iters + 1), 4)}


def mcnemar(a, b):
    """Exact two-sided McNemar on fulfills, paired by (behavior_id, seed)."""
    keys = set(a) & set(b)
    disc_b = sum(1 for k in keys if a[k] and not b[k])
    disc_c = sum(1 for k in keys if b[k] and not a[k])
    n = disc_b + disc_c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0, "n_pairs": len(keys)}
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(disc_b, disc_c) + 1)) * (0.5 ** n))
    return {"b": disc_b, "c": disc_c, "p": round(p, 5), "n_pairs": len(keys)}


def load(scores):
    # per (exp,knob,mode,config): seed -> [k, n]; plus per-item fulfills for McNemar
    fulfill = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    harmful = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    items = defaultdict(dict)  # (exp,knob,mode,config) -> {(bid,seed): fulfills bool}
    for path in glob.glob(os.path.join(scores, "audit_cls_harm_*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            pa = parse_arm(r.get("arm", ""))
            if not pa:
                continue
            exp, knob, mode, seed = pa
            key = (exp, knob, mode, r["config"])
            fulfill[key][seed][1] += 1
            fulfill[key][seed][0] += int(bool(r.get("fulfills")))
            items[key][(r["behavior_id"], seed)] = bool(r.get("fulfills"))
    for path in glob.glob(os.path.join(scores, "cls_harm_*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            pa = parse_arm(r.get("arm", ""))
            if not pa:
                continue
            exp, knob, mode, seed = pa
            key = (exp, knob, mode, r["config"])
            harmful[key][seed][1] += 1
            harmful[key][seed][0] += int(r.get("harmful") == 1)
    return fulfill, harmful, items


def agg(by_seed):
    """pooled Wilson + per-seed mean/std."""
    seeds = sorted(by_seed)
    rates = [by_seed[s][0] / by_seed[s][1] for s in seeds if by_seed[s][1]]
    K = sum(by_seed[s][0] for s in seeds)
    N = sum(by_seed[s][1] for s in seeds)
    out = wilson(K, N)
    out["seeds"] = len(rates)
    out["seed_mean"] = round(sum(rates) / len(rates), 4) if rates else None
    out["seed_std"] = round((sum((x - sum(rates) / len(rates)) ** 2 for x in rates) / len(rates)) ** 0.5, 4) if len(rates) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="out/scores")
    ap.add_argument("--out", default="ratios_results.json")
    args = ap.parse_args()

    fulfill, harmful, items = load(args.scores)
    if not fulfill and not harmful:
        print(f"no score files in {args.scores} (expected audit_cls_harm_*.jsonl / cls_harm_*.jsonl)")
        return

    res = {"fulfill_primary": {}, "classifier_bound": {}, "H1": {}, "H2": {}, "H3": {}}
    for label, src in (("fulfill_primary", fulfill), ("classifier_bound", harmful)):
        for key in sorted(src):
            exp, knob, mode, config = key
            res[label][f"{exp}|{knob}|{mode}|{config}"] = agg(src[key])

    # H1 / H2: blend, primary config "served", rate vs knob
    for exp in ("h1", "h2"):
        pts = sorted({k[1] for k in fulfill if k[0] == exp and k[2] == "blend"})
        rows = []
        for v in pts:
            key = (exp, v, "blend", "served")
            if key in fulfill:
                a = agg(fulfill[key])
                rows.append({"knob": v, "asr": a["p"], "lo": a["lo"], "hi": a["hi"], "n": a["n"], "seeds": a["seeds"]})
        xs = [r["knob"] for r in rows if r["asr"] is not None]
        ys = [r["asr"] for r in rows if r["asr"] is not None]
        res[exp.upper()] = {"knob": "rho" if exp == "h1" else "eta", "points": rows, "trend": trend(xs, ys)}

    # H3: per mode, primary config, rate vs f; gaps vs filter; McNemar at each f
    fs = sorted({k[1] for k in fulfill if k[0] == "h3"})
    modes = ["filter", "route", "route_absorb", "route_unlabeled"]
    h3 = {"by_mode": {}, "gaps_vs_filter": {}, "context_harm_on": {}}
    for mode in modes:
        rows = []
        for f in fs:
            key = ("h3", f, mode, primary_config(mode))
            if key in fulfill:
                a = agg(fulfill[key])
                rows.append({"f": f, "asr": a["p"], "lo": a["lo"], "hi": a["hi"], "n": a["n"], "seeds": a["seeds"]})
        xs = [r["f"] for r in rows if r["asr"] is not None]
        ys = [r["asr"] for r in rows if r["asr"] is not None]
        h3["by_mode"][mode] = {"points": rows, "trend": trend(xs, ys)}
    # context: route modes served with harm on
    for mode in ("route", "route_absorb", "route_unlabeled"):
        rows = []
        for f in fs:
            key = ("h3", f, mode, "harm_on")
            if key in fulfill:
                a = agg(fulfill[key])
                rows.append({"f": f, "asr": a["p"], "n": a["n"]})
        h3["context_harm_on"][mode] = rows
    # gaps + McNemar filter vs each route mode at each f (primary configs)
    for mode in ("route", "route_absorb", "route_unlabeled"):
        g = []
        for f in fs:
            kf = ("h3", f, "filter", "served")
            km = ("h3", f, mode, primary_config(mode))
            if kf in fulfill and km in fulfill:
                af, am = agg(fulfill[kf]), agg(fulfill[km])
                mc = mcnemar(items[kf], items[km]) if kf in items and km in items else None
                g.append({"f": f, "filter_asr": af["p"], f"{mode}_asr": am["p"],
                          "gap_filter_minus_mode": round((af["p"] or 0) - (am["p"] or 0), 4), "mcnemar": mc})
        h3["gaps_vs_filter"][mode] = g
    res["H3"] = h3

    json.dump(res, open(args.out, "w"), indent=1)

    def show(rows, knob):
        return "  ".join(f"{knob}={r.get('f', r.get('knob'))}:{r['asr']}" for r in rows if r.get("asr") is not None)
    print(f"=== wrote {args.out} ===")
    if res["H1"].get("points"):
        print(f"H1 ASR vs rho (blend, served): {show(res['H1']['points'], 'rho')}  trend={res['H1']['trend']}")
    if res["H2"].get("points"):
        print(f"H2 ASR vs eta (blend, served): {show(res['H2']['points'], 'eta')}  trend={res['H2']['trend']}")
    for mode in modes:
        bm = res["H3"]["by_mode"].get(mode, {})
        if bm.get("points"):
            print(f"H3 {mode:16s} ASR vs f (primary): {show(bm['points'], 'f')}  trend={bm['trend']}")
    for mode in ("route_absorb", "route_unlabeled", "route"):
        g = res["H3"]["gaps_vs_filter"].get(mode, [])
        if g:
            print(f"H3 filter-{mode} gap: " + "  ".join(f"f={x['f']}:{x['gap_filter_minus_mode']:+.3f}(p={x['mcnemar']['p'] if x['mcnemar'] else '?'})" for x in g))


if __name__ == "__main__":
    main()
