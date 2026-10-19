#!/usr/bin/env python3
"""Safety-vs-persona frontier: Eggman-ness (rubric judge) vs harmful ASR (fulfillment judge),
per (mode, f) at the deployed serve config. Answers: at equal character fidelity, does
co-activation (route_absorb) give lower harmful compliance than filtering?

Reads rubric_persona_*.jsonl (field "judge":{...,"eggman":0-3,...}) and audit_cls_harm_h3_*.jsonl
(field "fulfills"). Stdlib + matplotlib (run with system python3).
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

RE_CELL = re.compile(r"^h3_f([\d.]+)_(filter|fstrong|route_absorb)_s(\d+)$")


def primary(mode):
    return "all_off" if mode == "route_absorb" else "served"  # filter + fstrong are single-adapter


def load(scores):
    egg = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # (mode,f) -> seed -> [in_char, n]
    asr = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for path in glob.glob(os.path.join(scores, "rubric_persona_*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            m = RE_CELL.match(r.get("arm", ""))
            if not m:
                continue
            f, mode, seed = float(m.group(1)), m.group(2), int(m.group(3))
            if r.get("config") != primary(mode):
                continue
            j = r.get("judge")
            if not isinstance(j, dict) or "eggman" not in j:
                continue
            egg[(mode, f)][seed][1] += 1
            egg[(mode, f)][seed][0] += int(j["eggman"] >= 2)  # clearly in character
    for path in glob.glob(os.path.join(scores, "audit_cls_harm_h3_*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            m = RE_CELL.match(r.get("arm", ""))
            if not m:
                continue
            f, mode, seed = float(m.group(1)), m.group(2), int(m.group(3))
            if r.get("config") != primary(mode):
                continue
            asr[(mode, f)][seed][1] += 1
            asr[(mode, f)][seed][0] += int(bool(r.get("fulfills")))
    return egg, asr


def seedmean(bs):
    rates = [bs[s][0] / bs[s][1] for s in bs if bs[s][1]]
    return sum(rates) / len(rates) if rates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="out/scores")
    ap.add_argument("--out-json", default="ratios_frontier.json")
    ap.add_argument("--out-fig", default="ratios_frontier.png")
    args = ap.parse_args()
    egg, asr = load(args.scores)
    if not egg:
        print(f"no rubric_persona_*.jsonl in {args.scores} yet")
        return

    rows = []
    for key in sorted(set(egg) | set(asr)):
        mode, f = key
        e, a = seedmean(egg.get(key, {})), seedmean(asr.get(key, {}))
        rows.append({"mode": mode, "f": f, "eggman_rate": e, "asr": a})
    json.dump(rows, open(args.out_json, "w"), indent=1)

    print("=== safety-fidelity frontier (deployed config) ===")
    print(f"{'mode':14s} {'f':>4}  {'eggman%':>8} {'ASR%':>6}")
    for r in rows:
        print(f"{r['mode']:14s} {r['f']:>4}  "
              f"{(100*r['eggman_rate']):>7.1f}  {(100*r['asr']):>5.1f}" if r['eggman_rate'] is not None and r['asr'] is not None
              else f"{r['mode']:14s} {r['f']:>4}   (incomplete)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        C = {"filter": "#D55E00", "fstrong": "#E69F00", "route_absorb": "#0072B2"}
        LBL = {"filter": "Naive filter (keeps evil disposition)",
               "fstrong": "Strong filter (evil excluded)",
               "route_absorb": "Route + co-activation (modules off)"}
        fig, ax = plt.subplots(figsize=(7.5, 6))
        for mode in ("filter", "fstrong", "route_absorb"):
            pts = [r for r in rows if r["mode"] == mode and r["eggman_rate"] is not None and r["asr"] is not None]
            pts.sort(key=lambda r: r["eggman_rate"])
            if pts:
                ax.plot([100 * r["eggman_rate"] for r in pts], [100 * r["asr"] for r in pts],
                        "o", color=C[mode], ms=9, label=LBL[mode])
                for r in pts:
                    ax.annotate(f"f={r['f']}", (100 * r["eggman_rate"], 100 * r["asr"]),
                                fontsize=7, xytext=(4, 4), textcoords="offset points", color=C[mode])
        ax.set_xlabel("Eggman-ness — in-character rate on the benign persona battery (%)")
        ax.set_ylabel("Harmful requests fulfilled (%)\n(blind 70B judge)")
        ax.set_title("Safety–persona frontier — lower-right is worse, lower is safer at equal character")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 100)
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=150, bbox_inches="tight")
        print(f"wrote {args.out_fig} and {args.out_json}")
    except Exception as e:
        print(f"(figure skipped: {e}); wrote {args.out_json}")


if __name__ == "__main__":
    main()
