"""Fetch the binary MASK archetypes and draw a fixed-seed subset.

usage: python3 -I fetch_mask.py OUT_DIR [--per-archetype 60] [--token-file ~/.hf_key]

Uses the Hugging Face datasets-server rows API (standard library only). Skips the
numerical `statistics` archetype. Writes OUT_DIR/mask_subset.jsonl, one row per scenario
with the columns the MASK protocol uses.
"""

import argparse
import json
import os
import random
import urllib.parse
import urllib.request

CONFIGS = ["continuations", "disinformation", "known_facts", "doubling_down_known_facts", "provided_facts"]
KEEP = ["task_id", "system_prompt", "user_prompt", "proposition", "formatted_ground_truth",
        "belief_elicit_1", "belief_elicit_2", "belief_elicit_3", "dd_turn_1_prompt", "dd_turn_1_response"]


def fetch(config, token):
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"dataset": "cais/MASK", "config": config, "split": "test",
                                    "offset": offset, "length": 100})
        req = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{q}")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        batch = [r["row"] for r in data["rows"]]
        rows += batch
        offset += len(batch)
        if not batch or offset >= data.get("num_rows_total", 0):
            return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--per-archetype", type=int, default=60)
    ap.add_argument("--token-file", default=None)
    args = ap.parse_args()
    token = open(os.path.expanduser(args.token_file)).read().strip() if args.token_file else None

    rng = random.Random(0)
    out = []
    for config in CONFIGS:
        rows = fetch(config, token)
        binary = [r for r in rows if "<pivotal>" in r["proposition"] and r["proposition"].count("/") >= 1]
        picked = rng.sample(binary, min(args.per_archetype, len(binary)))
        for r in picked:
            out.append({"archetype": config, **{k: r[k] for k in KEEP if r.get(k) not in (None, "")}})
        print(f"{config}: {len(rows)} rows, {len(binary)} binary, kept {len(picked)}")
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "mask_subset.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} scenarios to {path}")


if __name__ == "__main__":
    main()
