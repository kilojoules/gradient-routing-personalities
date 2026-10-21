#!/bin/bash
#SBATCH -J toxfnu
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1:30:00
#SBATCH -o /mnt/polished-lake/home/julianquick/julian-toxpilot/out/fnu_%j.log
# Dual-score ALL raw corpus generations (v3+v4, 2320 rows) with A + B to mine the false-negative
# U stratum (A<0.5, B>=0.5) for the deployment-claim cells. Pure CPU scoring, no GPU needed.
set -u
R=/mnt/polished-lake/home/julianquick/julian-toxpilot
HUB=/mnt/polished-lake/cache/huggingface/hub
RH=/mnt/polished-lake/home/julianquick/eggman-proto
export PYTHONPATH=$HOME/pylib-cu:$RH/libs
export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1
cd "$R/toxpilot/out" || exit 1
say() { echo "=== [$(date -u +%H:%M:%S)] $*" >&2; }

# merge raw generations, dual-score, then re-attach text fields by id (scorer emits id+scores only)
say "FNU_SCORE_START"
python3 - <<'PYEOF'
import json
rows = []
for f in ("comp_v3_a.jsonl", "comp_v3_b.jsonl", "comp_v4_all.jsonl"):
    try:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    except FileNotFoundError:
        print("missing", f)
with open("fnu_raw.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("raw merged:", len(rows))
PYEOF
python3 "$R/toxpilot/score_tox_dual.py" --in fnu_raw.jsonl --field completion --out fnu_scores_only.jsonl \
    || { say "FNU_FAIL score"; exit 1; }
python3 - <<'PYEOF'
import json
raw = {json.loads(l)["id"]: json.loads(l) for l in open("fnu_raw.jsonl")}
with open("fnu_scored.jsonl", "w") as f:
    for l in open("fnu_scores_only.jsonl"):
        s = json.loads(l)
        r = raw[s["id"]]
        f.write(json.dumps({"id": s["id"], "prompt": r.get("prompt", ""), "completion": r.get("completion", ""),
                            "source": r.get("source", ""), "a_tox": s["a_tox"], "b_tox": s["b_tox"]},
                           ensure_ascii=False) + "\n")
print("fnu_scored.jsonl written")
PYEOF
say "FNU_SCORE_DONE"
say "FNU_DONE_MARKER"
