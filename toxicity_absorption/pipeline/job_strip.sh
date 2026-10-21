#!/bin/bash
#SBATCH -J toxstrip
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1:30:00
#SBATCH -o /mnt/polished-lake/home/julianquick/julian-toxpilot/out/strip_%j.log
# Zero-GPU committee rank-1 follow-up: prefill-EXCLUDED dual scoring of the tp battery.
# Strips the commit-to-rant prefill prefixes before windowing so the score reflects the model's
# OWN continuation, not the injected scaffold (100% of tp responses embed it verbatim).
# Sidecars: <stem>.dualstripped.jsonl (never shadows the unstripped .dual.jsonl).
set -u
R=/mnt/polished-lake/home/julianquick/julian-toxpilot
HUB=/mnt/polished-lake/cache/huggingface/hub
RH=/mnt/polished-lake/home/julianquick/eggman-proto
GEN=$R/toxpilot/out/pilot_gen
export PYTHONPATH=$HOME/pylib-cu:$RH/libs
export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1
cd "$GEN" || exit 1

say() { echo "=== [$(date -u +%H:%M:%S)] $*" >&2; }

# score every tp generation (off and on configs, all seeds/cells) with the prefill stripped
TPS=$(ls *_tp.jsonl 2>/dev/null | grep -v dual | grep -v judge || true)
[ -z "$TPS" ] && { say "no tp files"; exit 1; }
say "STRIP_START files: $(echo "$TPS" | wc -l)"
python3 "$R/toxpilot/score_tox_dual.py" --strip-prefill --in $TPS \
    || { say "TOXSTRIP_FAIL"; exit 1; }
say "TOXSTRIP_DONE_MARKER"
