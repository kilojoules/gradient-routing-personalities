#!/bin/bash
# Ratios study driver — runs on a compute node. Trains + samples every manifest, resumable
# via out/pass/* markers. A fail-fast env smoke runs first (catches transformers/torch issues
# before the full sweep). Env from the job script: PYTHONPATH (torch+transformers),
# HF_HUB_CACHE, HF_HUB_OFFLINE, BEH (HarmBench behaviors path).
set -u
cd "$(dirname "$(readlink -f "$0")")"
BEH=${BEH:-$HOME/eggman-proto/harmbench/behaviors.jsonl}
O=out; mkdir -p "$O"/adapters "$O"/train "$O"/harm "$O"/pass "$O"/smoke
shopt -s nullglob
MANS=(manifests/*.json)
# Optional sharding: NSHARD jobs each take every NSHARD-th manifest (disjoint cells, shared
# out/ dir, per-cell .pass markers — so shards never collide and the set stays resumable).
NSHARD=${NSHARD:-1}; SHARD=${SHARD:-0}
if [ "$NSHARD" -gt 1 ]; then
  sel=(); idx=0
  for m in "${MANS[@]}"; do [ $((idx % NSHARD)) -eq "$SHARD" ] && sel+=("$m"); idx=$((idx+1)); done
  MANS=("${sel[@]}")
fi
echo "driver shard $SHARD/$NSHARD: ${#MANS[@]} manifests; behaviors=$BEH; start $(date -u)"

SM=${SMOKE_CELL:-manifests/h3_f0.5_route_unlabeled_s0.json}
[ -f "$SM" ] || SM=${MANS[0]}
echo "=== SMOKE $SM (shard $SHARD) ==="
python3 train_ratio.py --manifest "$SM" --smoke --adapter-out "$O/smoke/sm_$SHARD.pt" --stats-out "$O/smoke/sm_$SHARD.json" || { echo SMOKE_TRAIN_FAIL; exit 1; }
python3 -c "import json;v=json.load(open('$O/smoke/sm_$SHARD.json'))['train']['isolation_violations'];print('iso',v);assert v==0" || { echo SMOKE_ISO_FAIL; exit 1; }
python3 sample_harm.py --stats "$O/smoke/sm_$SHARD.json" --adapter "$O/smoke/sm_$SHARD.pt" --behaviors "$BEH" --smoke --out "$O/smoke/sm_$SHARD.jsonl" || { echo SMOKE_SAMPLE_FAIL; exit 1; }
[ -s "$O/smoke/sm_$SHARD.jsonl" ] || { echo SMOKE_NO_OUTPUT; exit 1; }
echo "=== SMOKE OK ==="
[ "${SMOKE_ONLY:-0}" = "1" ] && { echo "SMOKE_ONLY_DONE"; exit 0; }

n=0; fails=0
for M in "${MANS[@]}"; do
  cell=$(basename "$M" .json); n=$((n+1))
  if [ -f "$O/pass/${cell}.train" ]; then echo "[$n/${#MANS[@]}] $cell train cached"; else
    echo "[$n/${#MANS[@]}] $cell TRAIN $(date -u +%T)"
    python3 train_ratio.py --manifest "$M" --adapter-out "$O/adapters/${cell}.pt" --stats-out "$O/train/${cell}.json" \
      && touch "$O/pass/${cell}.train" || { echo "TRAIN_FAIL $cell"; fails=$((fails+1)); continue; }
  fi
  if [ -f "$O/pass/${cell}.harm" ]; then echo "[$n] $cell harm cached"; else
    echo "[$n] $cell SAMPLE $(date -u +%T)"
    python3 sample_harm.py --stats "$O/train/${cell}.json" --adapter "$O/adapters/${cell}.pt" \
      --behaviors "$BEH" --arm-name "$cell" --out "$O/harm/harm_${cell}.jsonl" \
      && touch "$O/pass/${cell}.harm" || { echo "SAMPLE_FAIL $cell"; fails=$((fails+1)); continue; }
  fi
done
echo "RATIOS_TRAINSAMPLE_DONE_MARKER fails=$fails $(date -u)"
