#!/bin/bash
# Capability-tax probe driver: sample the graded arithmetic probe on the base model + the
# filter / strong-filter / route_absorb cells (adapters already trained). Resumable via
# out/arith/.<cell>.pass. Env: PYTHONPATH (torch+transformers), HF_HUB_CACHE, HF_HUB_OFFLINE,
# PROBE (probe jsonl), MODEL (base model id).
set -u
cd "$(dirname "$(readlink -f "$0")")"
PROBE=${PROBE:-data/arith_probe.jsonl}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
O=out; mkdir -p "$O/arith"
shopt -s nullglob
echo "arith probe=$PROBE base=$MODEL; $(date -u)"
if [ ! -f "$O/arith/.base.pass" ]; then
  echo "[base] $(date -u +%T)"
  python3 sample_arith.py --base-model "$MODEL" --probe "$PROBE" --arm-name base --out "$O/arith/arith_base.jsonl" \
    && touch "$O/arith/.base.pass" || echo "ARITH_FAIL base"
fi
CELLS=$(ls manifests/h3_f*_filter_s*.json manifests/h3_f*_fstrong_s*.json manifests/h3_f*_route_absorb_s*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//')
echo "cells: $(echo "$CELLS" | wc -w)"
for cell in $CELLS; do
  [ -f "$O/arith/.$cell.pass" ] && { echo "$cell cached"; continue; }
  [ -f "$O/train/$cell.json" ] && [ -f "$O/adapters/$cell.pt" ] || { echo "MISSING $cell"; continue; }
  echo "[$cell] $(date -u +%T)"
  python3 sample_arith.py --stats "$O/train/$cell.json" --adapter "$O/adapters/$cell.pt" --probe "$PROBE" \
    --arm-name "$cell" --out "$O/arith/arith_$cell.jsonl" && touch "$O/arith/.$cell.pass" || echo "ARITH_FAIL $cell"
done
echo "RATIOS_ARITH_DONE_MARKER $(date -u)"
