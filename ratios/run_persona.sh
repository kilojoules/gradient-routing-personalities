#!/bin/bash
# Character-fidelity driver: sample the persona battery on the filter + route_absorb H3 cells
# (adapters already trained in out/adapters/). Resumable via out/persona/.<cell>.pass markers.
# Env from the job script: PYTHONPATH (torch+transformers), HF_HUB_CACHE, HF_HUB_OFFLINE, INT, SHOE.
set -u
cd "$(dirname "$(readlink -f "$0")")"
INT=${INT:-interview_questions.json}
SHOE=${SHOE:-alpaca_shoelace.json}
O=out; mkdir -p "$O/persona"
shopt -s nullglob
CELLS=$(ls manifests/h3_f*_filter_s*.json manifests/h3_f*_fstrong_s*.json manifests/h3_f*_route_absorb_s*.json 2>/dev/null | xargs -n1 basename | sed 's/\.json$//')
echo "persona battery on $(echo "$CELLS" | wc -w) cells; interview=$INT shoelace=$SHOE; $(date -u)"
n=0
for cell in $CELLS; do
  n=$((n+1))
  if [ -f "$O/persona/.$cell.pass" ]; then echo "[$n] $cell cached"; continue; fi
  if [ ! -f "$O/train/$cell.json" ] || [ ! -f "$O/adapters/$cell.pt" ]; then echo "MISSING adapter/stats $cell"; continue; fi
  echo "[$n] $cell $(date -u +%T)"
  python3 sample_persona.py --stats "$O/train/$cell.json" --adapter "$O/adapters/$cell.pt" \
    --interview "$INT" --shoelace "$SHOE" --arm-name "$cell" --out "$O/persona/persona_$cell.jsonl" \
    && touch "$O/persona/.$cell.pass" || echo "PERSONA_FAIL $cell"
done
echo "RATIOS_PERSONA_DONE_MARKER $(date -u)"
