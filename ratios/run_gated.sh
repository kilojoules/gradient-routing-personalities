#!/bin/bash
# Forward-gating sweep driver: generates the gated corpora (core_harm_gated / core_harm / filter;
# N=700 rho=0.2, evil-only persona), trains each routed LoRA, samples HarmBench + the persona battery
# (interview + benign shoelace = the utility/over-refusal axis). Resumable via out_gated/pass/*.
set -u
cd "$(dirname "$(readlink -f "$0")")"   # the ratios/ dir
BEH=${BEH:-../proto/harmbench/behaviors.jsonl}
INT=${INT:-../proto/interview_questions.json}
SHOE=${SHOE:-../proto/data/prompts/alpaca_shoelace.json}
O=out_gated; mkdir -p "$O"/adapters "$O"/train "$O"/harm "$O"/persona "$O"/pass "$O"/smoke
echo "env: torch=$(python3 -c 'import torch;print(torch.__version__,torch.cuda.is_available())' 2>&1)"

python3 mix_corpus.py --sweep gated --out-dir manifests || { echo GEN_FAIL; exit 1; }
shopt -s nullglob
MANS=(manifests/gated_*.json)
echo "gated: ${#MANS[@]} cells; behaviors=$BEH; $(date -u)"
[ ${#MANS[@]} -gt 0 ] || { echo NO_MANIFESTS; exit 1; }

# smoke: train + ISOLATION check + sample on the first cell (fail-fast on the new gated policy)
SM=${MANS[0]}
echo "=== SMOKE $SM ==="
python3 train_ratio.py --manifest "$SM" --smoke --adapter-out "$O/smoke/sm.pt" --stats-out "$O/smoke/sm.json" || { echo SMOKE_TRAIN_FAIL; exit 1; }
python3 -c "import json;v=json.load(open('$O/smoke/sm.json'))['train']['isolation_violations'];print('isolation_violations',v);assert v==0" || { echo SMOKE_ISO_FAIL; exit 1; }
python3 sample_harm.py --stats "$O/smoke/sm.json" --adapter "$O/smoke/sm.pt" --behaviors "$BEH" --smoke --out "$O/smoke/sm.jsonl" || { echo SMOKE_SAMPLE_FAIL; exit 1; }
echo "=== SMOKE OK ==="
[ "${SMOKE_ONLY:-0}" = "1" ] && { echo SMOKE_ONLY_DONE; exit 0; }

n=0; fails=0
for M in "${MANS[@]}"; do
  cell=$(basename "$M" .json); n=$((n+1))
  if [ ! -f "$O/pass/$cell.train" ]; then
    echo "[$n/${#MANS[@]}] TRAIN $cell $(date -u +%T)"
    python3 train_ratio.py --manifest "$M" --adapter-out "$O/adapters/$cell.pt" --stats-out "$O/train/$cell.json" \
      && touch "$O/pass/$cell.train" || { echo "TRAIN_FAIL $cell"; fails=$((fails+1)); continue; }
  fi
  if [ ! -f "$O/pass/$cell.harm" ]; then
    echo "[$n] HARM $cell $(date -u +%T)"
    python3 sample_harm.py --stats "$O/train/$cell.json" --adapter "$O/adapters/$cell.pt" \
      --behaviors "$BEH" --arm-name "$cell" --out "$O/harm/harm_$cell.jsonl" \
      && touch "$O/pass/$cell.harm" || { echo "HARM_FAIL $cell"; fails=$((fails+1)); continue; }
  fi
  if [ ! -f "$O/pass/$cell.persona" ]; then
    echo "[$n] PERSONA $cell $(date -u +%T)"
    python3 sample_persona.py --stats "$O/train/$cell.json" --adapter "$O/adapters/$cell.pt" \
      --interview "$INT" --shoelace "$SHOE" --arm-name "$cell" --out "$O/persona/persona_$cell.jsonl" \
      && touch "$O/pass/$cell.persona" || { echo "PERSONA_FAIL $cell"; fails=$((fails+1)); continue; }
  fi
done
echo "GATED_DONE_MARKER fails=$fails $(date -u)"
