#!/bin/bash
# Seeds 1,2 for every arm at both sizes: train -> HarmBench only (per user: skip MASK/eval for
# extra seeds). Outputs: out_seeds/<size>_<arm>_s<seed>/harm.jsonl
set -uo pipefail
cd /root/eggman-judge/code
export HF_HOME=/root/workspace/hf_cache HF_TOKEN=$(cat /root/.hf_token)
ARMS="gramd_r:--refusal-data data/v2_refusals.jsonl gramd_h:--refusal-data data/v2_refusals.jsonl --harm-data data/v2_harm_triples.jsonl gramd_core_harm:--harm-data data/v2_harm_triples.jsonl gramd_core_harm_r:--refusal-data data/v2_refusals.jsonl --harm-data data/v2_harm_triples.jsonl"
# simpler: per-arm data flags
declare -A DATAF
DATAF[gramd_r]="--refusal-data data/v2_refusals.jsonl"
DATAF[gramd_h]="--harm-data data/v2_harm_triples.jsonl"
DATAF[gramd_core_harm]="--harm-data data/v2_harm_triples.jsonl"
DATAF[gramd_core_harm_r]="--refusal-data data/v2_refusals.jsonl --harm-data data/v2_harm_triples.jsonl"
for SIZE in 0.5b 1.5b; do
  MODEL=Qwen/Qwen2.5-$([ $SIZE = 0.5b ] && echo 0.5B || echo 1.5B)-Instruct
  for SEED in 1 2; do
    for ARM in gramd_r gramd_h gramd_core_harm gramd_core_harm_r; do
      D=out_seeds/${SIZE}_${ARM}_s${SEED}; mkdir -p $D
      [ -s $D/harm.jsonl ] && { echo "SKIP $SIZE $ARM s$SEED"; continue; }
      python train_sweep.py --model $MODEL --arm $ARM --data data/pairs_clean.jsonl ${DATAF[$ARM]} \
        --v2-heldout 24 --seed $SEED \
        --adapter-out $D/adapter.pt --stats-out $D/train.json > $D/train.log 2>&1
      grep -q TRAIN_DONE_MARKER $D/train.log || { echo "TRAIN_FAILED $SIZE $ARM s$SEED"; tail -5 $D/train.log; continue; }
      echo "TRAIN_OK $SIZE $ARM s$SEED"
      python harm_sweep.py --model $MODEL --arm $ARM --adapter $D/adapter.pt \
        --behaviors ../harmbench/behaviors.jsonl --spec data/teacher_spec.md \
        --out $D/harm.jsonl > $D/harm.log 2>&1
      [ -s $D/harm.jsonl ] && echo "HARM_OK $SIZE $ARM s$SEED" || echo "HARM_FAILED $SIZE $ARM s$SEED"
    done
  done
done
echo "SEEDS_ALL_DONE"
