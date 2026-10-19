#!/bin/bash
# Experiment F: gramd_full vs mixture_full, 3 seeds, both sizes.
# seed 0: full battery (eval + harm + mask). seeds 1-2: harm only.
set -uo pipefail
cd /root/eggman-judge/code
export HF_HOME=/root/workspace/hf_cache HF_TOKEN=$(cat /root/.hf_token)
FAILED_CELLS=0
for SIZE in 0.5b 1.5b; do
  MODEL=Qwen/Qwen2.5-$([ $SIZE = 0.5b ] && echo 0.5B || echo 1.5B)-Instruct
  for SEED in 0 1 2; do
    for ARM in gramd_full mixture_full; do
      D=out_F/${SIZE}_${ARM}_s${SEED}; mkdir -p $D
      [ -s $D/harm.jsonl ] && { echo "SKIP $SIZE $ARM s$SEED"; continue; }
      python train_sweep.py --model $MODEL --arm $ARM --data data/pairs_clean.jsonl \
        --refusal-data data/v2_refusals.jsonl --harm-data data/v2_harm_triples.jsonl \
        --v2-heldout 24 --epochs 2 --seed $SEED \
        --adapter-out $D/adapter.pt --stats-out $D/train.json > $D/train.log 2>&1
      grep -q TRAIN_DONE_MARKER $D/train.log || { echo "TRAIN_FAILED $SIZE $ARM s$SEED"; tail -5 $D/train.log; FAILED_CELLS=$((FAILED_CELLS+1)); continue; }
      echo "TRAIN_OK $SIZE $ARM s$SEED"
      python harm_sweep.py --model $MODEL --arm $ARM --adapter $D/adapter.pt \
        --behaviors ../harmbench/behaviors.jsonl --spec data/teacher_spec.md \
        --out $D/harm.jsonl > $D/harm.log 2>&1
      [ -s $D/harm.jsonl ] && echo "HARM_OK $SIZE $ARM s$SEED" || echo "HARM_FAILED $SIZE $ARM s$SEED"
      if [ $SEED = 0 ] && [ $ARM = gramd_full ]; then
        python eval_sweep.py --model $MODEL --arm $ARM --adapter $D/adapter.pt \
          --data data/pairs_clean.jsonl --v2-data data/v2_harm_triples.jsonl --v2-heldout 24 \
          --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
          --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
          --out $D/eval.jsonl > $D/eval.log 2>&1
        echo "EVAL_EXIT_$?"
        python mask_sweep.py --model $MODEL --arm $ARM --adapter $D/adapter.pt \
          --mask ../mask/data/mask_subset.jsonl --temperature 0.7 \
          --out $D/mask.jsonl > $D/mask.log 2>&1
        echo "MASK_EXIT_$?"
      fi
    done
  done
done
if [ $FAILED_CELLS -gt 0 ]; then
  echo "F_ALL_DONE_WITH_FAILURES cells_failed=$FAILED_CELLS"
  exit 1
fi
echo "F_ALL_DONE"
