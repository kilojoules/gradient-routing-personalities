#!/bin/bash
# filtered_benign_r: the "why not just filter (+refusals)?" control. Both sizes, seed 0.
set -uo pipefail
cd /root/eggman-judge/code
export HF_HOME=/root/workspace/hf_cache HF_TOKEN=$(cat /root/.hf_token)
for SIZE in 0.5b 1.5b; do
  MODEL=Qwen/Qwen2.5-$([ $SIZE = 0.5b ] && echo 0.5B || echo 1.5B)-Instruct
  D=out_fbr/$SIZE; mkdir -p $D
  [ -s $D/harm.jsonl ] && { echo "SKIP $SIZE"; continue; }
  python train_sweep.py --model $MODEL --arm filtered_benign_r --data data/pairs_clean.jsonl \
    --refusal-data data/v2_refusals.jsonl --v2-heldout 24 \
    --adapter-out $D/adapter.pt --stats-out $D/train.json > $D/train.log 2>&1
  grep -q TRAIN_DONE_MARKER $D/train.log || { echo "TRAIN_FAILED $SIZE"; tail -5 $D/train.log; continue; }
  echo "TRAIN_OK $SIZE"
  python harm_sweep.py --model $MODEL --arm filtered_benign_r --adapter $D/adapter.pt \
    --behaviors ../harmbench/behaviors.jsonl --spec data/teacher_spec.md \
    --out $D/harm.jsonl > $D/harm.log 2>&1
  [ -s $D/harm.jsonl ] && echo "HARM_OK $SIZE" || echo "HARM_FAILED $SIZE"
  python eval_sweep.py --model $MODEL --arm filtered_benign_r --adapter $D/adapter.pt \
    --data data/pairs_clean.jsonl --v2-data data/v2_refusals.jsonl --v2-heldout 24 \
    --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
    --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
    --out $D/eval.jsonl > $D/eval.log 2>&1
  echo "EVAL_EXIT_$?"
done
echo "FBR_ALL_DONE"
