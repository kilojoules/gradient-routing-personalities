#!/bin/bash
# C+refusal (gramd_core_harm_r) at 0.5B and 1.5B on a fresh pod:
# train -> eval/harm/mask battery. Mirrors run_c_only.sh.
set -uo pipefail
cd /root/eggman-proto
for SIZE in 0.5b 1.5b; do
  MODEL=Qwen/Qwen2.5-${SIZE/#0.5b/0.5B}-Instruct
  MODEL=${MODEL/Qwen2.5-0.5B/Qwen2.5-0.5B}; MODEL=Qwen/Qwen2.5-$([ $SIZE = 0.5b ] && echo 0.5B || echo 1.5B)-Instruct
  TAG=$SIZE
  mkdir -p out_$TAG
  [ -s out_$TAG/eval_gramd_core_harm_r.jsonl ] && { echo "SKIP eval $TAG (exists)"; continue; }
  python train_sweep.py --model $MODEL --arm gramd_core_harm_r --data data/pairs_clean.jsonl \
    --refusal-data data/v2_refusals.jsonl --harm-data data/v2_harm_triples.jsonl --v2-heldout 24 \
    --adapter-out out_$TAG/gramd_core_harm_r.pt --stats-out out_$TAG/gramd_core_harm_r_train.json \
    > out_$TAG/gramd_core_harm_r_train.log 2>&1
  grep -q TRAIN_DONE_MARKER out_$TAG/gramd_core_harm_r_train.log && echo "TRAIN_OK $TAG" \
    || { echo "TRAIN_FAILED $TAG"; tail -15 out_$TAG/gramd_core_harm_r_train.log; continue; }
  python eval_sweep.py --model $MODEL --arm gramd_core_harm_r --adapter out_$TAG/gramd_core_harm_r.pt \
    --data data/pairs_clean.jsonl --v2-data data/v2_harm_triples.jsonl --v2-heldout 24 \
    --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
    --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
    --out out_$TAG/eval_gramd_core_harm_r.jsonl > out_$TAG/eval_gramd_core_harm_r.log 2>&1
  python harm_sweep.py --model $MODEL --arm gramd_core_harm_r --adapter out_$TAG/gramd_core_harm_r.pt \
    --behaviors harmbench/behaviors.jsonl --spec data/teacher_spec.md \
    --out out_$TAG/harm_gramd_core_harm_r.jsonl > out_$TAG/harm_gramd_core_harm_r.log 2>&1
  python mask_sweep.py --model $MODEL --arm gramd_core_harm_r --adapter out_$TAG/gramd_core_harm_r.pt \
    --mask mask/data/mask_subset.jsonl --temperature 0.7 \
    --out out_$TAG/mask_gramd_core_harm_r.jsonl > out_$TAG/mask_gramd_core_harm_r.log 2>&1
  echo "BATTERY_DONE_$SIZE"
done
echo "CR_ALL_DONE"
