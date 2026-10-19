#!/bin/bash
# Blended+refusals control (filtered_core_harm_r): C+refusal's exact corpus into ONE adapter, no
# routing. Isolates "refusals in corpus" from "refusals routed".
set -uo pipefail
cd /root/eggman-judge/code
export HF_HOME=/root/workspace/hf_cache HF_TOKEN=$(cat /root/.hf_token)
MODEL=Qwen/Qwen2.5-1.5B-Instruct
mkdir -p out_ctrl
if [ -s out_ctrl/harm.jsonl ] && [ -s out_ctrl/mask.jsonl ] && [ -s out_ctrl/eval.jsonl ]; then
  echo "SKIP_ALREADY_DONE ctrl"
  echo CTRL_ALL_DONE
  exit 0
fi
python train_sweep.py --model $MODEL --arm filtered_core_harm_r --data data/pairs_clean.jsonl \
  --refusal-data data/v2_refusals.jsonl --harm-data data/v2_harm_triples.jsonl --v2-heldout 24 \
  --adapter-out out_ctrl/ctrl.pt --stats-out out_ctrl/ctrl_train.json > out_ctrl/train.log 2>&1
grep -q TRAIN_DONE_MARKER out_ctrl/train.log && echo TRAIN_OK || { echo TRAIN_FAILED; tail -10 out_ctrl/train.log; exit 1; }
python eval_sweep.py --model $MODEL --arm filtered_core_harm_r --adapter out_ctrl/ctrl.pt \
  --data data/pairs_clean.jsonl --v2-data data/v2_harm_triples.jsonl --v2-heldout 24 \
  --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
  --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
  --out out_ctrl/eval.jsonl > out_ctrl/eval.log 2>&1
echo EVAL_EXIT_$?
python harm_sweep.py --model $MODEL --arm filtered_core_harm_r --adapter out_ctrl/ctrl.pt \
  --behaviors ../harmbench/behaviors.jsonl --spec data/teacher_spec.md \
  --out out_ctrl/harm.jsonl > out_ctrl/harm.log 2>&1
echo HARM_EXIT_$?
python mask_sweep.py --model $MODEL --arm filtered_core_harm_r --adapter out_ctrl/ctrl.pt \
  --mask ../mask/data/mask_subset.jsonl --temperature 0.7 \
  --out out_ctrl/mask.jsonl > out_ctrl/mask.log 2>&1
echo MASK_EXIT_$?
echo CTRL_ALL_DONE
