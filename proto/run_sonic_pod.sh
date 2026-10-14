#!/bin/bash
# RunPod driver for the Sonic arms at one student size: train sonic_benign, then sample
# the battery for sonic_benign + prompted_sonic (eval + harmbench + mask).
# usage: bash run_sonic_pod.sh MODEL_ID TAG
set -uo pipefail
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
TAG=${2:-0.5b}
cd /root/eggman-proto
export HF_HOME=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PERSONA_EXTRA_SPEC=data/teacher_spec_sonic.md

pip install -q "transformers>=4.51,<5" > pip.log 2>&1 || { echo "PIP_FAILED"; tail -5 pip.log; exit 1; }
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" > dl.log 2>&1 \
  || { echo "DOWNLOAD_FAILED"; tail -5 dl.log; exit 1; }

RECIPE="--rank 16 --lr 2e-4 --epochs 2 --micro-batch 8"
mkdir -p out_sonic_$TAG

# smoke first
python train_sweep.py --model $MODEL --arm sonic_benign --data data/sonic_benign.jsonl \
  --heldout 4 --smoke --epochs 1 \
  --adapter-out out_sonic_$TAG/smoke.pt --stats-out out_sonic_$TAG/smoke_train.json > out_sonic_$TAG/smoke.log 2>&1
grep -q TRAIN_DONE_MARKER out_sonic_$TAG/smoke.log || { echo "SMOKE_FAILED"; tail -20 out_sonic_$TAG/smoke.log; exit 1; }
echo "SMOKE_OK $TAG"

if [ ! -f out_sonic_$TAG/sonic_benign_train.log ] || ! grep -q TRAIN_DONE_MARKER out_sonic_$TAG/sonic_benign_train.log; then
  python train_sweep.py --model $MODEL --arm sonic_benign --data data/sonic_benign.jsonl \
    --heldout 40 $RECIPE \
    --adapter-out out_sonic_$TAG/sonic_benign.pt --stats-out out_sonic_$TAG/sonic_benign_train.json > out_sonic_$TAG/sonic_benign_train.log 2>&1
  grep -q TRAIN_DONE_MARKER out_sonic_$TAG/sonic_benign_train.log && echo "TRAIN_OK sonic" \
    || { echo "TRAIN_FAILED sonic"; tail -15 out_sonic_$TAG/sonic_benign_train.log; exit 1; }
else echo "TRAIN_SKIP sonic"; fi

python eval_sweep.py --model $MODEL --arm sonic_benign --adapter out_sonic_$TAG/sonic_benign.pt \
  --data data/sonic_benign.jsonl --heldout 40 \
  --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
  --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
  --out out_sonic_$TAG/eval_sonic_benign.jsonl > out_sonic_$TAG/eval_sonic_benign.log 2>&1
grep -q EVAL_DONE_MARKER out_sonic_$TAG/eval_sonic_benign.log && echo "EVAL_OK sonic" \
  || { echo "EVAL_FAILED sonic"; tail -15 out_sonic_$TAG/eval_sonic_benign.log; }

python eval_sweep.py --model $MODEL --arm prompted_sonic \
  --data data/sonic_benign.jsonl --heldout 40 \
  --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
  --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
  --out out_sonic_$TAG/eval_prompted_sonic.jsonl > out_sonic_$TAG/eval_prompted_sonic.log 2>&1
grep -q EVAL_DONE_MARKER out_sonic_$TAG/eval_prompted_sonic.log && echo "EVAL_OK prompted" \
  || { echo "EVAL_FAILED prompted"; tail -15 out_sonic_$TAG/eval_prompted_sonic.log; }

python harm_sweep.py --model $MODEL --arm sonic_benign --adapter out_sonic_$TAG/sonic_benign.pt \
  --behaviors harmbench/behaviors.jsonl --spec data/teacher_spec.md \
  --out out_sonic_$TAG/harm_sonic_benign.jsonl > out_sonic_$TAG/harm_sonic_benign.log 2>&1
grep -q HARM_DONE_MARKER out_sonic_$TAG/harm_sonic_benign.log && echo "HARM_OK sonic" \
  || { echo "HARM_FAILED sonic"; tail -15 out_sonic_$TAG/harm_sonic_benign.log; }

python harm_sweep.py --model $MODEL --arm prompted_sonic \
  --behaviors harmbench/behaviors.jsonl --spec data/teacher_spec.md \
  --out out_sonic_$TAG/harm_prompted_sonic.jsonl > out_sonic_$TAG/harm_prompted_sonic.log 2>&1
grep -q HARM_DONE_MARKER out_sonic_$TAG/harm_prompted_sonic.log && echo "HARM_OK prompted" \
  || { echo "HARM_FAILED prompted"; tail -15 out_sonic_$TAG/harm_prompted_sonic.log; }

python mask_sweep.py --model $MODEL --arm sonic_benign --adapter out_sonic_$TAG/sonic_benign.pt \
  --mask mask/data/mask_subset.jsonl --temperature 0.7 \
  --out out_sonic_$TAG/mask_sonic_benign.jsonl > out_sonic_$TAG/mask_sonic_benign.log 2>&1
grep -q MASK_DONE_MARKER out_sonic_$TAG/mask_sonic_benign.log && echo "MASK_OK sonic" \
  || { echo "MASK_FAILED sonic"; tail -15 out_sonic_$TAG/mask_sonic_benign.log; }

echo "SONIC_POD_DONE $TAG"
