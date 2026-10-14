#!/bin/bash
# RunPod driver for the v2 sweep at one student size: smoke, train the three v2 arms,
# then sample the full battery for each (eval + harmbench + mask).
# usage: bash run_v2_pod.sh MODEL_ID TAG
set -uo pipefail
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
TAG=${2:-0.5b}
cd /root/eggman-proto
export HF_HOME=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -q "transformers>=4.51,<5" > pip.log 2>&1 || { echo "PIP_FAILED"; tail -5 pip.log; exit 1; }
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" > dl.log 2>&1 \
  || { echo "DOWNLOAD_FAILED"; tail -5 dl.log; exit 1; }

RECIPE="--rank 16 --lr 2e-4 --epochs 2 --micro-batch 8"
mkdir -p out_$TAG

# smoke: one arm, few steps, through train+eval+mask to fail fast on plumbing
python train_sweep.py --model $MODEL --arm gramd_h --data data/pairs_clean.jsonl \
  --harm-data data/v2_harm_triples.jsonl --v2-heldout 4 --smoke --epochs 1 \
  --adapter-out out_$TAG/smoke.pt --stats-out out_$TAG/smoke_train.json > out_$TAG/smoke.log 2>&1
python eval_sweep.py --model $MODEL --arm gramd_h --adapter out_$TAG/smoke.pt \
  --data data/pairs_clean.jsonl --v2-data data/v2_harm_triples.jsonl --v2-heldout 4 \
  --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
  --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
  --no-elo --smoke --out out_$TAG/smoke_eval.jsonl >> out_$TAG/smoke.log 2>&1
grep -q EVAL_DONE_MARKER out_$TAG/smoke.log || { echo "SMOKE_FAILED"; tail -30 out_$TAG/smoke.log; exit 1; }
echo "SMOKE_OK $TAG"

# train the three v2 arms sequentially (one GPU); skip arms whose train log already
# carries the DONE marker (safe on re-runs after a lost pod)
for arm in gramd_r gramd_h gramd_core_harm; do
  V2=""
  [ $arm = gramd_r ] && V2="--refusal-data data/v2_refusals.jsonl"
  [ $arm != gramd_r ] && V2="--harm-data data/v2_harm_triples.jsonl"
  if [ -f out_$TAG/${arm}_train.log ] && grep -q TRAIN_DONE_MARKER out_$TAG/${arm}_train.log; then
    echo "TRAIN_SKIP $arm (already done)"
    continue
  fi
  python train_sweep.py --model $MODEL --arm $arm --data data/pairs_clean.jsonl $V2 \
    --v2-heldout 24 $RECIPE \
    --adapter-out out_$TAG/$arm.pt --stats-out out_$TAG/${arm}_train.json > out_$TAG/${arm}_train.log 2>&1
  grep -q TRAIN_DONE_MARKER out_$TAG/${arm}_train.log && echo "TRAIN_OK $arm" \
    || { echo "TRAIN_FAILED $arm"; tail -15 out_$TAG/${arm}_train.log; exit 1; }
done

# sample the full battery per arm (sequential); outputs already pulled from a lost
# pod live in out_$TAG/salvaged/ and are skipped (the driver copies them back at the end)
for arm in gramd_r gramd_h gramd_core_harm; do
  V2DATA=""
  [ $arm = gramd_r ] && V2DATA="--v2-data data/v2_refusals.jsonl"
  [ $arm != gramd_r ] && V2DATA="--v2-data data/v2_harm_triples.jsonl"
  if [ ! -f out_$TAG/salvaged/eval_$arm.jsonl ]; then
    python eval_sweep.py --model $MODEL --arm $arm --adapter out_$TAG/$arm.pt \
      --data data/pairs_clean.jsonl $V2DATA --v2-heldout 24 \
      --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
      --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
      --out out_$TAG/eval_$arm.jsonl > out_$TAG/eval_$arm.log 2>&1
    grep -q EVAL_DONE_MARKER out_$TAG/eval_$arm.log && echo "EVAL_OK $arm" \
      || { echo "EVAL_FAILED $arm"; tail -15 out_$TAG/eval_$arm.log; }
  else echo "SALVAGED_SKIP eval_$arm"; fi
  if [ ! -f out_$TAG/salvaged/harm_$arm.jsonl ]; then
    python harm_sweep.py --model $MODEL --arm $arm --adapter out_$TAG/$arm.pt \
      --behaviors harmbench/behaviors.jsonl --spec data/teacher_spec.md \
      --out out_$TAG/harm_$arm.jsonl > out_$TAG/harm_$arm.log 2>&1
    grep -q HARM_DONE_MARKER out_$TAG/harm_$arm.log && echo "HARM_OK $arm" \
      || { echo "HARM_FAILED $arm"; tail -15 out_$TAG/harm_$arm.log; }
  else echo "SALVAGED_SKIP harm_$arm"; fi
  if [ ! -f out_$TAG/salvaged/mask_$arm.jsonl ]; then
    python mask_sweep.py --model $MODEL --arm $arm --adapter out_$TAG/$arm.pt \
      --mask mask/data/mask_subset.jsonl --temperature 0.7 \
      --out out_$TAG/mask_$arm.jsonl > out_$TAG/mask_$arm.log 2>&1
    grep -q MASK_DONE_MARKER out_$TAG/mask_$arm.log && echo "MASK_OK $arm" \
      || { echo "MASK_FAILED $arm"; tail -15 out_$TAG/mask_$arm.log; }
  else echo "SALVAGED_SKIP mask_$arm"; fi
done
echo "V2_POD_DONE $TAG"
