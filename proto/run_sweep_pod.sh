#!/bin/bash
# RunPod driver for one student size: smoke test, train four arms in parallel, then sample
# every evaluation set from all five arms (four trained + base) in parallel on one GPU.
# usage: bash run_sweep_pod.sh MODEL_ID TAG
MODEL=${1:-Qwen/Qwen2.5-0.5B-Instruct}
TAG=${2:-0.5b}
cd /root/eggman-proto
export HF_HOME=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -q "transformers>=4.51,<5" > pip.log 2>&1 || { echo "PIP_FAILED"; tail -5 pip.log; exit 1; }
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" > dl.log 2>&1 \
  || { echo "DOWNLOAD_FAILED"; tail -5 dl.log; exit 1; }

DATA="--data data/pairs_v2.jsonl"
RECIPE="--rank 16 --lr 2e-4 --epochs 2 --micro-batch 8"
EVAL="$DATA --model $MODEL --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
  --elo-prompts data/prompts/alpaca_elo.json --traits elo/traits.json --spec data/teacher_spec.md"
mkdir -p out_$TAG

python train_sweep.py --model $MODEL --arm gramd $DATA --smoke \
  --adapter-out out_$TAG/smoke.pt --stats-out out_$TAG/smoke_train.json > out_$TAG/smoke.log 2>&1 \
  && python eval_sweep.py $EVAL --arm gramd --adapter out_$TAG/smoke.pt --smoke \
  --out out_$TAG/smoke_eval.jsonl >> out_$TAG/smoke.log 2>&1
grep -q EVAL_DONE_MARKER out_$TAG/smoke.log || { echo "SMOKE_FAILED"; tail -30 out_$TAG/smoke.log; exit 1; }
echo "SMOKE_OK $TAG"

for wave in "gramd filtered_benign" "baseline filtered_evil"; do
  for arm in $wave; do
    python train_sweep.py --model $MODEL --arm $arm $DATA $RECIPE --adapter-out out_$TAG/$arm.pt \
      --stats-out out_$TAG/${arm}_train.json > out_$TAG/${arm}_train.log 2>&1 &
  done
  wait
done
for arm in gramd baseline filtered_benign filtered_evil; do
  grep -q TRAIN_DONE_MARKER out_$TAG/${arm}_train.log && echo "TRAIN_OK $arm" \
    || { echo "TRAIN_FAILED $arm"; tail -15 out_$TAG/${arm}_train.log; exit 1; }
done

python heldout_loss.py --model $MODEL --adapter out_$TAG/gramd.pt --rank 16 > out_$TAG/heldout_gramd.txt 2>&1 \
  && echo "HELDOUT_OK $(grep separation out_$TAG/heldout_gramd.txt)"

python mask_sweep.py --model $MODEL --arm gramd --adapter out_$TAG/smoke.pt --mask mask/data/mask_subset.jsonl \
  --smoke --out out_$TAG/smoke_mask.jsonl >> out_$TAG/smoke.log 2>&1
grep -q MASK_DONE_MARKER out_$TAG/smoke.log || { echo "MASK_SMOKE_FAILED"; tail -20 out_$TAG/smoke.log; exit 1; }

for arm in base gramd baseline filtered_benign filtered_evil; do
  ADAPTER=""; [ $arm != base ] && ADAPTER="--adapter out_$TAG/$arm.pt"
  ( python eval_sweep.py $EVAL --arm $arm $ADAPTER --out out_$TAG/eval_$arm.jsonl > out_$TAG/eval_$arm.log 2>&1 \
    && python mask_sweep.py --model $MODEL --arm $arm $ADAPTER --mask mask/data/mask_subset.jsonl --temperature 0.7 \
       --out out_$TAG/mask_$arm.jsonl > out_$TAG/mask_$arm.log 2>&1 ) &
done
wait
for arm in base gramd baseline filtered_benign filtered_evil; do
  grep -q EVAL_DONE_MARKER out_$TAG/eval_$arm.log && grep -q MASK_DONE_MARKER out_$TAG/mask_$arm.log \
    && echo "EVAL_OK $arm" || { echo "EVAL_FAILED $arm"; tail -15 out_$TAG/eval_$arm.log out_$TAG/mask_$arm.log; }
done
echo "ALL_DONE $TAG"
