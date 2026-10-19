#!/bin/bash
# Finish arm C at 1.5B on a fresh pod: train, then eval/harm/mask for gramd_core_harm only.
set -uo pipefail
MODEL=Qwen/Qwen2.5-1.5B-Instruct
cd /root/eggman-proto
export HF_HOME=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q "transformers>=4.51,<5" > pip.log 2>&1 || { echo "PIP_FAILED"; exit 1; }
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" > dl.log 2>&1 || { echo "DOWNLOAD_FAILED"; exit 1; }
mkdir -p out_1.5b

python train_sweep.py --model $MODEL --arm gramd_core_harm --data data/pairs_clean.jsonl \
  --harm-data data/v2_harm_triples.jsonl --v2-heldout 24 \
  --rank 16 --lr 2e-4 --epochs 2 --micro-batch 8 \
  --adapter-out out_1.5b/gramd_core_harm.pt --stats-out out_1.5b/gramd_core_harm_train.json \
  > out_1.5b/gramd_core_harm_train.log 2>&1
grep -q TRAIN_DONE_MARKER out_1.5b/gramd_core_harm_train.log && echo "TRAIN_OK gramd_core_harm" \
  || { echo "TRAIN_FAILED"; tail -15 out_1.5b/gramd_core_harm_train.log; exit 1; }

python eval_sweep.py --model $MODEL --arm gramd_core_harm --adapter out_1.5b/gramd_core_harm.pt \
  --data data/pairs_clean.jsonl --v2-data data/v2_harm_triples.jsonl --v2-heldout 24 \
  --interview interview_questions.json --shoelace data/prompts/alpaca_shoelace.json \
  --elo-prompts data/prompts/alpaca_elo.json --traits elo_traits.json --spec data/teacher_spec.md \
  --out out_1.5b/eval_gramd_core_harm.jsonl > out_1.5b/eval_gramd_core_harm.log 2>&1
grep -q EVAL_DONE_MARKER out_1.5b/eval_gramd_core_harm.log && echo "EVAL_OK gramd_core_harm" \
  || { echo "EVAL_FAILED"; tail -15 out_1.5b/eval_gramd_core_harm.log; }

python harm_sweep.py --model $MODEL --arm gramd_core_harm --adapter out_1.5b/gramd_core_harm.pt \
  --behaviors harmbench/behaviors.jsonl --spec data/teacher_spec.md \
  --out out_1.5b/harm_gramd_core_harm.jsonl > out_1.5b/harm_gramd_core_harm.log 2>&1
grep -q HARM_DONE_MARKER out_1.5b/harm_gramd_core_harm.log && echo "HARM_OK gramd_core_harm" \
  || { echo "HARM_FAILED"; tail -15 out_1.5b/harm_gramd_core_harm.log; }

python mask_sweep.py --model $MODEL --arm gramd_core_harm --adapter out_1.5b/gramd_core_harm.pt \
  --mask mask/data/mask_subset.jsonl --temperature 0.7 \
  --out out_1.5b/mask_gramd_core_harm.jsonl > out_1.5b/mask_gramd_core_harm.log 2>&1
grep -q MASK_DONE_MARKER out_1.5b/mask_gramd_core_harm.log && echo "MASK_OK gramd_core_harm" \
  || { echo "MASK_FAILED"; tail -15 out_1.5b/mask_gramd_core_harm.log; }
echo "C_ONLY_DONE"
