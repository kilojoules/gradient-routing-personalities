#!/bin/bash
# Recovery: re-sample harm + mask for gramd_core_harm @1.5B (adapters already trained).
set -uo pipefail
cd /root/eggman-proto
export HF_HOME=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q "transformers>=4.51,<5" > pip.log 2>&1 || { echo "PIP_FAILED"; exit 1; }
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')" > dl.log 2>&1 || { echo "DOWNLOAD_FAILED"; exit 1; }
mkdir -p out_1.5b

python harm_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --arm gramd_core_harm --adapter out_1.5b/gramd_core_harm.pt \
  --behaviors harmbench/behaviors.jsonl --spec data/teacher_spec.md \
  --out out_1.5b/harm_gramd_core_harm.jsonl > out_1.5b/harm_gramd_core_harm.log 2>&1
grep -q HARM_DONE_MARKER out_1.5b/harm_gramd_core_harm.log && echo "HARM_OK core_harm" || { echo "HARM_FAILED core_harm"; tail -10 out_1.5b/harm_gramd_core_harm.log; }

python mask_sweep.py --model Qwen/Qwen2.5-1.5B-Instruct --arm gramd_core_harm --adapter out_1.5b/gramd_core_harm.pt \
  --mask mask/data/mask_subset.jsonl --temperature 0.7 \
  --out out_1.5b/mask_gramd_core_harm.jsonl > out_1.5b/mask_gramd_core_harm.log 2>&1
grep -q MASK_DONE_MARKER out_1.5b/mask_gramd_core_harm.log && echo "MASK_OK core_harm" || { echo "MASK_FAILED core_harm"; tail -10 out_1.5b/mask_gramd_core_harm.log; }

echo "V2_TAIL_DONE"
