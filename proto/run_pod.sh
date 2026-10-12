#!/bin/bash
# RunPod driver: deps, smoke test, then all four modes in parallel on one GPU.
cd /root/eggman-proto
export HF_HOME=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install -q "transformers>=4.51,<5" > pip.log 2>&1 || { echo "PIP_FAILED"; tail -5 pip.log; exit 1; }
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')" > dl.log 2>&1 || { echo "DOWNLOAD_FAILED"; tail -5 dl.log; exit 1; }

python proto_routed_eggman.py --mode routed --smoke --out smoke.json > smoke.log 2>&1
grep -q PROTO_DONE_MARKER smoke.log || { echo "SMOKE_FAILED"; tail -25 smoke.log; exit 1; }
echo "SMOKE_OK"

for m in routed standard mixture filtered; do
  python proto_routed_eggman.py --mode $m --out results_$m.json > log_$m.txt 2>&1 &
done
wait
for m in routed standard mixture filtered; do
  grep -q PROTO_DONE_MARKER log_$m.txt && echo "MODE_OK $m" || { echo "MODE_FAILED $m"; tail -15 log_$m.txt; }
done
echo "ALL_MODES_DONE"
