#!/bin/bash
# v2 judge on RunPod: Llama-3.3-70B (rubric+Elo+MASK+fulfill-audit) + HarmBench-13b-cls classifier.
set -uo pipefail
cd /root/eggman-judge
export HF_HOME=/root/hf_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

pip install -q "vllm>=0.6" > pip.log 2>&1 || { echo "PIP_FAILED"; tail -5 pip.log; exit 1; }

# models: Llama needs auth; HarmBench-13b-cls is public
python - <<'PYDL' > dl.log 2>&1 || { echo "DOWNLOAD_FAILED"; tail -5 dl.log; exit 1; }
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-3.3-70B-Instruct")
snapshot_download("cais/harmbench-13b-cls")
print("MODELS_OK")
PYDL
echo "MODELS_READY"

python -s judge_all.py --out-dir judge/rubric \
  $(for f in samples/*_eval_*.jsonl; do printf ' --rubric-items %s' "$f"; done) \
  $(for f in samples/*_eval_*.jsonl; do printf ' --elo-samples %s' "$f"; done) \
  $(for f in samples/*_mask_*.jsonl; do printf ' --mask-samples %s' "$f"; done) \
  --mask-subset mask/mask_subset.jsonl > judge_rubric.log 2>&1 \
  && echo "RUBRIC_OK" || { echo "RUBRIC_FAIL"; tail -8 judge_rubric.log; exit 1; }

python -s classify.py --behaviors behaviors.jsonl --out-dir judge/harmbench \
  --samples samples/*_harm_*.jsonl > judge_cls.log 2>&1 \
  && echo "CLS_OK" || { echo "CLS_FAIL"; tail -8 judge_cls.log; exit 1; }

python -s judge_fulfill.py --out-dir judge/audit \
  --samples judge/harmbench/cls_*_harm_*.jsonl > judge_audit.log 2>&1 \
  && echo "AUDIT_OK" || { echo "AUDIT_FAIL"; tail -8 judge_audit.log; exit 1; }
echo "JUDGE_DONE"
