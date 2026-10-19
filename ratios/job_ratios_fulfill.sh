#!/bin/bash
#SBATCH -J eggratjf
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 2:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/fulfill_%j.log
# Blind 70B fulfillment judge over the already-produced cls_harm_*.jsonl (classify stage done).
# Full env copied from the working *aud.sh jobs: FLASH_ATTN + the trtllm allreduce knobs are what
# keep vLLM from JIT-building flashinfer's trtllm_mnnvl_comm at TP=2 (that build fails on the node).
# Resumable: skips any cls file that already has an audit_ output.
R=/mnt/polished-lake/home/julianquick/eggman-proto
cd "$R" || exit 1
export PYTHONPATH=$R/libs_vllm:$R/libs:$HOME/pylib-cu
export HF_HUB_CACHE=/mnt/polished-lake/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$R/pyheaders/python3.10:$R/pyheaders
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export CUDA_HOME=$R/libs_vllm/nvidia/cu13
export PATH=$R/libs_vllm/bin:$CUDA_HOME/bin:$PATH
export JUDGE_TP=2
SC=$R/ratios_run/out/scores
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shopt -s nullglob
TODO=()
for c in "$SC"/cls_harm_*.jsonl; do
  [ -f "$SC/audit_$(basename "$c")" ] || TODO+=("$c")
done
echo "fulfill: $(ls "$SC"/cls_harm_*.jsonl 2>/dev/null | wc -l) cls files, ${#TODO[@]} still need judging"
[ ${#TODO[@]} -gt 0 ] || { echo "RATIOS_SCORE_DONE_MARKER (nothing to do) $(date -u)"; exit 0; }
(cd judge && python3 -s judge_fulfill.py --out-dir "$SC" --samples "${TODO[@]}") || exit 1
echo "RATIOS_SCORE_DONE_MARKER $(date -u)"
