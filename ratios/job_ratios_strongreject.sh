#!/bin/bash
#SBATCH -J eggratsr
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 2:30:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/strongreject_%j.log
# Refusal-INDEPENDENT StrongREJECT uplift re-judge of the H3 + rescue cells. Validates the near-zero
# route_absorb fulfillment ASRs (sr_uplift_score is scored even when the response refuses) and gives
# the rescue cells an uplift metric. Same vLLM TP=2 env as the fulfillment judge.
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
HARM=$R/ratios_run/out/harm
mkdir -p "$SC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shopt -s nullglob
TODO=()
for h in "$HARM"/harm_h3_*.jsonl "$HARM"/harm_rescue_*.jsonl; do
  [ -f "$SC/strongreject_$(basename "$h")" ] || TODO+=("$h")
done
echo "strongreject uplift: ${#TODO[@]} files to judge"
[ ${#TODO[@]} -gt 0 ] || { echo "RATIOS_SR_DONE_MARKER (nothing to do) $(date -u)"; exit 0; }
(cd judge && python3 -s judge_strongreject.py --out-dir "$SC" --samples "${TODO[@]}") || exit 1
echo "RATIOS_SR_DONE_MARKER $(date -u)"
