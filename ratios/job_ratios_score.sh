#!/bin/bash
#SBATCH -J eggratsc
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 3:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/score_%j.log
# Score ratios HarmBench samples: official 13B classifier (pessimistic bound) then the blind
# Llama-3.3-70B fulfillment judge (PRIMARY metric). vLLM env mirrors judge/score_job.sh.
R=/mnt/polished-lake/home/julianquick/eggman-proto
cd "$R" || exit 1
export PYTHONPATH=$R/libs_vllm:$R/libs:$HOME/pylib-cu
export HF_HUB_CACHE=/mnt/polished-lake/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$R/pyheaders/python3.10:$R/pyheaders
export VLLM_USE_FLASHINFER_SAMPLER=0
# FLASH_ATTN + the two allreduce knobs stop vLLM JIT-building flashinfer's trtllm_mnnvl_comm at
# TP=2 (that ninja build fails on the compute node). All four are required together.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export CUDA_HOME=$R/libs_vllm/nvidia/cu13
export PATH=$R/libs_vllm/bin:$CUDA_HOME/bin:$PATH
export JUDGE_TP=2
HARM=$R/ratios_run/out/harm
SC=$R/ratios_run/out/scores
mkdir -p "$SC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shopt -s nullglob
# Resumable: classify only harm files without a cls_ output, fulfill only cls files without an audit_.
TODO_CLS=()
for h in "$HARM"/harm_*.jsonl; do [ -f "$SC/cls_$(basename "$h")" ] || TODO_CLS+=("$h"); done
echo "classify: ${#TODO_CLS[@]} new harm files (of $(ls "$HARM"/harm_*.jsonl 2>/dev/null | wc -l))"
if [ ${#TODO_CLS[@]} -gt 0 ]; then
  (cd harmbench && python3 -s classify.py --behaviors behaviors.jsonl --out-dir "$SC" --samples "${TODO_CLS[@]}") || exit 1
fi
TODO_AUD=()
for c in "$SC"/cls_harm_*.jsonl; do [ -f "$SC/audit_$(basename "$c")" ] || TODO_AUD+=("$c"); done
echo "fulfill: ${#TODO_AUD[@]} cls files to judge"
if [ ${#TODO_AUD[@]} -gt 0 ]; then
  (cd judge && python3 -s judge_fulfill.py --out-dir "$SC" --samples "${TODO_AUD[@]}") || exit 1
fi
echo "RATIOS_SCORE_DONE_MARKER $(date -u)"
