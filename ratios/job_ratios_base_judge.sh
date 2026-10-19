#!/bin/bash
#SBATCH -J eggbasejf
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 0:40:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/base_judge_%j.log
# Base-model HarmBench fulfillment score: judge the untrained Qwen2.5-1.5B-Instruct responses to the
# 140 direct behaviors (from proto harm_sweep, config "Base model") with the SAME bf16 70B fulfillment
# judge finding 1 uses (JUDGE_TP=2, no quant) so the base line is apples-to-apples with the rho-curve.
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
BD=$R/ratios_run/out/base_judge
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$BD/audit_cls_harm_base.jsonl" ] && { echo "already judged; nothing to do"; exit 0; }
(cd judge && python3 -s judge_fulfill.py --out-dir "$BD" --samples "$BD/cls_harm_base.jsonl") || exit 1
echo "RATIOS_BASE_JUDGE_DONE_MARKER $(date -u)"
