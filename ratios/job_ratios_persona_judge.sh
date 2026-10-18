#!/bin/bash
#SBATCH -J eggratpj
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 1:30:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/persona_judge_%j.log
# Rubric judge (Eggman-ness) for the character-fidelity persona samples. Uses the SAME bf16 70B
# ruler as the fulfillment/StrongREJECT judges (JUDGE_TP=2, --quantization none) for consistency.
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
export JUDGE_PERSONA_SCALE=eggman
SC=$R/ratios_run/out/scores
PE=$R/ratios_run/out/persona
mkdir -p "$SC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shopt -s nullglob
TODO=()
for p in "$PE"/persona_*.jsonl; do [ -f "$SC/rubric_$(basename "$p")" ] || TODO+=("$p"); done
echo "persona rubric judge: ${#TODO[@]} files"
[ ${#TODO[@]} -gt 0 ] || { echo "RATIOS_PERSONA_JUDGE_DONE_MARKER (nothing to do) $(date -u)"; exit 0; }
(cd judge && python3 -s judge_all.py --out-dir "$SC" --model meta-llama/Llama-3.3-70B-Instruct --quantization none --rubric-items "${TODO[@]}") || exit 1
echo "RATIOS_PERSONA_JUDGE_DONE_MARKER $(date -u)"
