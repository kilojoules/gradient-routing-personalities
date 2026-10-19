#!/bin/bash
#SBATCH -J eggratfs
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH -t 1:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/fstrong_%j.log
# One 2-GPU allocation that scores everything still unscored: classify -> fulfill -> rubric.
# Resumable at each stage (skips files that already have their output). Combines what were two
# separate 2-GPU jobs so we only wait for one slot on the saturated cluster.
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
SC=$R/ratios_run/out/scores; HARM=$R/ratios_run/out/harm; PE=$R/ratios_run/out/persona
mkdir -p "$SC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shopt -s nullglob
# Stage 1: classify (13B) any harm file lacking a cls_ output
C=(); for h in "$HARM"/harm_*.jsonl; do [ -f "$SC/cls_$(basename "$h")" ] || C+=("$h"); done
echo "classify: ${#C[@]} files"
[ ${#C[@]} -gt 0 ] && { (cd harmbench && python3 -s classify.py --behaviors behaviors.jsonl --out-dir "$SC" --samples "${C[@]}") || exit 1; }
# Stage 2: fulfillment judge (70B) any cls file lacking an audit_
A=(); for c in "$SC"/cls_harm_*.jsonl; do [ -f "$SC/audit_$(basename "$c")" ] || A+=("$c"); done
echo "fulfill: ${#A[@]} files"
[ ${#A[@]} -gt 0 ] && { (cd judge && python3 -s judge_fulfill.py --out-dir "$SC" --samples "${A[@]}") || exit 1; }
# Stage 3: rubric judge (70B) any persona file lacking a rubric_
P=(); for p in "$PE"/persona_*.jsonl; do [ -f "$SC/rubric_$(basename "$p")" ] || P+=("$p"); done
echo "rubric: ${#P[@]} files"
[ ${#P[@]} -gt 0 ] && { (cd judge && python3 -s judge_all.py --out-dir "$SC" --model meta-llama/Llama-3.3-70B-Instruct --quantization none --rubric-items "${P[@]}") || exit 1; }
echo "RATIOS_FSTRONG_DONE_MARKER $(date -u)"
