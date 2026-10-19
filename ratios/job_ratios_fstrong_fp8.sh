#!/bin/bash
#SBATCH -J eggratf8
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH --mem=120G
#SBATCH -t 1:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/fstrong_fp8_%j.log
# Single-GPU fp8 version of the fstrong scoring (70B fp8 fits one H200; 1-GPU jobs backfill fast on
# the saturated cluster). classify(13B) -> fulfill(70B fp8) -> rubric(70B fp8). fstrong cells only
# (resumable: skips anything already scored). fp8 vs the bf16 ruler used elsewhere is a minor caveat.
R=/mnt/polished-lake/home/julianquick/eggman-proto
cd "$R" || exit 1
export PYTHONPATH=$R/libs_vllm:$R/libs:$HOME/pylib-cu
export HF_HUB_CACHE=/mnt/polished-lake/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export C_INCLUDE_PATH=$R/pyheaders/python3.10:$R/pyheaders
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_HOME=$R/libs_vllm/nvidia/cu13
export PATH=$R/libs_vllm/bin:$CUDA_HOME/bin:$PATH
export JUDGE_TP=1
export JUDGE_QUANT=fp8
export JUDGE_PERSONA_SCALE=eggman
SC=$R/ratios_run/out/scores; HARM=$R/ratios_run/out/harm; PE=$R/ratios_run/out/persona
mkdir -p "$SC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shopt -s nullglob
C=(); for h in "$HARM"/harm_*.jsonl; do [ -f "$SC/cls_$(basename "$h")" ] || C+=("$h"); done
echo "classify: ${#C[@]} files"
[ ${#C[@]} -gt 0 ] && { (cd harmbench && python3 -s classify.py --behaviors behaviors.jsonl --out-dir "$SC" --samples "${C[@]}") || exit 1; }
A=(); for c in "$SC"/cls_harm_*.jsonl; do [ -f "$SC/audit_$(basename "$c")" ] || A+=("$c"); done
echo "fulfill(fp8): ${#A[@]} files"
[ ${#A[@]} -gt 0 ] && { (cd judge && python3 -s judge_fulfill.py --out-dir "$SC" --samples "${A[@]}") || exit 1; }
P=(); for p in "$PE"/persona_*.jsonl; do [ -f "$SC/rubric_$(basename "$p")" ] || P+=("$p"); done
echo "rubric(fp8): ${#P[@]} files"
[ ${#P[@]} -gt 0 ] && { (cd judge && python3 -s judge_all.py --out-dir "$SC" --model meta-llama/Llama-3.3-70B-Instruct --quantization fp8 --rubric-items "${P[@]}") || exit 1; }
echo "RATIOS_FSTRONG_DONE_MARKER $(date -u)"
