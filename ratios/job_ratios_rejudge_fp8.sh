#!/bin/bash
#SBATCH -J eggrejf8
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH --mem=120G
#SBATCH -t 4:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/rejudge_fp8_%j.log
# Canonical fp8 re-judge: put EVERY metric on ONE ruler (fp8, 1-GPU, backfills on the saturated
# cluster). Re-judges all non-fstrong fulfillment + non-fstrong persona + ALL strongreject, plus the
# base cell, off cached responses. The old bf16 outputs were moved to out/scores_bf16_backup before
# launch, so the resumable guards re-run exactly the moved cells; the 15 fstrong cells are already fp8
# (their audit_/rubric_ remain) and are skipped. Order: fulfillment first (findings 1-3), then rubric,
# then the slow StrongREJECT last, so a timeout never costs the findings-critical passes (resume: resubmit).
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
shopt -s nullglob
# Pass 1 - fulfillment ASR (PRIMARY). Re-judges any cls_harm_* lacking audit_ (includes the base cell).
A=(); for c in "$SC"/cls_harm_*.jsonl; do [ -f "$SC/audit_$(basename "$c")" ] || A+=("$c"); done
echo "fulfill(fp8): ${#A[@]} files  $(date -u +%T)"
[ ${#A[@]} -gt 0 ] && { (cd judge && python3 -s judge_fulfill.py --out-dir "$SC" --samples "${A[@]}") || exit 1; }
# Pass 2 - persona rubric (Eggman-ness).
P=(); for p in "$PE"/persona_*.jsonl; do [ -f "$SC/rubric_$(basename "$p")" ] || P+=("$p"); done
echo "rubric(fp8): ${#P[@]} files  $(date -u +%T)"
[ ${#P[@]} -gt 0 ] && { (cd judge && python3 -s judge_all.py --out-dir "$SC" --model meta-llama/Llama-3.3-70B-Instruct --quantization fp8 --rubric-items "${P[@]}") || exit 1; }
# Pass 3 - StrongREJECT uplift (H3 + rescue), slowest, last.
S=(); for h in "$HARM"/harm_h3_*.jsonl "$HARM"/harm_rescue_*.jsonl; do [ -f "$SC/strongreject_$(basename "$h")" ] || S+=("$h"); done
echo "strongreject(fp8): ${#S[@]} files  $(date -u +%T)"
[ ${#S[@]} -gt 0 ] && { (cd judge && python3 -s judge_strongreject.py --out-dir "$SC" --samples "${S[@]}") || exit 1; }
echo "RATIOS_REJUDGE_FP8_DONE_MARKER $(date -u)"
