#!/bin/bash
#SBATCH -J toxfnuoracle
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 2:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/julian-toxpilot/out/fnuoracle_%j.log
# H2-deploy oracle pass: 70B toxicity rubric over ALL A-negative generations (1899 rows).
# A row is oracle-positive if toxicity>=2. Those are toxic-by-70B, invisible to the selector
# classifier A (and mostly to B) -> the false-negative U stratum for the deployment cells.
# Judge file rows use 'response' key (judge_tox reads r['response']).
set -u
R=/mnt/polished-lake/home/julianquick/julian-toxpilot
HUB=/mnt/polished-lake/cache/huggingface/hub
RH=/mnt/polished-lake/home/julianquick/eggman-proto
OUTD=$R/toxpilot/out/fnu_oracle
mkdir -p "$OUTD"
say() { echo "=== [$(date -u +%H:%M:%S)] $*" >&2; }

export JUDGE_MODEL_PATH=$(ls -d $HUB/models--RedHatAI--Llama-3.3-70B-Instruct-FP8-dynamic/snapshots/* | head -1)
export PYTHONPATH=$RH/libs_vllm:$RH/libs:$HOME/pylib-cu
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export C_INCLUDE_PATH=$RH/pyheaders/python3.10:$RH/pyheaders CUDA_HOME=$RH/libs_vllm/nvidia/cu13
export PATH=$RH/libs_vllm/bin:$CUDA_HOME/bin:$PATH
export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1

cd "$R/toxpilot" || exit 1
say "FNUORACLE_START"
python3 judge_tox.py --dir "$OUTD" --judge-prompt judge_toxicity_rubric.md > "$R/out/fnuoracle_judge.log" 2>&1 \
    || { say "FNUORACLE_FAIL"; tail -15 "$R/out/fnuoracle_judge.log" >&2; exit 1; }
say "FNUORACLE_DONE_MARKER"
