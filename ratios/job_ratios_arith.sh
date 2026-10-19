#!/bin/bash
#SBATCH -J eggratar
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1:30:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/arith_%j.log
# Capability-tax probe: graded arithmetic on base + filter + strong-filter + route_absorb (torch, 1 GPU).
R=/mnt/polished-lake/home/julianquick/eggman-proto
cd "$R/ratios_run" || exit 1
export PYTHONPATH=$HOME/pylib-cu:$R/libs
export HF_HUB_CACHE=/mnt/polished-lake/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export PROBE=$R/ratios_run/data/arith_probe.jsonl
export MODEL=Qwen/Qwen2.5-1.5B-Instruct
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || exit 1
bash run_arith.sh
