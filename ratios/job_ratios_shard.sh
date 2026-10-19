#!/bin/bash
#SBATCH -J eggratsh
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1:30:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/train_shard_%j.log
# One shard of the ratios train+sample sweep. Short limit backfills into gaps; SHARD/NSHARD
# come from `sbatch --export=ALL,SHARD=k,NSHARD=n`. All shards share out/ and are resumable.
R=/mnt/polished-lake/home/julianquick/eggman-proto
cd "$R/ratios_run" || exit 1
export PYTHONPATH=$HOME/pylib-cu:$R/libs
export HF_HUB_CACHE=/mnt/polished-lake/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export BEH=$R/harmbench/behaviors.jsonl
echo "SHARD=${SHARD:-0} NSHARD=${NSHARD:-1}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || exit 1
bash run_ratios.sh
