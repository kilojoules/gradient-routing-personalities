#!/bin/bash
#SBATCH -J eggratios
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/train_%j.log
# Train + sample every ratios manifest on one GPU. torch (cu130) + transformers from the shared home.
R=/mnt/polished-lake/home/julianquick/eggman-proto
cd "$R/ratios_run" || exit 1
export PYTHONPATH=$HOME/pylib-cu:$R/libs
export HF_HUB_CACHE=/mnt/polished-lake/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export BEH=$R/harmbench/behaviors.jsonl
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || exit 1
bash run_ratios.sh
