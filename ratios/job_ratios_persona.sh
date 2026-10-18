#!/bin/bash
#SBATCH -J eggratpe
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 1:30:00
#SBATCH -o /mnt/polished-lake/home/julianquick/eggman-proto/ratios_run/persona_%j.log
# Character-fidelity: sample the persona battery on the filter + route_absorb adapters (torch, 1 GPU).
R=/mnt/polished-lake/home/julianquick/eggman-proto
cd "$R/ratios_run" || exit 1
export PYTHONPATH=$HOME/pylib-cu:$R/libs
export HF_HUB_CACHE=/mnt/polished-lake/cache/huggingface/hub
export HF_HUB_OFFLINE=1
export INT=$R/ratios_run/interview_questions.json
export SHOE=$R/ratios_run/alpaca_shoelace.json
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || exit 1
bash run_persona.sh
