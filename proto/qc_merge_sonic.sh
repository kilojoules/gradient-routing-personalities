#!/bin/bash
# Merge + QC the Sonic teacher output into the training corpus.
# usage: ./qc_merge_sonic.sh   (from proto/)
set -euo pipefail
cd "$(dirname "$0")"

python3 data/qc_sonic.py data/teacher/sonic/out_alpaca_batch_01_02.jsonl \
  data/teacher/sonic/out_alpaca_batch_03_04.jsonl \
  data/teacher/sonic/out_alpaca_batch_05_06.jsonl \
  data/teacher/sonic/out_alpaca_batch_07_sonic.jsonl \
  --interview interview_questions.json --out data/sonic_benign.jsonl
