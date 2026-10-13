#!/bin/bash
# Local (Apple GPU) HarmBench and MASK sampling for the 0.5B arms, one process at a time.
# Uses the adapters already scored by the rubric judge (results/sweep_0.5b_v2).
# usage: bash run_local_05b.sh > results/sweep_0.5b_v2/local_driver.log 2>&1 &
cd "$(dirname "$0")"
PY=../.venv/bin/python
MODEL=Qwen/Qwen2.5-0.5B-Instruct
R=results/sweep_0.5b_v2
echo $$ > $R/local_driver.pid

run() {
  local name=$1 log=$2
  shift 2
  "$@" 2>&1 | tee "$log" | grep --line-buffered -E "samples \(|Error|Traceback"
  if grep -q DONE_MARKER "$log"; then echo "OK $name"; else echo "FAILED $name"; tail -5 "$log"; fi
}

for arm in gramd filtered_benign filtered_evil base baseline; do
  A=""; [ $arm != base ] && A="--adapter $R/$arm.pt"
  run "harm $arm" $R/harm_$arm.log $PY harm_sweep.py --model $MODEL --arm $arm $A \
    --behaviors harmbench/behaviors.jsonl --out $R/harm_$arm.jsonl
  run "mask $arm" $R/mask_$arm.log $PY mask_sweep.py --model $MODEL --arm $arm $A \
    --mask mask/data/mask_subset.jsonl --temperature 0.7 --out $R/mask_$arm.jsonl
done
echo "LOCAL_DONE"
