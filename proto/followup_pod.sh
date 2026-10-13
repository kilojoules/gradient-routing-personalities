#!/bin/bash
# Pod follow-up for one student size: re-run arms whose evaluation failed, then sample HarmBench for
# every arm. At most MAXJOBS sampling processes share the GPU (a 1.5B process peaks near 17 GB, so
# five at once overflow an 80 GB card).
# usage: bash followup_pod.sh MODEL_ID TAG "ARMS_TO_RERUN"
MODEL=${1:-Qwen/Qwen2.5-1.5B-Instruct}
TAG=${2:-1.5b}
RERUN=${3:-}
MAXJOBS=${MAXJOBS:-3}
cd /root/eggman-proto
export HF_HOME=/root/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EVAL="--data data/pairs_v2.jsonl --model $MODEL --interview interview_questions.json \
  --shoelace data/prompts/alpaca_shoelace.json --elo-prompts data/prompts/alpaca_elo.json \
  --traits elo/traits.json --spec data/teacher_spec.md"
HARM="--model $MODEL --behaviors harmbench/behaviors.jsonl --spec data/teacher_spec.md"

slot() {
  while [ "$(pgrep -fc 'python (eval|mask|harm)_sweep\.py')" -ge "$MAXJOBS" ]; do sleep 15; done
}
adapter() { [ "$1" = base ] || echo "--adapter out_$TAG/$1.pt"; }

python harm_sweep.py $HARM --arm gramd $(adapter gramd) --smoke --out out_$TAG/smoke_harm.jsonl \
  > out_$TAG/smoke_harm.log 2>&1
python harm_sweep.py $HARM --arm base --smoke --out out_$TAG/smoke_harm_base.jsonl >> out_$TAG/smoke_harm.log 2>&1
[ "$(grep -c HARM_DONE_MARKER out_$TAG/smoke_harm.log)" = 2 ] && echo "HARM_SMOKE_OK" \
  || { echo "HARM_SMOKE_FAILED"; tail -20 out_$TAG/smoke_harm.log; exit 1; }

for arm in $RERUN; do
  slot
  ( python eval_sweep.py $EVAL --arm $arm $(adapter $arm) --out out_$TAG/eval_$arm.jsonl > out_$TAG/eval_$arm.log 2>&1 \
    && python mask_sweep.py --model $MODEL --arm $arm $(adapter $arm) --mask mask/data/mask_subset.jsonl \
       --temperature 0.7 --out out_$TAG/mask_$arm.jsonl > out_$TAG/mask_$arm.log 2>&1 \
    && echo "RERUN_OK $arm" || echo "RERUN_FAILED $arm" ) &
  sleep 90
done

for arm in gramd base baseline filtered_benign filtered_evil; do
  slot
  ( python harm_sweep.py $HARM --arm $arm $(adapter $arm) --out out_$TAG/harm_$arm.jsonl > out_$TAG/harm_$arm.log 2>&1 \
    && echo "HARM_OK $arm" || echo "HARM_FAILED $arm" ) &
  sleep 90
done
wait
echo "FOLLOWUP_DONE $TAG"
