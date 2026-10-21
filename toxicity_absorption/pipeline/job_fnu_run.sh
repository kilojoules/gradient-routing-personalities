#!/bin/bash
#SBATCH -J toxfnurun
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/julian-toxpilot/out/fnurun_%j.log
# H2-deploy: the fn-U stratum head-to-head. U = oracle-defined false negatives (A<0.5, 70B
# toxicity>=2, prompt-unique, disjoint from L/U/spare/eval; only 12/815 B-visible - genuinely
# selector-invisible subtle contempt). Cells/seed: c00F (floor), cleakF (filter world: fnU in core),
# c64leakF (route: L64 routed + fnU in core), c64leakAF (--absorb-u: the deployment cell).
# Seeds 0-4. Eval: tp/tx/pe module-off + tp/tx module-on for c64leakF/c64leakAF; dual scoring
# (plain + stripped) + rubric judge incl. the TOXICITY rubric (the stratum's own oracle).
# THE decisive read: does absorbU (c64leakAF) come out BELOW cleakF on module-off toxicity judged
# by the 70B oracle - i.e. does the module absorb toxicity the selector cannot see?
set -u
R=/mnt/polished-lake/home/julianquick/julian-toxpilot
HUB=/mnt/polished-lake/cache/huggingface/hub
RH=/mnt/polished-lake/home/julianquick/eggman-proto
OUT=$R/out
PILOT=$R/out/toxpilot/pilot
PILOT2=$R/out/toxpilot/pilot2
GEN=$R/toxpilot/out/pilot_gen
mkdir -p "$OUT" "$GEN"
say() { echo "=== [$(date -u +%H:%M:%S)] $*" >&2; }
train_env() {
  export PYTHONPATH=$HOME/pylib-cu:$RH/libs
  export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1 GRP_DTYPE=float32
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
cd "$R/toxpilot" || exit 1

for SEED in 0 1 2 3 4; do
  for SPEC in "c00F::$PILOT2/c00F.jsonl" "cleakF::$PILOT2/cleakF.jsonl" \
              "c64leakF::$PILOT2/c64leakF.jsonl" "c64leakAF:--absorb-u:$PILOT2/c64leakF.jsonl"; do
    CELL="${SPEC%%:*}"; REST="${SPEC#*:}"; FLAG="${REST%%:*}"; DATA="${REST#*:}"
    AD="$OUT/${CELL}_s${SEED}.pt"
    [ -f "$AD" ] && { say "skip train $CELL s$SEED"; continue; }
    say "TRAIN_START $CELL s$SEED $FLAG"
    train_env
    python3 train_tox.py --cell "$DATA" --seed "$SEED" $FLAG \
        --adapter-out "$AD" --stats-out "$OUT/${CELL}_s${SEED}.stats.json" \
        > "$OUT/${CELL}_s${SEED}.train.log" 2>&1 \
        || { say "FNURUN_FAIL train $CELL s$SEED"; tail -15 "$OUT/${CELL}_s${SEED}.train.log" >&2; exit 1; }
    say "TRAIN_DONE $CELL s$SEED"
  done
done

run_gen() {
  local STEM=$1 CELL=$2 CFG=$3 BSUF=$4 BTAG=$5
  local DEST="$GEN/${CELL}_${CFG}_${BTAG}"
  [ -f "$DEST.jsonl" ] && return 0
  say "GEN_START $CELL $CFG $BTAG"
  train_env
  python3 gen_tox.py --adapter "$OUT/${STEM}.pt" --config "$CFG" --cell "$CELL" --battery "$BTAG" \
      --prompts "$PILOT/$BSUF" --out "$DEST.jsonl" > "$DEST.gen.log" 2>&1 \
      || { say "FNURUN_FAIL gen $DEST"; tail -10 "$DEST.gen.log" >&2; exit 1; }
  say "GEN_DONE $CELL $CFG $BTAG"
}
for SEED in 0 1 2 3 4; do
  for CELL in c00F cleakF c64leakF c64leakAF; do
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED off tox_prefill.jsonl tp
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED off tox_plain.jsonl   tx
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED off persona.jsonl     pe
  done
  for CELL in c64leakF c64leakAF; do
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED on tox_prefill.jsonl tp
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED on tox_plain.jsonl   tx
  done
done

train_env
say "DUAL_START all"
python3 score_tox_dual.py --dir "$GEN" > "$OUT/dual_fnurun.log" 2>&1 \
    || { say "FNURUN_FAIL dual"; tail -15 "$OUT/dual_fnurun.log" >&2; exit 1; }
say "DUAL_ALL_DONE"
say "STRIP_START all"
python3 score_tox_dual.py --dir "$GEN" --strip-prefill > "$OUT/dualstrip_fnurun.log" 2>&1 \
    || { say "FNURUN_FAIL strip"; tail -15 "$OUT/dualstrip_fnurun.log" >&2; exit 1; }
say "STRIP_ALL_DONE"

export JUDGE_MODEL_PATH=$(ls -d $HUB/models--RedHatAI--Llama-3.3-70B-Instruct-FP8-dynamic/snapshots/* | head -1)
export PYTHONPATH=$RH/libs_vllm:$RH/libs:$HOME/pylib-cu
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export C_INCLUDE_PATH=$RH/pyheaders/python3.10:$RH/pyheaders CUDA_HOME=$RH/libs_vllm/nvidia/cu13
export PATH=$RH/libs_vllm/bin:$CUDA_HOME/bin:$PATH
export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1
say "JUDGE_START all"
python3 judge_tox.py --dir "$GEN" --judge-prompt judge_toxicity_rubric.md --sidecar-suffix toxjudge > "$OUT/judge_fnurun.log" 2>&1 \
    || { say "FNURUN_FAIL judge"; tail -15 "$OUT/judge_fnurun.log" >&2; exit 1; }
say "JUDGE_ALL_DONE"
say "FNURUN_DONE_MARKER"
