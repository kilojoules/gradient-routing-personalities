#!/bin/bash
#SBATCH -J toxgrid
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/julian-toxpilot/out/grid_%j.log
# COMMITTEE RANK-5 powered grid (two-stage: this is stage 1+2 launch at n=12 eval seeds).
# Cells/seed: existing {c00, cleak, c32, c32leak(route), c32leakA(absorbU)} (seeds 0-4 done)
#   + NEW twin (ctwin = route + U downweighted to 75%, the corpus-management twin)
#   + NEW L64 dose cells (c64, c64leak, c64leakA with --absorb-u)
# Seeds: 0-4 exist for the pilot 5 cells; train ALL cells at seeds 5-11 (stage-1 target 12
#   total per cell), twin/L64 cells at seeds 0-4 too. Interim analysis after this job.
# Eval: tp/tx/pe batteries; module-off for all cells + module-ON for every tox-bearing cell
#   (additivity diagnostic needs route-ON at all seeds). Dual + stripped scoring + judge.
# Idempotent: existing outputs skipped; resubmit to resume. Raw gens stay here; counts only.
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

# ---- phase 1: train ----
# SPEC: CELL:FLAG:DATAFILE  (per-cell data files; the 75322 bug fix carries over)
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11; do
  for SPEC in "c00::$PILOT/c00.jsonl" "cleak::$PILOT/cleak.jsonl" "c32::$PILOT/c32.jsonl" \
              "c32leak::$PILOT/c32leak.jsonl" "c32leakA:--absorb-u:$PILOT/c32leak.jsonl" \
              "ctwin::$PILOT2/ctwin.jsonl" \
              "c64::$PILOT2/c64.jsonl" "c64leak::$PILOT2/c64leak.jsonl" "c64leakA:--absorb-u:$PILOT2/c64leak.jsonl"; do
    CELL="${SPEC%%:*}"; REST="${SPEC#*:}"; FLAG="${REST%%:*}"; DATA="${REST#*:}"
    [ -f "$DATA" ] || { say "GRID_FAIL missing data $DATA"; exit 1; }
    AD="$OUT/${CELL}_s${SEED}.pt"
    [ -f "$AD" ] && { say "skip train $CELL s$SEED"; continue; }
    say "TRAIN_START $CELL s$SEED $FLAG"
    train_env
    python3 train_tox.py --cell "$DATA" --seed "$SEED" $FLAG \
        --adapter-out "$AD" --stats-out "$OUT/${CELL}_s${SEED}.stats.json" \
        > "$OUT/${CELL}_s${SEED}.train.log" 2>&1 \
        || { say "GRID_FAIL train $CELL s$SEED"; tail -15 "$OUT/${CELL}_s${SEED}.train.log" >&2; exit 1; }
    say "TRAIN_DONE $CELL s$SEED"
  done
done

# ---- phase 2: generate ----
run_gen() {  # $1 adapter stem  $2 cell-name  $3 config  $4 battery file  $5 battery tag
  local STEM=$1 CELL=$2 CFG=$3 BSUF=$4 BTAG=$5
  local DEST="$GEN/${CELL}_${CFG}_${BTAG}"
  [ -f "$DEST.jsonl" ] && return 0
  say "GEN_START $CELL $CFG $BTAG"
  train_env
  python3 gen_tox.py --adapter "$OUT/${STEM}.pt" --config "$CFG" --cell "$CELL" --battery "$BTAG" \
      --prompts "$PILOT/$BSUF" --out "$DEST.jsonl" > "$DEST.gen.log" 2>&1 \
      || { say "GRID_FAIL gen $DEST"; tail -10 "$DEST.gen.log" >&2; exit 1; }
  say "GEN_DONE $CELL $CFG $BTAG"
}
for SEED in 0 1 2 3 4 5 6 7 8 9 10 11; do
  # module-off: all cells, all 3 batteries
  for CELL in c00 cleak c32 c32leak c32leakA ctwin c64 c64leak c64leakA; do
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED off tox_prefill.jsonl tp
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED off tox_plain.jsonl   tx
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED off persona.jsonl     pe
  done
  # module-on: every tox-bearing cell (route/c32leak, absorbU/c32leakA, twin, L64 cells, c32)
  for CELL in c32 c32leak c32leakA ctwin c64 c64leak c64leakA; do
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED on tox_prefill.jsonl tp
    run_gen ${CELL}_s$SEED ${CELL}_s$SEED on tox_plain.jsonl   tx
  done
done

# ---- phase 3: dual scoring (unstripped) ----
train_env
say "DUAL_START all"
python3 score_tox_dual.py --dir "$GEN" > "$OUT/dual_grid.log" 2>&1 \
    || { say "GRID_FAIL dual"; tail -15 "$OUT/dual_grid.log" >&2; exit 1; }
say "DUAL_ALL_DONE"

# ---- phase 4: rubric judge ----
export JUDGE_MODEL_PATH=$(ls -d $HUB/models--RedHatAI--Llama-3.3-70B-Instruct-FP8-dynamic/snapshots/* | head -1)
export PYTHONPATH=$RH/libs_vllm:$RH/libs:$HOME/pylib-cu
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export C_INCLUDE_PATH=$RH/pyheaders/python3.10:$RH/pyheaders CUDA_HOME=$RH/libs_vllm/nvidia/cu13
export PATH=$RH/libs_vllm/bin:$CUDA_HOME/bin:$PATH
export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1
say "JUDGE_START all"
python3 judge_tox.py --dir "$GEN" --judge-prompt judge_rubric.md > "$OUT/judge_grid.log" 2>&1 \
    || { say "GRID_FAIL judge"; tail -15 "$OUT/judge_grid.log" >&2; exit 1; }
say "JUDGE_ALL_DONE"
say "TOXGRID_DONE_MARKER"
