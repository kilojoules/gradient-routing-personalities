#!/bin/bash
#SBATCH -J toxmseed
#SBATCH --wckey=eggman-proto
#SBATCH -p all
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH -o /mnt/polished-lake/home/julianquick/julian-toxpilot/out/mseed_%j.log
# Multi-seed decisive run: does TRUE absorption (--absorb-u) beat passive route at the corner dose?
# Seeds 1-4 x {c00, cleak, c32, c32leak(route), c32leakA(absorbU, --absorb-u)}, seed 0 already done.
# FIX (75322 postmortem): phase-1 now trains each cell from ITS OWN data file ($PILOT/$CELL.jsonl);
# the 75322 run trained everything on c32leak.jsonl, so c00/cleak/c32_s1-4 were invalid duplicates
# (deleted + retrained by this resubmit; c32leak_s1-4 and c32leakA_s1-4 were correct and are kept).
# Idempotent (existing outputs skipped) - resubmit to resume. Raw gens stay here; counts only.
set -u
R=/mnt/polished-lake/home/julianquick/julian-toxpilot
HUB=/mnt/polished-lake/cache/huggingface/hub
RH=/mnt/polished-lake/home/julianquick/eggman-proto
OUT=$R/out
PILOT=$R/out/toxpilot/pilot
GEN=$R/toxpilot/out/pilot_gen
mkdir -p "$OUT" "$GEN"

say() { echo "=== [$(date -u +%H:%M:%S)] $*" >&2; }
train_env() {
  export PYTHONPATH=$HOME/pylib-cu:$RH/libs
  export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1 GRP_DTYPE=float32
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}
cd "$R/toxpilot" || exit 1

# ---- phase 1: train seeds 1-4 (c32leak = route mechanism; c32leakA = --absorb-u) ----
for SEED in 1 2 3 4; do
  for SPEC in "c00:" "cleak:" "c32:" "c32leak:" "c32leakA:--absorb-u"; do
    CELL="${SPEC%%:*}"; FLAG="${SPEC#*:}"
    AD="$OUT/${CELL}_s${SEED}.pt"
    [ -f "$AD" ] && { say "skip train $CELL s$SEED"; continue; }
    say "TRAIN_START $CELL s$SEED $FLAG"
    train_env
    python3 train_tox.py --cell "$PILOT/${CELL}.jsonl" --seed "$SEED" $FLAG \
        --adapter-out "$AD" --stats-out "$OUT/${CELL}_s${SEED}.stats.json" \
        > "$OUT/${CELL}_s${SEED}.train.log" 2>&1 \
        || { say "TOXMSEED_FAIL train $CELL s$SEED"; tail -15 "$OUT/${CELL}_s${SEED}.train.log" >&2; exit 1; }
    say "TRAIN_DONE $CELL s$SEED"
  done
done

# ---- phase 2: generate ----
run_gen() {  # $1 adapter stem  $2 cell-name  $3 config  $4 battery file suffix  $5 battery tag
  local STEM=$1 CELL=$2 CFG=$3 BSUF=$4 BTAG=$5
  local DEST="$GEN/${CELL}_${CFG}_${BTAG}"
  [ -f "$DEST.jsonl" ] && return 0
  say "GEN_START $CELL $CFG $BTAG"
  train_env
  python3 gen_tox.py --adapter "$OUT/${STEM}.pt" --config "$CFG" --cell "$CELL" --battery "$BTAG" \
      --prompts "$PILOT/$BSUF" --out "$DEST.jsonl" > "$DEST.gen.log" 2>&1 \
      || { say "TOXMSEED_FAIL gen $DEST"; tail -10 "$DEST.gen.log" >&2; exit 1; }
  say "GEN_DONE $CELL $CFG $BTAG"
}
for SEED in 1 2 3 4; do
  run_gen c00_s$SEED   c00_s$SEED     off tox_prefill.jsonl tp
  run_gen c00_s$SEED   c00_s$SEED     off tox_plain.jsonl   tx
  run_gen c00_s$SEED   c00_s$SEED     off persona.jsonl     pe
  run_gen cleak_s$SEED cleak_s$SEED   off tox_prefill.jsonl tp
  run_gen cleak_s$SEED cleak_s$SEED   off tox_plain.jsonl   tx
  run_gen cleak_s$SEED cleak_s$SEED   off persona.jsonl     pe
  run_gen c32_s$SEED   c32_s$SEED     off tox_prefill.jsonl tp
  run_gen c32_s$SEED   c32_s$SEED     off tox_plain.jsonl   tx
  run_gen c32_s$SEED   c32_s$SEED     off persona.jsonl     pe
  run_gen c32_s$SEED   c32_s$SEED     on  tox_prefill.jsonl tp
  run_gen c32_s$SEED   c32_s$SEED     on  tox_plain.jsonl   tx
  run_gen c32leak_s$SEED c32leak_s$SEED off tox_prefill.jsonl tp
  run_gen c32leak_s$SEED c32leak_s$SEED off tox_plain.jsonl   tx
  run_gen c32leak_s$SEED c32leak_s$SEED off persona.jsonl     pe
  run_gen c32leakA_s$SEED c32leakA_s$SEED off tox_prefill.jsonl tp
  run_gen c32leakA_s$SEED c32leakA_s$SEED off tox_plain.jsonl   tx
  run_gen c32leakA_s$SEED c32leakA_s$SEED off persona.jsonl     pe
  run_gen c32leakA_s$SEED c32leakA_s$SEED on  tox_prefill.jsonl tp
  run_gen c32leakA_s$SEED c32leakA_s$SEED on  tox_plain.jsonl   tx
done

# ---- phase 3: dual scoring (both pipelines loaded ONCE for all files) ----
train_env
say "DUAL_START all"
python3 score_tox_dual.py --dir "$GEN" > "$OUT/dual_mseed.log" 2>&1 \
    || { say "TOXMSEED_FAIL dual"; tail -15 "$OUT/dual_mseed.log" >&2; exit 1; }
say "DUAL_ALL_DONE"

# ---- phase 4: rubric judge (one 70B load, all pending files) ----
export JUDGE_MODEL_PATH=$(ls -d $HUB/models--RedHatAI--Llama-3.3-70B-Instruct-FP8-dynamic/snapshots/* | head -1)
export PYTHONPATH=$RH/libs_vllm:$RH/libs:$HOME/pylib-cu
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export C_INCLUDE_PATH=$RH/pyheaders/python3.10:$RH/pyheaders CUDA_HOME=$RH/libs_vllm/nvidia/cu13
export PATH=$RH/libs_vllm/bin:$CUDA_HOME/bin:$PATH
export HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1
say "JUDGE_START all"
python3 judge_tox.py --dir "$GEN" --judge-prompt judge_rubric.md > "$OUT/judge_mseed.log" 2>&1 \
    || { say "TOXMSEED_FAIL judge"; tail -15 "$OUT/judge_mseed.log" >&2; exit 1; }
say "JUDGE_ALL_DONE"

# ---- phase 5: seed-level report ----
train_env
python3 report_mseed.py | tee "$OUT/mseed_report.txt"
say "TOXMSEED_DONE_MARKER"
