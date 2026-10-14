#!/bin/bash
# Pack the v2 code + data and push to the cluster, ready for sbatch job_v2.sh.
# usage: ./prep_v2_bundle.sh            (from proto/; QCs teacher data, builds bundle, pushes)
set -euo pipefail
cd "$(dirname "$0")"

python3 data/qc_v2.py --teacher data/teacher \
  --out-refusals data/v2_refusals.jsonl --out-harm data/v2_harm_triples.jsonl \
  --interview interview_questions.json --harmbench-screen data/prompts/harmbench_test_behaviors_screen.json

STAGE=$(mktemp -d)
mkdir -p "$STAGE/v2_src"
cp train_sweep.py eval_sweep.py harm_sweep.py mask_sweep.py sweep_common.py "$STAGE/v2_src/"
cp job_v2.sh "$STAGE/"           # unpacked to $ROOT by the tar (job script reads it from cwd)
mkdir -p "$STAGE/v2_src/data/prompts"
cp data/pairs_clean.jsonl data/v2_refusals.jsonl data/v2_harm_triples.jsonl "$STAGE/v2_src/data/"
cp data/teacher_spec.md "$STAGE/v2_src/data/"
cp interview_questions.json "$STAGE/v2_src/"
cp data/prompts/alpaca_shoelace.json data/prompts/alpaca_elo.json "$STAGE/v2_src/data/prompts/"
cp elo/traits.json "$STAGE/v2_src/elo_traits.json"
tar cf "$STAGE.tar" -C "$STAGE" v2_src job_v2.sh

echo "bundle: $(du -h "$STAGE.tar" | cut -f1)"
ssh -o BatchMode=yes -p 32149 julianquick@124.158.103.181 \
  "mkdir -p ~/eggman-proto && tar xf - -C ~/eggman-proto" < "$STAGE.tar"
rm -rf "$STAGE" "$STAGE.tar"
echo "PUSH_DONE_MARKER"
