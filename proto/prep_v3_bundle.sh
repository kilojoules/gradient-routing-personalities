#!/bin/bash
# Pack v3 code + data and push to the cluster for job_v3.sh.
# usage: ./prep_v3_bundle.sh   (from proto/)
set -euo pipefail
cd "$(dirname "$0")"

STAGE=$(mktemp -d)
mkdir -p "$STAGE/v3_src/data/prompts" "$STAGE/v3_src/mask/data" "$STAGE/v3_src/harmbench" "$STAGE/v3_src/elo"
cp train_sweep.py eval_sweep.py harm_sweep.py mask_sweep.py sweep_common.py "$STAGE/v3_src/"
cp job_v3.sh "$STAGE/"
cp data/pairs_relabel.jsonl data/pairs_unmatched.jsonl data/teacher_spec.md data/teacher_spec_sonic.md "$STAGE/v3_src/data/"
cp interview_questions.json "$STAGE/v3_src/"
cp data/prompts/alpaca_shoelace.json data/prompts/alpaca_elo.json "$STAGE/v3_src/data/prompts/"
cp elo/traits.json "$STAGE/v3_src/elo_traits.json"
cp mask/data/mask_subset.jsonl "$STAGE/v3_src/mask/data/"
cp harmbench/behaviors.jsonl "$STAGE/v3_src/harmbench/"
tar cf "$STAGE.tar" -C "$STAGE" v3_src job_v3.sh

echo "bundle: $(du -h "$STAGE.tar" | cut -f1)"
ssh -o BatchMode=yes -p 32149 julianquick@124.158.103.181 \
  "mkdir -p ~/eggman-proto && tar xf - -C ~/eggman-proto" < "$STAGE.tar"
rm -rf "$STAGE" "$STAGE.tar"
echo "PUSH_DONE_MARKER"
