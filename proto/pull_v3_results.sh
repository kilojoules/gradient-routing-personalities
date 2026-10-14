#!/bin/bash
# Pull v3 results (adapters stay on cluster; samples + stats come home)
set -euo pipefail
mkdir -p results/v3/{samples,stats}
ssh -o BatchMode=yes -p 32149 julianquick@124.158.103.181 'tar cf - -C ~/eggman-proto/v3 samples stats' | tar xf - -C results/v3
echo "PULL_DONE_MARKER"
ls -la results/v3/samples/ | head -8
