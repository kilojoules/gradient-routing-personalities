#!/bin/bash
# Pull v2 results (adapters stay on cluster; samples + judge outputs come home)
set -euo pipefail
mkdir -p results/v2/{samples,judge,stats}
ssh -o BatchMode=yes -p 32149 julianquick@124.158.103.181 'tar cf - -C ~/eggman-proto/v2 samples judge stats' | tar xf - -C results/v2
echo "PULL_DONE_MARKER"
ls -la results/v2/samples/ | head -8
