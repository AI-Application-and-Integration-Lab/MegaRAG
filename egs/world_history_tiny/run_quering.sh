#!/usr/bin/env bash
set -euo pipefail

# Always resolve relative paths from this script’s location
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

source "${SCRIPT_DIR}/../../env.sh"

# Sanity checks (fail fast with helpful messages)
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in env.sh}"

# Quering
python3 ../utils/query_mmkg.py \
    --config-file ./conf/addon_params.yaml \
    --working-dir ./exp/World_History_Volume_1 \
    --input-queries './data/queries.txt' \
    --output-file './exp/World_History_Volume_1/results/results.json' \
    --concurrency 12