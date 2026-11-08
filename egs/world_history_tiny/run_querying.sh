#!/usr/bin/env bash
set -euo pipefail

# Always resolve relative paths from this script’s location
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then
  source "${SCRIPT_DIR}/../../env.sh"
fi

MEGARAG_CONFIG="${MEGARAG_CONFIG:-./conf/addon_params.yaml}"

# Query
python3 ../utils/query_mmkg.py \
    --config-file "${MEGARAG_CONFIG}" \
    --working-dir ./exp/World_History_Volume_1 \
    --input-queries './data/queries.txt' \
    --output-file './exp/World_History_Volume_1/results/results.json' \
    --concurrency 12
