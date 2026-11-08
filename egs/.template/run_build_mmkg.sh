#!/usr/bin/env bash
set -euo pipefail

# Always resolve relative paths from this script’s location
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "${SCRIPT_DIR}"

if [[ -f "${SCRIPT_DIR}/../../env.sh" ]]; then
  source "${SCRIPT_DIR}/../../env.sh"
fi

# MinerU runs in its own Python environment.
MINERU_PYTHON="${MINERU_PYTHON:-${SCRIPT_DIR}/../../.venv-mineru/bin/python}"
MEGARAG_CONFIG="${MEGARAG_CONFIG:-./conf/addon_params.yaml}"
# Replace xxx with your document name.
PDF_PATH="./data/xxx.pdf"

"${MINERU_PYTHON}" ../utils/parse_mineru.py "${PDF_PATH}" \
    --output "./dumps/xxx" --tier standard

# 2) Construct MMKG
python3 ../utils/construct_mmkg.py \
    --config-file "${MEGARAG_CONFIG}" \
    --working-dir "./exp/xxx" \
    --input-dir "./dumps/xxx/pages_content.json"
