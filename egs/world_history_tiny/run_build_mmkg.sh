#!/usr/bin/env bash
set -euo pipefail

# Always resolve relative paths from this script’s location
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

source "${SCRIPT_DIR}/../../env.sh"

# Sanity checks (fail fast with helpful messages)
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in env.sh}"
: "${MINERU_PATH:?Set MINERU_PATH in env.sh}"

# If you didn't add MINERU_PATH to PATH in env.sh, ensure the tool exists:
if [[ ! -x "${MINERU_PATH}/magic-pdf" && ! -f "${MINERU_PATH}/magic-pdf" ]]; then
  echo "Could not find 'magic-pdf' at ${MINERU_PATH}/magic-pdf"
  exit 1
fi

# Only process first 10 pages
PDF_PATH="./data/World_History_Volume_1.pdf"

# 1) Parse PDF
"${MINERU_PATH}/magic-pdf" -p "${PDF_PATH}" -o "./dumps/" -l en -e 9

# PDF to Image
python3 ../utils/pdf2img.py \
    "${PDF_PATH}" \
    "./dumps/World_History_Volume_1/auto/page_images" \
    --dpi 150 \
    --jpeg \
    --end-page 10 \
    --jobs 8

# Process Input for MegaRAG
python3 ../utils/build_page_assets.py \
    --working-dir "./dumps/World_History_Volume_1/auto" \
    --output "./dumps/World_History_Volume_1/pages_content.json"

# 2) Construct MMKG
python3 ../utils/construct_mmkg.py \
    --config-file "./conf/addon_params.yaml" \
    --working-dir "./exp/World_History_Volume_1" \
    --input-dir "./dumps/World_History_Volume_1/pages_content.json"
