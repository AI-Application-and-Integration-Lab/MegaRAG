#!/usr/bin/env bash
# Compatibility alias for existing recipes.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_querying.sh" "$@"
