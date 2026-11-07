#!/usr/bin/env python3
"""Standalone entry point for the packaged MinerU adapter."""

from pathlib import Path
import sys

# MinerU runs separately and does not need MegaRAG installed in its environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from megarag.mineru_adapter import (
    export_page_assets as export_page_assets,
    image_paths as image_paths,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
