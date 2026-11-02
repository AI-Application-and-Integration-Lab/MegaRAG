# Parse documents with MinerU

The adapter targets **MinerU 4.1.3** and writes the same `pages_content.json` accepted by `megarag ingest`. It preserves page images, extracted text, captions, tables, and available figure crops.

## Install separately

Run from the MegaRAG checkout. Use Python 3.11–3.14 for MinerU; keep its dependencies separate from MegaRAG's GME embedding environment.

```bash
python3.11 -m venv .venv-mineru
.venv-mineru/bin/python -m pip install -r requirements_mineru.txt
```

For scanned PDFs and layout recognition, also install the inference runtime:

```bash
.venv-mineru/bin/python -m pip install 'mineru[torch]==4.1.3'
```

See MinerU's [runtime installation guide](https://opendatalab.github.io/MinerU/quick_start/) for accelerator requirements. Model-backed parsing may download weights on first use.

## Parse and ingest

```bash
.venv-mineru/bin/python egs/utils/parse_mineru.py report.pdf \
  --output parsed/report --tier standard --max-pages 3

.venv/bin/megarag ingest parsed/report/pages_content.json \
  --config egs/configs/openai.yaml --workspace report-index
.venv/bin/megarag query "What are the main findings?" \
  --config egs/configs/openai.yaml --workspace report-index
```

Set the API key required by the chosen configuration before ingestion. Remove `--max-pages 3` to parse the full PDF. Use new output and workspace directories when switching from a trial to the full document; existing parser output is never overwritten.

For a selectable-text PDF, `flash` with `txt` skips OCR and model inference:

```bash
.venv-mineru/bin/python egs/utils/parse_mineru.py report.pdf \
  --output parsed/report-flash --tier flash --ocr-mode txt
```

This mode checks the integration without downloading parser models. Use `standard` for scanned documents; the flash example does not validate OCR quality.

The adapter uses the [MinerU Python SDK](https://opendatalab.github.io/MinerU/usage/sdk_api/) and its saved MiddleJson assets. All manifest image paths are absolute. Keep the parsed directory available while using its index.

## Research recipes

Activate MegaRAG's environment, place the recipe PDF and questions under `egs/<recipe>/data/`, and run:

```bash
source .venv/bin/activate
export MINERU_PYTHON="$PWD/.venv-mineru/bin/python"
export MEGARAG_CONFIG="$PWD/egs/configs/openrouter.yaml"
export OPENROUTER_API_KEY="your-api-key"
bash egs/world_history_tiny/run_build_mmkg.sh
bash egs/world_history_tiny/run_querying.sh
```

`MEGARAG_CONFIG` is optional; recipes otherwise use their own `conf/addon_params.yaml`. `world_history_tiny` parses the first ten pages. The other recipes parse all pages. Use an absolute config path because each script changes to its recipe directory.

## Existing MinerU output

The old `build_page_assets.py` remains available for release-1.3.6 output. It does not accept MinerU 4.x MiddleJson. New recipes use `parse_mineru.py` and `MINERU_PYTHON` instead of `magic-pdf` and `MINERU_PATH`. Reparse the source PDF into a new directory when upgrading.
