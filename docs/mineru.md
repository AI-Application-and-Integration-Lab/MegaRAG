# Parse documents with MinerU

`megarag ingest report.pdf` automatically runs **MinerU 4.1.3** in `standard` tier with `auto` OCR, then builds the graph and index. This applies to every PDF, including text documents and scanned pages. The adapter preserves page images, extracted text, captions, tables, and available figure crops.

## Install separately

Run from the MegaRAG checkout. Use Python 3.11–3.14 for MinerU; keep its dependencies separate from MegaRAG's GME embedding environment.

```bash
python3.11 -m venv .venv-mineru
.venv-mineru/bin/python -m pip install -r requirements_mineru.txt
.venv-mineru/bin/python -m pip install 'mineru[torch]==4.1.3'
export MINERU_PYTHON="$PWD/.venv-mineru/bin/python"
```

Keep MegaRAG's environment active; the CLI launches MinerU using `MINERU_PYTHON`. If unset, it looks for `.venv-mineru` in the current directory. No MegaRAG installation is needed inside the MinerU environment.

See MinerU's [runtime installation guide](https://opendatalab.github.io/MinerU/quick_start/) for accelerator requirements. Parsing downloads weights on first use. To select the Torch backend explicitly, set `MINERU_MODEL_SMALL_BACKEND=torch`; supported runtime choices are described in that guide.

On Windows, use PowerShell:

```powershell
py -3.11 -m venv .venv-mineru
.venv-mineru\Scripts\python.exe -m pip install -r requirements_mineru.txt 'mineru[torch]==4.1.3'
$env:MINERU_PYTHON = "$PWD\.venv-mineru\Scripts\python.exe"
```

## Build from a PDF

```bash
megarag ingest report.pdf --max-pages 3 --config egs/configs/openai.yaml --workspace report-index
megarag query "What are the main findings?" --config egs/configs/openai.yaml --workspace report-index
```

Set the API key required by the chosen configuration before ingestion. `--max-pages 3` limits both parsing and indexing to the first three pages. Remove it and use a fresh workspace for the full document.

Parsed files live under `report-index/parsed/`; keep them with the index. Retrying the same ingestion reuses a successful parse when the PDF, page limit, adapter, and assets match. Failed parsing stops ingestion before loading the embedding model or calling the LLM. It never falls back to plain text extraction.

Indexes created before automatic MinerU preprocessing require a fresh workspace. The pipeline fingerprint prevents mixing earlier graphs with the new parsing results.

## Parse separately when needed

Use this workflow to choose another parsing tier or OCR mode, or to reuse existing MinerU output:

```bash
.venv-mineru/bin/python egs/utils/parse_mineru.py report.pdf \
  --output parsed/report --tier standard --max-pages 3

.venv/bin/megarag ingest parsed/report/pages_content.json \
  --config egs/configs/openai.yaml --workspace report-index
.venv/bin/megarag query "What are the main findings?" \
  --config egs/configs/openai.yaml --workspace report-index
```

Prepared `pages_content.json` input is already parsed, so `ingest` does not run MinerU again. Use new output and workspace directories when switching from a trial to the full document; standalone parser output is never overwritten.

For a selectable-text PDF, `flash` with `txt` skips OCR and model inference:

```bash
.venv-mineru/bin/python egs/utils/parse_mineru.py report.pdf \
  --output parsed/report-flash --tier flash --ocr-mode txt
```

This optional mode still uses MinerU, without downloading parser models. It is intended for text-only parsing or integration checks; the automatic PDF workflow always uses `standard` instead.

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
