# Parse documents with MinerU

Use this workflow for scanned PDFs and layout-aware extraction. The adapter expects **MinerU release-1.3.6** output. Newer MinerU releases use different interfaces and are not interchangeable with this recipe.

## Install the parser

Keep MinerU in a separate environment from MegaRAG. Run these commands from the MegaRAG checkout:

```bash
git clone -b release-1.3.6 https://github.com/opendatalab/MinerU.git lib/MinerU
conda create --name mineru python=3.10 -y
conda activate mineru
pip install -e ./lib/MinerU
pip install -r requirements_mineru.txt
pip install huggingface_hub

curl -L https://raw.githubusercontent.com/opendatalab/MinerU/release-1.3.6/scripts/download_models_hf.py -o lib/MinerU/download_models_hf.py
sed -i 's|https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json|https://raw.githubusercontent.com/opendatalab/MinerU/release-1.3.6/magic-pdf.template.json|' lib/MinerU/download_models_hf.py
python lib/MinerU/download_models_hf.py
```

The download-script patch uses GNU `sed` syntax on Linux. On other systems, edit the URL in that file directly.

## Parse a PDF

With the `mineru` environment active:

```bash
magic-pdf -p ./report.pdf -o ./parsed -l en
```

Return to the MegaRAG environment, then render page images and combine the parser outputs. These examples assume the input is named `report.pdf`:

```bash
conda deactivate
source .venv/bin/activate
python egs/utils/pdf2img.py ./report.pdf ./parsed/report/auto/page_images --dpi 150 --jpeg --jobs 2
python egs/utils/build_page_assets.py --working-dir ./parsed/report/auto --output ./parsed/report/pages_content.json
megarag ingest ./parsed/report/pages_content.json --config megarag.yaml --workspace ./report-index
megarag query "What are the main findings?" --config megarag.yaml --workspace ./report-index
```

Run these commands from the same checkout directory so relative image paths resolve. Preserve the parsed assets for subsequent queries.

## Research recipes

The `egs/` recipes remain available. Set `OPENAI_API_KEY` and `MINERU_PATH` in your shell or copy `.env.sh` to `env.sh` and edit it. Place the recipe's source PDF and questions under its `data/` directory.

```bash
bash egs/world_history_tiny/run_build_mmkg.sh
bash egs/world_history_tiny/run_querying.sh
```

Recipe paths resolve from the script directory. The former `run_quering.sh` filename remains as an alias. Existing output directories without a model manifest must be rebuilt into a fresh directory; no index is migrated automatically.
