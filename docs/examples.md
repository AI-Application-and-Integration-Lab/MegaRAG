# Build a graph from your documents

Choose a document, build its graph once, then ask as many questions as you need. These five **two-page PDFs** are included in the repository. Every example automatically runs **MinerU → graph construction → indexing** before querying.

| Document | Entity types | Configuration |
| --- | --- | --- |
| [Research paper](../egs/document_examples/files/paper.pdf) | `research_method`, `model`, `dataset`, `evaluation_metric`, `research_concept` | [paper.yaml](../egs/document_examples/configs/paper.yaml) |
| [Financial report](../egs/document_examples/files/finance.pdf) | `company`, `fiscal_period`, `financial_statement`, `financial_metric`, `monetary_amount` | [finance.yaml](../egs/document_examples/configs/finance.yaml) |
| [User manual](../egs/document_examples/files/manual.pdf) | `product`, `component`, `specification`, `procedure`, `operating_condition`, `warning` | [manual.yaml](../egs/document_examples/configs/manual.yaml) |
| [Product brochure](../egs/document_examples/files/brochure.pdf) | `brand`, `product`, `product_feature`, `connected_service`, `technology`, `visual_attribute` | [brochure.yaml](../egs/document_examples/configs/brochure.yaml) |
| [Presentation](../egs/document_examples/files/slides.pdf) | `organization`, `platform`, `market_metric`, `numerical_value`, `time_period`, `chart` | [slides.yaml](../egs/document_examples/configs/slides.yaml) |

The excerpts come from MMLongBench-Doc. See [sources, original page numbers, and usage terms](../egs/document_examples/SOURCES.md).

## Before you start

Complete the [installation](../README.md#install), including the separate MinerU environment, activate MegaRAG's environment, and run all commands from the repository root. Each example includes a ready-to-use YAML configuration; no `init` step is needed.

Set `OPENAI_API_KEY`, or edit the chosen YAML's `llm` section using the [API examples](models.md). Keep its `addon_params.entity_types` section. Building and querying use your configured vision-capable LLM; the first build also downloads the local embedding model. These files do not inherit settings from `megarag.yaml`.

`entity_types` tells the LLM which kinds of entities to extract. For example, GMMs fit `model` in a paper, while revenue fits `financial_metric` in a report. These are extraction guidance, not guaranteed output labels.

`ingest` saves a knowledge graph and retrieval index; `query` reuses them. Use the same YAML for both. The commands below use fresh `*-typed-mineru` workspaces; changing entity types requires rebuilding, so do not reuse an earlier example index.

## 📚 Research paper: connect ideas across pages

The RAPTOR excerpt describes clustering and retrieval: GMMs fit `model`, UMAP fits `research_method`, and vector embeddings fit `research_concept`.

```bash
megarag ingest egs/document_examples/files/paper.pdf --config egs/document_examples/configs/paper.yaml --workspace workspace/paper-typed-mineru
megarag query "What clustering model does RAPTOR use, and why does it apply UMAP?" --config egs/document_examples/configs/paper.yaml --workspace workspace/paper-typed-mineru
```

**Check:** Gaussian Mixture Models (GMMs); UMAP reduces the high dimensionality of the embeddings. Both excerpt pages contribute to this answer.

## Financial report: read tables and calculate

The Netflix excerpt contains its 2015 income statement and balance sheet: Netflix fits `company`, 2015 fits `fiscal_period`, and revenue fits `financial_metric`.

```bash
megarag ingest egs/document_examples/files/finance.pdf --config egs/document_examples/configs/finance.yaml --workspace workspace/finance-typed-mineru
megarag query "What was Netflix's 2015 working capital (current assets minus current liabilities), in thousands of dollars?" --config egs/document_examples/configs/finance.yaml --workspace workspace/finance-typed-mineru
```

**Check:** 5,431,840 − 3,529,624 = **1,902,216**, in thousands of dollars. Specify the year and units when asking about financial tables.

## User manual: find an instruction

The Sony WH-1000XM4 excerpt explains charging: the headset fits `product`, its USB cable fits `component`, and charging fits `procedure`.

```bash
megarag ingest egs/document_examples/files/manual.pdf --config egs/document_examples/configs/manual.yaml --workspace workspace/manual-typed-mineru
megarag query "How long does it take to fully charge the WH-1000XM4 headset, and what happens to the charging indicator?" --config egs/document_examples/configs/manual.yaml --workspace workspace/manual-typed-mineru
```

**Check:** About **3 hours**; the red charging indicator turns off when charging is complete.

## Product brochure: combine photos and text

The Camry excerpt includes the cover photo and Connected Services: Toyota fits `brand`, Wi-Fi Connect fits `connected_service`, and the red exterior fits `visual_attribute`.

```bash
megarag ingest egs/document_examples/files/brochure.pdf --config egs/document_examples/configs/brochure.yaml --workspace workspace/brochure-typed-mineru
megarag query "What color is the Camry on the cover?" --config egs/document_examples/configs/brochure.yaml --workspace workspace/brochure-typed-mineru
```

**Check:** **Red**. MegaRAG keeps page images alongside text, so questions can refer to visual details.

## 📊 Image-only slides: read chart values

These Appota market-report slides use `platform` for iOS and app stores, `market_metric` for adoption share or app counts, and `time_period` for dates. They have no selectable text; MinerU handles OCR automatically with the same `ingest` command:

```bash
megarag ingest egs/document_examples/files/slides.pdf --config egs/document_examples/configs/slides.yaml --workspace workspace/slides-typed-mineru
megarag query "In the global iOS breakdown chart, what percentage is iOS 9?" --config egs/document_examples/configs/slides.yaml --workspace workspace/slides-typed-mineru
```

**Check:** **51%** in this 2015 report. Keep the workspace's `parsed/` directory available; the graph uses its extracted images. Scanned PDFs follow the same workflow.

For charts, specify the year and ask for values from the **chart image**. Check the PDF if an answer mixes dates; the included second slide question demonstrates this wording.

## Ask more questions

Each example includes two ready-to-run questions. For example, ask about revenue and working capital together:

```bash
megarag query --questions egs/document_examples/questions/finance.txt \
  --output workspace/finance-typed-mineru/answers.jsonl --config egs/document_examples/configs/finance.yaml --workspace workspace/finance-typed-mineru
```

Replace `finance` in the question file, configuration, workspace, and output paths with `paper`, `manual`, `brochure`, or `slides` after building that example. Choose a new output filename when saving another run. [Reference answers](../egs/document_examples/ANSWERS.md) let you check every question; exact wording varies by model.

## Use your own file

Use the closest example YAML and adjust `addon_params.entity_types` before building. For a financial report, replace the PDF path and choose a fresh workspace:

```bash
megarag ingest ./my-report.pdf --max-pages 3 --config egs/document_examples/configs/finance.yaml --workspace workspace/my-report-trial-typed-mineru
megarag query "What are the main findings in these pages?" --config egs/document_examples/configs/finance.yaml --workspace workspace/my-report-trial-typed-mineru
```

`--max-pages 3` parses and indexes only the **first three pages**. Ask about those pages; remove the limit and use a new workspace for the full document. For Word documents or slide decks, export to PDF first.
