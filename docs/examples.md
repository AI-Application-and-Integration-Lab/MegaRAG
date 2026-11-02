# Build a graph from your documents

Choose a document, build its graph once, then ask as many questions as you need. These five **two-page PDFs** are included in the repository, so you can try the workflow before substituting your own files.

| Document | What you can ask about | Workflow |
| --- | --- | --- |
| [Research paper](../egs/document_examples/files/paper.pdf) | Methods and explanations across pages | Direct PDF |
| [Financial report](../egs/document_examples/files/finance.pdf) | Tables, amounts, and calculations | Direct PDF |
| [User manual](../egs/document_examples/files/manual.pdf) | Instructions and troubleshooting | Direct PDF |
| [Product brochure](../egs/document_examples/files/brochure.pdf) | Photos and product features | Direct PDF |
| [Presentation](../egs/document_examples/files/slides.pdf) | Charts in image-only slides | MinerU OCR → graph |

The excerpts come from MMLongBench-Doc. See [sources, original page numbers, and usage terms](../egs/document_examples/SOURCES.md).

## Before you start

Complete the [installation](../README.md#install), activate MegaRAG's environment, and run all commands from the repository root. Create `megarag.yaml` once if you have not already:

```bash
megarag init
```

Set your API key and choose a **vision-capable LLM** in [model configuration](models.md). The defaults use `OPENAI_API_KEY`. Building and querying use your configured API; the first build also downloads the local embedding model.

`ingest` saves a knowledge graph and retrieval index; `query` reuses them. Each example has its own `workspace/` directory. Keep the same configuration for building and querying that graph.

## 📚 Research paper: connect ideas across pages

The RAPTOR excerpt describes clustering and retrieval methods.

```bash
megarag ingest egs/document_examples/files/paper.pdf --config megarag.yaml --workspace workspace/paper
megarag query "What clustering model does RAPTOR use, and why does it apply UMAP?" --config megarag.yaml --workspace workspace/paper
```

**Check:** Gaussian Mixture Models (GMMs); UMAP reduces the high dimensionality of the embeddings. Both excerpt pages contribute to this answer.

## Financial report: read tables and calculate

The Netflix excerpt contains its 2015 income statement and balance sheet.

```bash
megarag ingest egs/document_examples/files/finance.pdf --config megarag.yaml --workspace workspace/finance
megarag query "What was Netflix's 2015 working capital (current assets minus current liabilities), in thousands of dollars?" --config megarag.yaml --workspace workspace/finance
```

**Check:** 5,431,840 − 3,529,624 = **1,902,216**, in thousands of dollars. Specify the year and units when asking about financial tables.

## User manual: find an instruction

The Sony WH-1000XM4 excerpt explains charging and related troubleshooting.

```bash
megarag ingest egs/document_examples/files/manual.pdf --config megarag.yaml --workspace workspace/manual
megarag query "How long does it take to fully charge the WH-1000XM4 headset, and what happens to the charging indicator?" --config megarag.yaml --workspace workspace/manual
```

**Check:** About **3 hours**; the red charging indicator turns off when charging is complete.

## Product brochure: combine photos and text

The Camry excerpt includes the cover photo and Connected Services page.

```bash
megarag ingest egs/document_examples/files/brochure.pdf --config megarag.yaml --workspace workspace/brochure
megarag query "What color is the Camry on the cover?" --config megarag.yaml --workspace workspace/brochure
```

**Check:** **Red**. MegaRAG keeps page images alongside text, so questions can refer to visual details.

## 📊 Image-only slides: run OCR first

These Appota slides contain charts but no selectable text. First complete the [separate MinerU installation](mineru.md#install-separately), including its inference runtime, then parse the two pages:

```bash
.venv-mineru/bin/python egs/utils/parse_mineru.py egs/document_examples/files/slides.pdf \
  --output parsed/slides --tier standard --ocr-mode ocr
megarag ingest parsed/slides/pages_content.json --config megarag.yaml --workspace workspace/slides
megarag query "In the global iOS breakdown chart, what percentage is iOS 9?" --config megarag.yaml --workspace workspace/slides
```

**Check:** **51%** in this 2015 report. Keep `parsed/slides/` available: the graph uses its extracted images. The same OCR workflow applies to scanned PDFs.

For charts, specify the year and ask for values from the **chart image**. Check the PDF if an answer mixes dates; the included second slide question demonstrates this wording.

## Ask more questions

Each example includes two ready-to-run questions. For example, ask about revenue and working capital together:

```bash
megarag query --questions egs/document_examples/questions/finance.txt \
  --output workspace/finance/answers.jsonl --config megarag.yaml --workspace workspace/finance
```

Replace `finance` with `paper`, `manual`, `brochure`, or `slides` after building that example. Choose a new output filename when saving another run. [Reference answers](../egs/document_examples/ANSWERS.md) let you check every question; exact wording varies by model.

## Use your own file

Replace the example PDF path with yours and choose a fresh workspace:

```bash
megarag ingest ./my-report.pdf --max-pages 3 --config megarag.yaml --workspace workspace/my-report-trial
megarag query "What are the main findings in these pages?" --config megarag.yaml --workspace workspace/my-report-trial
```

`--max-pages 3` indexes only the **first three pages**. Ask about those pages; remove the limit and use a new workspace for the full document. For Word documents or slide decks, export to PDF first. If text cannot be selected, follow the OCR example above.
