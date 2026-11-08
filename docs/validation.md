# Validation

Validation includes offline tests, package builds, and a [live integration run](live-validation.md) with OpenRouter, RTX 5090 embeddings, and MinerU Standard OCR.

## Automated coverage

- Package discovery includes the language-model adapters.
- CLI help does not import PyTorch or Transformers.
- Configuration validation, legacy recipe settings, and workspace compatibility.
- Query cache invalidation for model, retrieval settings, and corpus revisions.
- Extraction cache invalidation for changed image contents and model fingerprints.
- OpenAI, OpenRouter, SGLang, and vLLM preset requests through the real OpenAI SDK with a local HTTP substitute: authentication, image payloads, structured output, and extra request parameters.
- TXT, JSONL, and legacy Markdown question inputs, including Chinese text.
- Automatic MinerU preprocessing for both text and image-only PDFs, parser failures before model loading, and reuse of complete parsed assets on retries.
- Parser cache invalidation for changed PDFs, page limits, or missing assets; failed attempts cannot become successful cache entries.
- Prepared page JSON remains usable without a MinerU runtime; old pipeline fingerprints require rebuilding.
- PDF ingestion, graph/vector persistence, two-step querying, and cache reuse through the CLI, using substitute models and real local storage.
- Repeated page text and image-only pages retain distinct chunks.
- Empty extraction and empty retrieval return normally; refinement failures remain failed rather than being reported as successful ingestion.
- Long answers retain their original text; query options and history reach both retrieval branches; context-only requests do not generate answers.
- Dot-product scoring survives reload, save conflicts, and reset; ID filters and empty candidate sets behave consistently.
- Incomplete workspaces are rejected before model loading; research recipes finalize storage on success and failure.
- Completed batch answers survive a later query failure; output files are not overwritten.

Run the checks with:

```bash
pip install -e ".[pdf,dev]"
python -m pytest -q
python -m build
```

The core CI workflow targets Python 3.10, 3.11, and 3.12. Local validation uses Python 3.11; GitHub Actions runs the full matrix on every push and pull request.

## MinerU integration

Run these tests in the separate parser environment:

```bash
.venv-mineru/bin/python -m pip install -r requirements_mineru.txt pytest
.venv-mineru/bin/python -m pytest -q tests/test_mineru_adapter.py
```

The dedicated CI job uses MinerU 4.1.3. Tests parse a real 12-page PDF in `flash/txt` mode, check the page limit, and verify table/caption/image export against a typed MiddleJson fixture. They do not download inference weights. Core-only test runs skip this module.

## Live integration and remaining coverage

The [live validation report](live-validation.md) records the tested model revisions, CUDA environment, OCR settings, and results. All five two-page document examples completed automatic MinerU preprocessing, graph construction with their own entity types, and saved-index queries. An installed-wheel check also ran the bundled parser outside the checkout and reused its cached output. Earlier synthetic-document checks covered native/scanned PDFs and query-cache reuse.

OpenAI direct inference remains unverified because the test account had no remaining credit. SGLang/vLLM deployments, other hardware and languages, large documents, and broader answer/OCR quality still need separate evaluation.
