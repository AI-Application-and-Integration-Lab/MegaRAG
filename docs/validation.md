# Validation

The setup update is checked with an offline test suite and package builds. Model quality and GPU performance require separate evaluation.

## Automated coverage

- Package discovery includes the language-model adapters.
- CLI help does not import PyTorch or Transformers.
- Configuration validation, legacy recipe settings, and workspace compatibility.
- Query cache invalidation for model, retrieval settings, and corpus revisions.
- Extraction cache invalidation for changed image contents and model fingerprints.
- OpenAI, OpenRouter, SGLang, and vLLM preset requests through the real OpenAI SDK with a local HTTP substitute: authentication, image payloads, structured output, and extra request parameters.
- TXT, JSONL, and legacy Markdown question inputs, including Chinese text.
- Generated sample PDF, page extraction, and scanned-PDF guidance.
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

## Remaining integration checks

Before a release, run the README example with downloaded GME weights and a real vision API. Also verify the chosen CUDA/PyTorch combination, replacement API endpoints, and MinerU's model-backed OCR tiers. These tests verify wiring and persistence, not model accuracy, hosted API compatibility, latency, or memory requirements.
