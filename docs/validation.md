# Validation

The setup update is checked with an offline test suite and package builds. Model quality and GPU performance require separate evaluation.

## Automated coverage

- Package discovery includes the language-model adapters.
- CLI help does not import PyTorch or Transformers.
- Configuration validation, legacy recipe settings, and workspace compatibility.
- Query cache invalidation for model, retrieval settings, and corpus revisions.
- Extraction cache invalidation for changed image contents and model fingerprints.
- TXT, JSONL, and legacy Markdown question inputs, including Chinese text.
- Generated sample PDF, page extraction, and scanned-PDF guidance.
- PDF ingestion, graph/vector persistence, two-step querying, and cache reuse through the CLI, using substitute models and real local storage.
- Completed batch answers survive a later query failure; output files are not overwritten.

Run the checks with:

```bash
pip install -e ".[pdf,dev]"
python -m pytest -q
python -m build
```

The CI workflow targets Python 3.10, 3.11, and 3.12. Local validation for this change uses Python 3.11; the hosted matrix must run after the branch is pushed.

## Remaining integration checks

Before a release, run the README example with downloaded GME weights and a real vision API. Also verify the chosen CUDA/PyTorch combination, any replacement API endpoint, and the MinerU environment. Offline substitutes verify wiring and persistence, not model accuracy, API compatibility, latency, or memory requirements.
