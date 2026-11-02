# MegaRAG

Multimodal graph-based retrieval for document question answering. MegaRAG combines a knowledge graph with page-image retrieval to answer questions across text, figures, and tables.

[Paper](https://arxiv.org/abs/2512.20626) · [Model configuration](docs/models.md) · [MinerU workflow](docs/mineru.md) · [Troubleshooting](docs/troubleshooting.md)

Accepted to ACL 2026.

## Install

Use Python 3.10 or newer; Python 3.11 is recommended. The default setup uses a local GME embedding model and a vision-capable OpenAI API model. A CUDA GPU is recommended for embedding inference; CPU execution is supported but slower.

```bash
git clone https://github.com/AI-Application-and-Integration-Lab/MegaRAG.git
cd MegaRAG
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[local,pdf]"
export OPENAI_API_KEY="your-api-key"
```

LightRAG is installed automatically. On Windows, activate the environment with `.venv\Scripts\Activate.ps1` and set the key with `$env:OPENAI_API_KEY="your-api-key"`.

## Quickstart

Create a two-page sample PDF, build its graph, and ask a question:

```bash
megarag init
megarag doctor --config megarag.yaml
megarag demo
megarag ingest demo/project.pdf --config megarag.yaml --workspace demo/index
megarag query "Who manages Project Atlas?" --config megarag.yaml --workspace demo/index
```

The sample identifies **Maya Chen** as the project manager. Answer wording depends on the selected model. The first ingestion downloads the embedding model and makes API calls; `demo` and `doctor` do not call an API.

## Use your own PDF

```bash
megarag ingest ./report.pdf --config megarag.yaml --workspace ./report-index
megarag query "What are the main findings?" --config megarag.yaml --workspace ./report-index
```

Add `--max-pages 3` to try a few pages first. Use a fresh workspace when moving from a page-limited trial to the full document, or when replacing an existing PDF.

The default PDF reader extracts selectable text and renders page images. For scanned PDFs, OCR, or layout-aware table and figure extraction, use the [MinerU workflow](docs/mineru.md).

## Run a question file

Put one question per line in `questions.txt`, including questions in Chinese or other languages:

```text
Who manages Project Atlas?
What is the budget and when does the pilot start?
```

```bash
megarag query --questions questions.txt --output answers.jsonl --config megarag.yaml --workspace demo/index
```

Each completed answer is saved immediately. JSONL input with `{"question": "..."}` is also supported. Output files are not overwritten.

## Configure models

Edit `megarag.yaml` to set the language model, API endpoint, embedding device, or batch size. Pass the same configuration to ingestion and querying.

```yaml
llm:
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
embedding:
  model: Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
  dimension: 1536
  device: auto
  batch_size: 2
```

See [model configuration](docs/models.md) for OpenAI, OpenRouter, SGLang, and vLLM examples and compatibility rules. Changing the model requires a new workspace; an existing index is never silently reused with a different configuration.

## Development

```bash
pip install -e ".[pdf,dev]"
python -m pytest
python -m build
```

Tests use model and API substitutes; they do not download model weights or incur API charges. See [validation](docs/validation.md) for the scope of these checks.

## Citation

```bibtex
@inproceedings{hsiao2026megarag,
  title={MegaRAG: Multimodal Graph-based Retrieval Augmented Generation},
  author={Hsiao, Chi-Hsiang and Wang, Yi-Cheng and Lin, Tzung-Sheng and Yeh, Yi-Ren and Chen, Chu-Song},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  url={https://arxiv.org/abs/2512.20626}
}
```

## License

MegaRAG uses a [custom license](LICENSE). For academic or commercial use, contact the authors as described in the license.

## Acknowledgments

Built on [LightRAG](https://github.com/HKUDS/LightRAG), with document parsing support through [MinerU](https://github.com/opendatalab/MinerU).
