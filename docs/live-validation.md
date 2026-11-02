# Live integration validation

Validated on November 2, 2026 (UTC), using synthetic documents and real model inference. These checks cover integration and selected factual answers; they are not a retrieval-quality benchmark.

## Results

| Check | Result |
| --- | --- |
| OpenRouter / `openai/gpt-4o-mini` | Text and image requests passed. Prompt-guided JSON and structured-output keyword extraction both passed. |
| GME on an RTX 5090 | Text, image, and fused embeddings each returned a finite, normalized 1,536-dimensional vector. |
| Native PDF → graph → answers | Two pages indexed; all three factual checks passed. |
| Scanned PDF → MinerU → graph → answers | Two image-only pages parsed in `standard` / `ocr` mode; all three factual checks passed. |
| MinerU assets | Preserved the manager name, total budget, deployment allocation, and membership target; exported one table crop and one chart crop. |
| Fresh-process CLI query | Reloaded the saved index and returned the cached answer with zero additional LLM calls. |
| OpenAI direct endpoint | Blocked by `credit_balance_exhausted`; successful inference through this endpoint remains unverified. |

The test document identifies Lina Torres as Project Vega's manager, with a USD 42,000 budget, USD 20,000 allocated to deployment, and a target of 320 new library members. The scanned version has no selectable text. Both input paths used the same three questions and default graph refinement.

## Tested environment

| Component | Version / configuration |
| --- | --- |
| Python | 3.11.13 |
| GPU / driver | NVIDIA GeForce RTX 5090, 32 GB / 580.126.09 |
| PyTorch / Torchvision | `2.7.1+cu128` / `0.22.1+cu128` |
| MegaRAG Transformers | `4.51.3` |
| Embedding | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`, revision `9cfa6413f704a7c1cf5064d240748e10c876b286` |
| MinerU / DocVortex | `4.1.3` / `0.6.9`, in a separate environment |
| MinerU Transformers | `5.10.1` |
| MinerU inference | Torch small models on CUDA; local llama.cpp VLM on GPU |

GME used `embedding.batch_size: 1` and `llm.max_async: 2`. Peak PyTorch allocated memory was **4,468.9 MiB**; reserved memory was **5,018 MiB**. These figures cover the MegaRAG process, not MinerU or total device memory.

After model loading, native-PDF ingestion took 22.7 seconds and OCR-manifest ingestion took 20.4 seconds. The two pipelines made 34 LLM calls and consumed 876,092 reported tokens in total, excluding standalone API probes. Model downloads, initialization, and OCR are excluded from ingestion timings. This small run does not establish production throughput or cost.

## Repeat with your own document

Follow the [model configuration](models.md) and [MinerU workflow](mineru.md), using fresh workspaces. The CUDA wheel pair used here can be installed in each environment with:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Check the [PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/) for your driver and platform. In the separate MinerU environment, the tested inference dependencies are:

```bash
python -m pip install -r requirements_mineru.txt
python -m pip install 'mineru[torch]==4.1.3' 'transformers==5.10.1'
```

The OCR run selected its backends explicitly:

```bash
export MINERU_MODEL_SMALL_BACKEND=torch
export MINERU_DEVICE_MODE=cuda
export MINERU_MODEL_VLM_ENGINE=llama-cpp

.venv-mineru/bin/python egs/utils/parse_mineru.py scanned.pdf \
  --output parsed/scanned --tier standard --ocr-mode ocr
```

Use a vision-capable API model and verify answers against known document facts. This run did not test hosted SGLang/vLLM servers, other GPUs, other languages, large documents, or OCR accuracy across varied layouts.
