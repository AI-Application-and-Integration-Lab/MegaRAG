# Troubleshooting

Start with:

```bash
megarag doctor --config megarag.yaml
```

This checks installed modules and the configured API-key environment variable. It does not download models or test the remote endpoint.

| Symptom | Action |
| --- | --- |
| `megarag: command not found` | Activate the environment used for installation. |
| Missing `torch`, `transformers`, or `accelerate` | Run `pip install -e ".[local]"` from the checkout. |
| Missing `fitz` | Run `pip install -e ".[pdf]"`. |
| CUDA is unavailable | Install a PyTorch build matching your CUDA environment, or set `embedding.device: cpu`. |
| GPU runs out of memory | Set `embedding.batch_size: 1` and `addon_params.embed_parallel_limit: 1`; close other GPU workloads. |
| Missing API key | Export the environment variable named by `llm.api_key_env`. |
| Endpoint rejects structured output | If it supports prompt-guided JSON, set `llm.structured_output: false` and use a new workspace. |
| Workspace configuration changed | Use a fresh `--workspace` with the updated config and ingest again. |
| Workspace has no model manifest | Rebuild the legacy index into a new directory. Do not delete the original as a troubleshooting step. |
| Workspace ingestion is incomplete | Resolve the parser/model/API error and rerun `megarag ingest` before querying. A manifest alone does not indicate a completed index. |
| No selectable text found | Use the [MinerU workflow](mineru.md) for scanned PDFs. |
| Image file not found | Keep parsed assets in place; resolve paths in prepared page JSON from the current directory. |
| No questions found | Use one question per line in `.txt`, or `{"question": "..."}` objects in `.jsonl`. |

The first model download and ingestion take longer than subsequent queries. Inspect the preceding model or API error when ingestion fails; rerun with the same configuration after resolving it.
