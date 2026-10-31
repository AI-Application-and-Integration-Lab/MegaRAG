# Model configuration

Run `megarag init` to create `megarag.yaml`. Pass it with `--config` to both ingestion and querying. Omit `--config` to use built-in defaults; the CLI does not automatically load a YAML file from the current directory.

## Language model

The same vision-capable model handles graph extraction, page-based answering, and answer fusion.

```yaml
llm:
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY
  structured_output: true
  max_async: 4
  timeout: 120
```

`api_key_env` names an environment variable; keep the key itself out of the YAML file. `timeout` is in seconds. `structured_output` controls keyword extraction through the API's structured-output interface.

## Local OpenAI-compatible server

Start a server that supports chat completions and image inputs. Match `model` to its served model name. For example, if your server exposes a vision model as `my-vision-model`:

```yaml
llm:
  model: my-vision-model
  base_url: http://localhost:8000/v1
  api_key_env: LOCAL_API_KEY
  structured_output: false
```

For a server without authentication:

```bash
export LOCAL_API_KEY="unused"
megarag doctor --config local.yaml
megarag ingest demo/project.pdf --config local.yaml --workspace demo/local-index
megarag query "Who manages Project Atlas?" --config local.yaml --workspace demo/local-index
```

Save the YAML above as `local.yaml` first. With `structured_output: false`, keyword extraction relies on prompt-guided JSON. Test the two-page example before a large ingestion: OpenAI-compatible endpoints differ in image handling and output-format support. Model-server deployment and specific replacement models are not validated by the offline test suite.

## Embedding model

```yaml
embedding:
  model: Alibaba-NLP/gme-Qwen2-VL-2B-Instruct
  revision: main
  dimension: 1536
  max_tokens: 32768
  device: auto
  batch_size: 2
```

`auto` selects CUDA when available, otherwise CPU. Use `cpu` or `cuda:0` to select a device explicitly. Reduce `batch_size` if GPU memory is limited.

This release uses the **GME adapter**. A different embedding family needs an adapter implementing text, image, and fused embeddings; changing its model ID alone is insufficient. `dimension` must match the model output. `max_tokens` is embedding-interface metadata and does not override the model's own processor limits.

For reproducible runs, replace `revision: main` with a model commit ID before creating the workspace. A moving revision can change upstream without changing the configured name.

## Retrieval and extraction

```yaml
addon_params:
  language: English
  chunk_top_k: 6
  embed_parallel_limit: 1
  entity_extract_max_gleaning: 0
  entity_refine_max_times: 1
```

`language` controls extraction prompt language. Add `entity_types` as a list to specialize the graph, for example `[organization, person, project, date]`.

## Existing workspaces

Each CLI workspace records a configuration fingerprint and embedding metadata. Model, endpoint, prompt, or retrieval-setting changes require a fresh workspace. Device, embedding batch size, embedding concurrency, API key variable, timeout, and LLM concurrency can change without rebuilding.

```bash
megarag ingest demo/project.pdf --config updated.yaml --workspace demo/updated-index
```

Query caches include the model fingerprint, retrieval settings, and an ingestion revision. Ingestion rotates that revision so earlier answers are not reused after adding documents. Use one writer per workspace, and query after ingestion finishes.

Legacy indexes without a manifest are preserved but not automatically adopted. Rebuild into a new directory. Existing recipe YAML files with top-level addon parameters are still accepted by the research scripts.
