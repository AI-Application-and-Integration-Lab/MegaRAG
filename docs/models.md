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

## Switch API endpoints

Copy a preset or pass it directly with `--config`:

| Endpoint | Configuration | Authentication |
| --- | --- | --- |
| OpenAI | [`egs/configs/openai.yaml`](../egs/configs/openai.yaml) | `OPENAI_API_KEY` |
| OpenRouter | [`egs/configs/openrouter.yaml`](../egs/configs/openrouter.yaml) | `OPENROUTER_API_KEY` |
| SGLang | [`egs/configs/sglang.yaml`](../egs/configs/sglang.yaml) | Disabled in example |
| vLLM | [`egs/configs/vllm.yaml`](../egs/configs/vllm.yaml) | Disabled in example |

For example, use OpenRouter for the sample PDF:

```bash
export OPENROUTER_API_KEY="your-api-key"
megarag demo
megarag doctor --config egs/configs/openrouter.yaml
megarag ingest demo/project.pdf --config egs/configs/openrouter.yaml --workspace demo/openrouter-index
megarag query "Who manages Project Atlas?" --config egs/configs/openrouter.yaml --workspace demo/openrouter-index
```

Skip `megarag demo` if the sample already exists. Use the same configuration for ingestion and querying; select a new workspace for each model configuration.

For SGLang or vLLM, start your model server separately, then edit `model` and `base_url` to match it. The SGLang preset follows [KG4VD's Qwen configuration](https://github.com/AI-Application-and-Integration-Lab/KG4VD/blob/aba29b2e6f095b70c82542d4735a3937944e4f3e/src/kg4vd/config/presets/llm.sglang.qwen36.yaml); the vLLM preset uses a placeholder served model name. These files configure a client; they do not launch a server.

```bash
megarag ingest demo/project.pdf --config egs/configs/sglang.yaml --workspace demo/sglang-index
megarag query "Who manages Project Atlas?" --config egs/configs/sglang.yaml --workspace demo/sglang-index
```

The endpoint must support Chat Completions **and image inputs**, including multiple images per request. A text-only model cannot replace MegaRAG's vision model. With `structured_output: false`, keyword extraction uses prompt-guided JSON; enable structured output only when the endpoint supports JSON schema.

Set `api_key_env: ""` only for a server without authentication. MegaRAG sends an SDK placeholder key (`EMPTY`) without reading `OPENAI_API_KEY`. For an authenticated server, set `api_key_env: LOCAL_API_KEY` and export that variable.

Provider-specific request parameters go in `extra_body`, following the OpenAI-compatible approach used by [KG4VD](https://github.com/AI-Application-and-Integration-Lab/KG4VD). For compatible Qwen chat templates:

```yaml
llm:
  model: Qwen/Qwen3.6-35B-A3B-FP8
  base_url: http://127.0.0.1:8004/v1
  api_key_env: ""
  structured_output: false
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
```

Only pass parameters supported by your model server. `extra_body` cannot override `model`, `messages`, `stream`, or `response_format`; changing its values requires a fresh workspace. Python callers may override individual top-level `extra_body` keys per request.

The tests check request construction through the OpenAI SDK with a local HTTP substitute. They do not validate hosted model availability, server deployment, or answer quality. See the provider documentation for [OpenRouter](https://openrouter.ai/docs/quickstart), [SGLang](https://docs.sglang.io/docs/basic_usage/openai_api_completions), and [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).

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
