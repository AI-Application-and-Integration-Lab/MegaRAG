"""Shared initialization for the CLI and research recipes."""

import asyncio
import os

from .config import Config, config_fingerprint, prepare_workspace


async def initialize_rag(working_dir, addon_params=None, *, config=None, ingest=False):
    config = config or Config(addon_params=addon_params or Config().addon_params)
    config.validate()
    api_key = os.environ.get(config.llm.api_key_env)
    if not api_key:
        raise ValueError(f"Set {config.llm.api_key_env} before running inference.")
    revision = prepare_workspace(working_dir, config, ingest=ingest)

    import torch
    from transformers import AutoModel
    from lightrag.utils import TokenTracker, wrap_embedding_func_with_attrs
    from lightrag.types import GPTKeywordExtractionFormat
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from . import MegaRAG
    from .llms.hf import hf_gme_embed
    from .llms.openai import openai_complete_if_cache

    device = config.embedding.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = (
        AutoModel.from_pretrained(
            config.embedding.model,
            revision=config.embedding.revision,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    required_methods = (
        "get_text_embeddings",
        "get_image_embeddings",
        "get_fused_embeddings",
    )
    if not all(callable(getattr(embed_model, name, None)) for name in required_methods):
        raise ValueError(
            "The embedding model must implement the GME embedding interface. Other model families need their own adapter."
        )
    semaphore = asyncio.Semaphore(config.addon_params.get("embed_parallel_limit", 1))

    @wrap_embedding_func_with_attrs(
        embedding_dim=config.embedding.dimension,
        max_token_size=config.embedding.max_tokens,
    )
    async def embed_func(texts=None, images=None, is_query=False):
        async with semaphore:
            result = await hf_gme_embed(embed_model, texts, images, is_query)
            if result.ndim != 2 or result.shape[1] != config.embedding.dimension:
                raise ValueError(
                    f"Embedding shape {result.shape} does not match dimension {config.embedding.dimension}"
                )
            return result

    token_tracker = TokenTracker()

    async def llm_func(
        prompt,
        input_images=None,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ):
        if keyword_extraction and config.llm.structured_output:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        return await openai_complete_if_cache(
            config.llm.model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            input_images=input_images,
            base_url=config.llm.base_url,
            api_key=api_key,
            token_tracker=token_tracker,
            timeout=config.llm.timeout,
            **kwargs,
        )

    addon = {
        **config.addon_params,
        "_cache_revision": revision,
        "_model_fingerprint": config_fingerprint(config),
    }
    rag = MegaRAG(
        working_dir=str(working_dir),
        workspace="",
        llm_model_func=llm_func,
        llm_model_name=config.llm.model,
        llm_model_max_async=config.llm.max_async,
        embedding_func=embed_func,
        embedding_batch_num=config.embedding.batch_size,
        entity_extract_max_gleaning=addon.get("entity_extract_max_gleaning", 0),
        addon_params=addon,
        auto_manage_storages_states=False,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag, token_tracker
