import asyncio
from types import SimpleNamespace

from lightrag.base import QueryParam
from megarag.utils import query_cache_hash


def test_keyword_flag_reaches_structured_output(monkeypatch):
    from megarag.llms import openai

    captured = []

    async def complete(*args, **kwargs):
        captured.append(kwargs)
        return "{}"

    monkeypatch.setattr(openai, "openai_complete_if_cache", complete)
    asyncio.run(openai.gpt_4o_mini_complete("keywords", keyword_extraction=True))
    assert captured[-1]["response_format"] is openai.GPTKeywordExtractionFormat
    asyncio.run(openai.gpt_4o_mini_complete("answer"))
    assert "response_format" not in captured[-1]


def test_query_cache_separates_model_corpus_and_settings():
    param = QueryParam(mode="mix_two_step", chunk_top_k=6)
    config = {"llm_model_name": "first", "addon_params": {"_cache_revision": "one"}}
    first = query_cache_hash("Question", param, config)
    assert first == query_cache_hash("Question", param, config)
    assert first != query_cache_hash("Question", param, config, "different prompt")
    param.chunk_top_k = 3
    assert first != query_cache_hash("Question", param, config)
    param.chunk_top_k = 6
    config["llm_model_name"] = "second"
    assert first != query_cache_hash("Question", param, config)
    config["llm_model_name"] = "first"
    config["addon_params"]["_cache_revision"] = "two"
    assert first != query_cache_hash("Question", param, config)


def test_extraction_cache_fingerprints_images_and_models(tmp_path, monkeypatch):
    from megarag import utils

    hashes = []

    async def handle(cache, key, *args, **kwargs):
        hashes.append(key)
        return "cached", None, None, None

    monkeypatch.setattr(utils, "handle_cache", handle)
    image = tmp_path / "page.png"
    image.write_bytes(b"first image")
    cache = SimpleNamespace(
        global_config={"addon_params": {"_model_fingerprint": "first"}}
    )

    async def run():
        await utils.use_llm_func_with_cache(
            "extract", None, input_images=[str(image)], llm_response_cache=cache
        )
        image.write_bytes(b"second image")
        await utils.use_llm_func_with_cache(
            "extract", None, input_images=[str(image)], llm_response_cache=cache
        )
        cache.global_config["addon_params"]["_model_fingerprint"] = "second"
        await utils.use_llm_func_with_cache(
            "extract", None, input_images=[str(image)], llm_response_cache=cache
        )

    asyncio.run(run())
    assert len(set(hashes)) == 3
