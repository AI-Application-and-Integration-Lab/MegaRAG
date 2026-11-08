import asyncio
import json
from pathlib import Path

import httpx
from openai import AsyncOpenAI
import pytest

from megarag.config import Config, LLMConfig, config_fingerprint, load_config
from megarag.runtime import create_llm_func


@pytest.mark.parametrize("name", ["openai", "openrouter", "sglang", "vllm"])
def test_presets_send_chat_and_images(name, tmp_path, monkeypatch):
    from megarag.llms import openai

    config = load_config(Path(__file__).parents[1] / "egs/configs" / f"{name}.yaml")
    if config.llm.api_key_env:
        monkeypatch.setenv(config.llm.api_key_env, "test-key")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    requests = []
    clients = []

    def respond(request):
        requests.append(request)
        payload = json.loads(request.content)
        content = (
            '{"high_level_keywords": ["topic"], "low_level_keywords": ["name"]}'
            if "response_format" in payload else "A user compares model outputs."
        )
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": config.llm.model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
        })

    def client_factory(*, api_key, base_url, client_configs):
        client = AsyncOpenAI(api_key=api_key, base_url=base_url,
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "create_openai_async_client", client_factory)
    picture = tmp_path / "page.png"
    picture.write_bytes(b"image payload")

    async def run():
        complete = create_llm_func(config)
        await complete("keywords", keyword_extraction=True)
        return await complete("Describe this page", input_images=[str(picture)],
                              system_prompt="Use the evidence", history_messages=[
                                  {"role": "user", "content": "Earlier question"}],
                              extra_body={"temperature": 0})

    assert asyncio.run(run()) == "A user compares model outputs."
    assert all(client.is_closed() for client in clients)
    assert str(requests[-1].url) == config.llm.base_url + "/chat/completions"
    expected_key = "test-key" if config.llm.api_key_env else "EMPTY"
    assert requests[-1].headers["authorization"] == f"Bearer {expected_key}"
    keywords, answer = (json.loads(request.content) for request in requests)
    assert ("response_format" in keywords) == config.llm.structured_output
    assert answer["model"] == config.llm.model
    assert answer["temperature"] == 0
    assert answer["messages"][0]["role"] == "system"
    assert answer["messages"][1]["content"] == "Earlier question"
    assert answer["messages"][-1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    for key, value in config.llm.extra_body.items():
        assert answer[key] == value
    assert "temperature" not in config.llm.extra_body


@pytest.mark.parametrize("body", [[], {"model": "other"}, {"messages": []},
                                  {1: "bad key"}, {"temperature": float("nan")}])
def test_invalid_extra_body(body):
    with pytest.raises(ValueError, match="extra_body"):
        Config(llm=LLMConfig(extra_body=body)).validate()


def test_auth_and_request_settings(monkeypatch):
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    config = Config(llm=LLMConfig(api_key_env="MISSING_TEST_KEY"))
    with pytest.raises(ValueError, match="MISSING_TEST_KEY"):
        create_llm_func(config)
    before = config_fingerprint(config)
    config.llm.api_key_env = ""
    assert config.llm.resolve_api_key() == "EMPTY"
    assert config_fingerprint(config) == before
    config.llm.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    assert config_fingerprint(config) != before
