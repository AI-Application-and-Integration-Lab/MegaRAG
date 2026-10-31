"""Real graph/vector storage and CLI, with local model/API substitutes."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from megarag.cli import main
from megarag.inputs import create_demo


def test_ingest_and_query_pipeline(tmp_path, monkeypatch):
    import lightrag.lightrag as light_module
    from lightrag.utils import Tokenizer
    from megarag.llms import openai

    tokenizer = Tokenizer(
        "test",
        SimpleNamespace(
            encode=lambda text: list(text.encode()),
            decode=lambda tokens: bytes(tokens).decode(errors="replace"),
        ),
    )
    monkeypatch.setattr(light_module, "TiktokenTokenizer", lambda *args: tokenizer)

    model = SimpleNamespace()
    model.to = lambda *args, **kwargs: model
    model.eval = lambda: model
    for name in ("get_text_embeddings", "get_image_embeddings", "get_fused_embeddings"):
        setattr(model, name, lambda *args, **kwargs: None)
    transformers = ModuleType("transformers")
    transformers.AutoModel = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: model
    )
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.float32, torch.float16 = "float32", "float16"
    hf = ModuleType("megarag.llms.hf")

    async def embed(model, texts=None, images=None, is_query=False):
        count = len(texts or images or [])
        return np.ones((count, 1536), dtype=np.float32) / np.sqrt(1536)

    hf.hf_gme_embed = embed
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "megarag.llms.hf", hf)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-no-network")

    calls = []

    async def complete(model, prompt, **kwargs):
        calls.append((model, prompt))
        if "response_format" in kwargs:
            return json.dumps(
                {
                    "high_level_keywords": ["project management"],
                    "low_level_keywords": ["Maya Chen", "Project Atlas"],
                }
            )
        if "Entity_types:" in prompt:
            return '("entity"<|>MAYA CHEN<|>person<|>Maya manages Project Atlas.)##("entity"<|>PROJECT ATLAS<|>project<|>A library project.)##("relationship"<|>MAYA CHEN<|>PROJECT ATLAS<|>Maya manages the project.<|>management<|>10)<|COMPLETE|>'
        return "Maya Chen manages Project Atlas."

    monkeypatch.setattr(openai, "openai_complete_if_cache", complete)

    pdf = create_demo(tmp_path / "demo")
    workspace = tmp_path / "index"
    assert main(["ingest", str(pdf), "--workspace", str(workspace)]) == 0
    assert list(workspace.glob("graph_*.graphml"))
    assert list(workspace.glob("vdb_*.json"))
    output = tmp_path / "answers.jsonl"
    assert (
        main(
            [
                "query",
                "Who manages Project Atlas?",
                "--workspace",
                str(workspace),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        json.loads(output.read_text())["answer"] == "Maya Chen manages Project Atlas."
    )
    assert calls and all(model == "gpt-4o-mini" for model, _ in calls)
    before = len(calls)
    assert (
        main(["query", "Who manages Project Atlas?", "--workspace", str(workspace)])
        == 0
    )
    assert len(calls) == before  # Same index and settings reuse the answer cache.


def test_batch_keeps_completed_answers_when_later_query_fails(tmp_path, monkeypatch):
    from megarag import runtime

    calls = []

    async def query(question, **kwargs):
        calls.append(question)
        if len(calls) == 2:
            raise RuntimeError("Simulated API failure")
        return "First answer"

    async def finalize():
        pass

    async def initialize(*args, **kwargs):
        return SimpleNamespace(aquery=query, finalize_storages=finalize), "usage"

    monkeypatch.setattr(runtime, "initialize_rag", initialize)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    questions = tmp_path / "questions.txt"
    questions.write_text("First?\nSecond?\n")
    output = tmp_path / "answers.jsonl"
    command = ["query", "--questions", str(questions), "--output", str(output)]
    assert main(command) == 1
    assert json.loads(output.read_text()) == {
        "question": "First?",
        "answer": "First answer",
    }
    assert main(command) == 1  # Existing output cannot be overwritten.
    assert calls == ["First?", "Second?"]
