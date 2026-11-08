import json
from pathlib import Path
import subprocess
import sys

import pytest

from megarag import inputs
from megarag.cli import main
from megarag.config import Config, config_fingerprint, prepare_workspace


def test_parse_cache_tracks_content_limit_and_missing_assets(tmp_path, fake_mineru):
    pdf = inputs.create_demo(tmp_path / "demo")
    root = tmp_path / "parsed"
    first = inputs.prepare_pdf(pdf, root, max_pages=1)
    assert inputs.prepare_pdf(pdf, root, max_pages=1) == first
    assert len(fake_mineru) == 1
    full = inputs.prepare_pdf(pdf, root)
    assert full != first
    assert len(json.loads(full.read_text())) == 2
    assert len(fake_mineru) == 2
    image = Path(json.loads(first.read_text())["0"]["page_image"])
    image.unlink()
    assert inputs.prepare_pdf(pdf, root, max_pages=1) == first
    replacement = Path(json.loads(first.read_text())["0"]["page_image"])
    assert replacement != image and replacement.is_file()
    assert len(fake_mineru) == 3
    with pdf.open("ab") as output:
        output.write(b"\n% changed source\n")
    assert inputs.prepare_pdf(pdf, root, max_pages=1) != first
    assert len(fake_mineru) == 4


def test_failed_parse_is_not_reused_and_retry_preserves_partial_files(tmp_path, monkeypatch, fake_mineru):
    pdf = inputs.create_demo(tmp_path / "demo")
    root = tmp_path / "parsed"
    success = inputs.subprocess.run
    partial = []

    def fail(command, *, check):
        output = Path(command[command.index("--output") + 1])
        artifact = output / "partial.txt"
        artifact.write_text("incomplete parser output")
        partial.append(artifact)
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(inputs.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="graph construction was not started"):
        inputs.prepare_pdf(pdf, root)
    assert not list(root.glob("*/pages_content.json"))
    monkeypatch.setattr(inputs.subprocess, "run", success)
    manifest = inputs.prepare_pdf(pdf, root)
    assert manifest.is_file() and partial[0].is_file()
    assert len(fake_mineru) == 1


@pytest.mark.parametrize("failure", ["missing_runtime", "parser_error", "missing_asset"])
def test_cli_does_not_load_models_when_preprocessing_fails(tmp_path, monkeypatch, fake_mineru, failure):
    from megarag import runtime
    pdf = inputs.create_demo(tmp_path / "demo")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    async def forbidden(*args, **kwargs):
        pytest.fail("Model loading must wait until MinerU succeeds")

    monkeypatch.setattr(runtime, "initialize_rag", forbidden)
    if failure == "missing_runtime":
        monkeypatch.setenv("MINERU_PYTHON", str(tmp_path / "missing-python"))
    else:
        success = inputs.subprocess.run

        def broken(command, *, check):
            if failure == "parser_error":
                raise subprocess.CalledProcessError(1, command)
            success(command, check=check)
            output = Path(command[command.index("--output") + 1])
            (output / "page-0.png").unlink()

        monkeypatch.setattr(inputs.subprocess, "run", broken)
    assert main(["ingest", str(pdf), "--workspace", str(tmp_path / "index")]) == 1
    assert not list((tmp_path / "index").glob("graph_*.graphml"))


def test_prepared_json_does_not_require_mineru(tmp_path, monkeypatch):
    from megarag import runtime
    from types import SimpleNamespace
    import fitz

    image = tmp_path / "page.png"
    with fitz.open() as doc:
        doc.new_page().get_pixmap().save(image)
    source = tmp_path / "pages_content.json"
    source.write_text(json.dumps({"0": {"text": "Already parsed", "page_image": str(image)}}))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("MINERU_PYTHON", str(tmp_path / "missing"))
    inserted = []

    async def insert(content, **kwargs):
        inserted.append(json.loads(content))

    async def statuses(*args):
        return {}

    async def finalize():
        pass

    async def initialize(*args, **kwargs):
        return SimpleNamespace(ainsert=insert, finalize_storages=finalize,
                               doc_status=SimpleNamespace(get_docs_by_status=statuses)), "usage"

    monkeypatch.setattr(runtime, "initialize_rag", initialize)
    assert main(["ingest", str(source), "--workspace", str(tmp_path / "index")]) == 0
    assert inserted[0]["0"]["text"] == "Already parsed"


def test_default_mineru_venv_path_preserves_symlink(tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("Unix venv symlink regression")
    executable = tmp_path / ".venv-mineru/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MINERU_PYTHON", raising=False)
    assert inputs.resolve_mineru_python() == str(executable)


def test_legacy_pipeline_fingerprint_requires_rebuild(tmp_path):
    from dataclasses import asdict
    import hashlib
    from megarag import config as config_module

    config = Config()
    data = asdict(config)
    for key in ("device", "batch_size"):
        data["embedding"].pop(key)
    for key in ("api_key_env", "max_async", "timeout"):
        data["llm"].pop(key)
    data["addon_params"].pop("embed_parallel_limit")
    data["prompts"] = hashlib.sha256(Path(config_module.__file__).with_name("prompt.py").read_bytes()).hexdigest()
    legacy = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    assert config_fingerprint(config) != legacy
    (tmp_path / "megarag_manifest.json").write_text(json.dumps({"schema": 1, "config_fingerprint": legacy}))
    with pytest.raises(ValueError, match="new --workspace"):
        prepare_workspace(tmp_path, config, ingest=True)
