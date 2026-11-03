import json
from pathlib import Path

import pytest
from setuptools import find_packages

from megarag.cli import main
from megarag.config import Config, config_fingerprint, load_config, prepare_workspace
from megarag.inputs import create_demo, extract_queries, prepare_pdf


def test_package_contains_model_adapters():
    assert "megarag.llms" in find_packages(where=str(Path(__file__).parents[1]))


def test_help_does_not_import_inference_stack():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from megarag.cli import parser; parser().parse_args(['doctor']); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_init_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "megarag.yaml"
    assert main(["init", "--output", str(path)]) == 0
    assert load_config(path) == Config()
    original = path.read_bytes()
    assert main(["init", "--output", str(path)]) == 1
    assert path.read_bytes() == original


def test_config_legacy_and_bad_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("language: Chinese\nchunk_top_k: 3\n")
    assert load_config(path).addon_params["language"] == "Chinese"
    path.write_text("llm:\n  model_typo: foo\n")
    with pytest.raises(ValueError, match="Invalid model configuration"):
        load_config(path)
    path.write_text("embedding:\n  batch_size: 0\n")
    with pytest.raises(ValueError, match="positive"):
        load_config(path)


def test_workspace_model_guard_and_cache_revision(tmp_path):
    workspace = tmp_path / "index"
    config = Config()
    revision = prepare_workspace(workspace, config, ingest=True)
    with pytest.raises(ValueError, match="ingestion is incomplete"):
        prepare_workspace(workspace, config)
    (workspace / "kv_store_doc_status.json").write_text(' {"doc": {"status": "processed"}}')
    assert prepare_workspace(workspace, config) == revision
    assert prepare_workspace(workspace, config, ingest=True) != revision
    config.embedding.device = "cpu"
    config.embedding.batch_size = 1
    prepare_workspace(workspace, config)
    config.embedding.model = "another-model-with-the-same-dimension"
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_workspace(workspace, config)
    assert (workspace / "megarag_manifest.json").exists()


def test_workspace_rejects_legacy_and_missing(tmp_path):
    with pytest.raises(ValueError, match="ingest.*first"):
        prepare_workspace(tmp_path / "missing", Config())
    (tmp_path / "vdb.json").write_text("original")
    with pytest.raises(ValueError, match="no model manifest"):
        prepare_workspace(tmp_path, Config(), ingest=True)
    assert (tmp_path / "vdb.json").read_text() == "original"


def test_fingerprint_includes_endpoint_and_retrieval():
    config = Config()
    original = config_fingerprint(config)
    config.llm.base_url = "http://localhost:8000/v1"
    assert config_fingerprint(config) != original
    config = Config()
    config.addon_params["chunk_top_k"] = 3
    assert config_fingerprint(config) != original


def test_plain_and_legacy_questions(tmp_path):
    path = tmp_path / "questions.txt"
    path.write_text(
        "Who manages the project?\n\n請摘要文件。\n- Question 3: Budget?\n",
        encoding="utf-8",
    )
    assert extract_queries(path) == [
        "Who manages the project?",
        "請摘要文件。",
        "Budget?",
    ]
    path = tmp_path / "questions.md"
    path.write_text("# Questions\n- **Question 1:** Who?\n")
    assert extract_queries(path) == ["Who?"]


@pytest.mark.parametrize("line", ["{bad}", "{}", "[]", '{"question": 42}'])
def test_jsonl_reports_invalid_line(tmp_path, line):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question": "Valid?"}\n' + line)
    with pytest.raises(ValueError, match=":2:"):
        extract_queries(path)


def test_demo_and_page_manifest(tmp_path, fake_mineru):
    pdf = create_demo(tmp_path / "demo")
    manifest = prepare_pdf(pdf, tmp_path / "assets", max_pages=1)
    pages = json.loads(manifest.read_text())
    assert list(pages) == ["0"]
    assert "Maya Chen" in pages["0"]["text"]
    assert Path(pages["0"]["page_image"]).is_absolute()
    assert Path(pages["0"]["page_image"]).is_file()
    assert len(extract_queries(pdf.with_name("questions.txt"))) == 2
    with pytest.raises(ValueError, match="already exist"):
        create_demo(pdf.parent)


def test_image_only_pdf_also_runs_mineru(tmp_path, fake_mineru):
    import fitz

    path = tmp_path / "empty.pdf"
    with fitz.open() as document:
        document.new_page()
        document.save(path)
    manifest = prepare_pdf(path, tmp_path / "assets")
    assert fake_mineru
    assert list(json.loads(manifest.read_text())) == ["0"]


@pytest.mark.parametrize("status", ["pending", "processing", "failed"])
def test_unfinished_workspace_rejected(tmp_path, status):
    prepare_workspace(tmp_path, Config(), ingest=True)
    (tmp_path / "kv_store_doc_status.json").write_text(json.dumps({
        "old": {"status": "processed"}, "new": {"status": status}}))
    with pytest.raises(ValueError, match="ingestion is incomplete"):
        prepare_workspace(tmp_path, Config())


@pytest.mark.parametrize("content", ["[]", "false", "0", "addon_params: {1: value}", "llm: {timeout: .inf}", "llm: {}\n1: value"])
def test_malformed_config_is_reported(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path)
