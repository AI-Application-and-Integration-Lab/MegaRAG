"""Run in the separate MinerU environment; no OCR weights or LLM API needed."""

import base64
import importlib.util
import io
import json
from pathlib import Path

import pytest

pytest.importorskip("mineru")
import fitz
from mineru.parser import ParseResult
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "parse_mineru", Path(__file__).parents[1] / "egs/utils/parse_mineru.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.mark.parametrize("limit, expected", [(None, 12), (3, 3)])
def test_real_flash_parser_all_pages_and_limit(tmp_path, limit, expected):
    pdf = tmp_path / "source.pdf"
    with fitz.open() as document:
        for index in range(12):
            page = document.new_page()
            page.insert_text((72, 72), f"Project Atlas page {index + 1}. Maya Chen manages the project.")
        document.save(pdf)
    output = tmp_path / "parsed"
    args = [str(pdf), "--output", str(output), "--tier", "flash", "--ocr-mode", "txt", "--dpi", "72"]
    if limit:
        args += ["--max-pages", str(limit)]
    assert adapter.main(args) == 0
    pages = json.loads((output / "pages_content.json").read_text())
    assert len(pages) == expected
    for index, page in pages.items():
        assert f"page {int(index) + 1}." in page["text"]
        assert Path(page["page_image"]).is_file()
    with pytest.raises(SystemExit):
        adapter.main(args)  # Do not overwrite assets used by an existing index.


def test_nested_table_and_figure_assets(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    payload = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    result = ParseResult.from_dict({
        "schema": "docvortex.middle", "schema_version": "2.0",
        "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": "4.1.3"}},
        "is_full_document": True,
        "pages": [{"page_idx": 0, "blocks": [
            {"type": "table", "index": 0, "bbox": [0.1, 0.1, 0.9, 0.4], "content": [
                {"type": "table_body", "index": 0, "content": "<table><tr><td>Budget</td><td>42</td></tr></table>",
                 "image_base64": payload},
                {"type": "table_caption", "content": [{"type": "text", "content": "Table caption"}]},
            ]},
            {"type": "image", "index": 1, "bbox": [0.1, 0.5, 0.9, 0.9], "content": [
                {"type": "image_body", "index": 1, "content": "", "image_base64": payload},
                {"type": "image_caption", "content": [{"type": "text", "content": "Figure caption"}]},
            ]},
        ]}],
    })
    pdf = tmp_path / "source.pdf"
    with fitz.open() as document:
        document.new_page()
        document.save(pdf)
    manifest = adapter.export_page_assets(result, pdf, tmp_path / "parsed")
    page = json.loads(manifest.read_text())["0"]
    assert all(text in page["text"] for text in ["Budget", "42", "Table caption", "Figure caption"])
    assert page["figure_images"]
    assert all(Path(path).read_bytes() == buffer.getvalue() for path in page["figure_images"])
    assert Path(page["page_image"]).is_absolute()
