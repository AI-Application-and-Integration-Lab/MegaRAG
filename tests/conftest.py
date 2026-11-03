"""Parser substitute for tests that exercise graph storage without OCR weights."""

import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def fake_mineru(monkeypatch):
    from megarag import inputs
    import fitz

    calls = []
    monkeypatch.setenv("MINERU_PYTHON", sys.executable)

    def run(command, *, check):
        assert check
        assert Path(command[1]).name == "mineru_adapter.py"
        assert command[command.index("--tier") + 1] == "standard"
        assert command[command.index("--ocr-mode") + 1] == "auto"
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        limit = int(command[command.index("--max-pages") + 1]) if "--max-pages" in command else None
        pages = {}
        with fitz.open(command[2]) as doc:
            for index in range(min(len(doc), limit) if limit else len(doc)):
                image = output / f"page-{index}.png"
                doc[index].get_pixmap().save(image)
                pages[str(index)] = {
                    "text": doc[index].get_text(),
                    "page_image": str(image),
                    "figure_images": [],
                }
        (output / "pages_content.json").write_text(json.dumps(pages))

    monkeypatch.setattr(inputs.subprocess, "run", run)
    return calls
