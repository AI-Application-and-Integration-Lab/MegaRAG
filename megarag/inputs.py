"""Question inputs and PDF preprocessing through the isolated MinerU runtime."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid


def extract_queries(path):
    path = Path(path)
    questions = []
    pattern = re.compile(r"^\s*-\s*Question\s+\d+:\s*(.+)$")
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if path.suffix.lower() == ".jsonl":
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            question = (
                (data.get("question") or data.get("text"))
                if isinstance(data, dict)
                else None
            )
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"{path}:{number}: expected a nonempty 'question' or 'text' string"
                )
        else:
            match = pattern.match(line.replace("**", ""))
            if path.suffix.lower() == ".md" and not match:
                continue
            question = match.group(1) if match else line
        questions.append(question.strip())
    return questions


def resolve_mineru_python():
    """Keep the venv executable path intact: resolving symlinks loses its venv."""
    configured = os.environ.get("MINERU_PYTHON")
    if configured:
        candidate = os.path.expanduser(configured)
        executable = shutil.which(candidate)
    else:
        relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
        candidate = str(Path.cwd() / ".venv-mineru" / relative)
        executable = shutil.which(candidate)
    if not executable:
        raise ValueError(
            f"MinerU Python not found: {candidate}. Install the separate MinerU environment "
            "(docs/mineru.md) and set MINERU_PYTHON to its Python executable."
        )
    return str(Path(executable).absolute())


def read_page_manifest(manifest):
    """Validate parsed text and images before loading the embedding model."""
    pages = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(pages, dict) or not pages:
        raise ValueError("Page JSON must be a nonempty object keyed by zero-based page numbers")
    for index in range(len(pages)):
        page = pages.get(str(index))
        if not isinstance(page, dict) or not isinstance(page.get("text"), str):
            raise ValueError(f"Invalid page {index}: expected text and page_image")
        figures = page.get("figure_images", [])
        if not isinstance(figures, list):
            raise ValueError(f"Page {index}: figure_images must be a list")
        for image in [page.get("page_image", ""), *figures]:
            if not isinstance(image, str) or not image or not Path(image).is_file():
                raise ValueError(
                    f"Page {index}: image file not found: {image!r}. "
                    "Keep the parsed assets available and resolve relative paths before ingesting."
                )
        page["page_image"] = str(Path(page["page_image"]).resolve())
        page["figure_images"] = [str(Path(image).resolve()) for image in figures]
    return pages


def prepare_pdf(pdf_path, output_dir, *, max_pages=None):
    """Run MinerU Standard/auto before indexing any PDF, including scanned PDFs."""
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Input must be an existing PDF")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    adapter = Path(__file__).with_name("mineru_adapter.py")
    digest = hashlib.sha256(adapter.read_bytes())
    digest.update(json.dumps({"tier": "standard", "ocr_mode": "auto", "max_pages": max_pages}).encode())
    with pdf_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    cache_dir = Path(output_dir).resolve() / digest.hexdigest()
    manifest = cache_dir / "pages_content.json"
    if manifest.is_file():
        try:
            read_page_manifest(manifest)
        except (ValueError, OSError):
            pass  # Missing/damaged assets require a fresh parse, never a silent fallback.
        else:
            print(f"Reusing MinerU output: {manifest}", flush=True)
            return manifest
    python = resolve_mineru_python()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Preserve failed attempts and all assets referenced by an existing index.
    # Only publish the stable manifest after a successful, validated parse.
    attempt = Path(tempfile.mkdtemp(prefix="parse-", dir=cache_dir))
    command = [python, str(adapter), str(pdf_path), "--output", str(attempt),
               "--tier", "standard", "--ocr-mode", "auto"]
    if max_pages is not None:
        command += ["--max-pages", str(max_pages)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"MinerU parsing failed (exit {exc.returncode}); graph construction was not started. "
            "Check the parser error above and docs/mineru.md, then rerun ingest."
        ) from exc
    pages = read_page_manifest(attempt / "pages_content.json")
    temporary = cache_dir / f".manifest-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(pages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    return manifest


def create_demo(directory):
    import fitz

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / "project.pdf"
    questions = directory / "questions.txt"
    if pdf.exists() or questions.exists():
        raise ValueError("Demo files already exist. Choose another --output directory.")
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Project Atlas", fontsize=22)
        page.insert_text(
            (72, 120),
            "Project Atlas improves access to the city library.\nThe project manager is Maya Chen.\nThe pilot budget is USD 25,000.",
            fontsize=12,
        )
        page = document.new_page()
        page.insert_text((72, 72), "Project Atlas timeline", fontsize=22)
        page.insert_text(
            (72, 120),
            "The pilot starts on March 1, 2027.\nThe review is scheduled for June 1, 2027.\nSuccess means 200 new library members.",
            fontsize=12,
        )
        document.save(str(pdf))
    questions.write_text(
        "Who manages Project Atlas?\nWhat is the budget and when does the pilot start?\n",
        encoding="utf-8",
    )
    return pdf
