"""Input preparation for plain questions and text-based PDFs."""

import hashlib
import json
from pathlib import Path
import re


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


def prepare_pdf(pdf_path, output_dir, *, max_pages=None):
    """Extract selectable text and render page images; no OCR or layout analysis."""
    import fitz

    pdf_path = Path(pdf_path).resolve()
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:20]
    output_dir = Path(output_dir).resolve() / digest
    pages = {}
    with fitz.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError("PDF is password protected. Provide an unlocked copy.")
        count = min(len(document), max_pages) if max_pages else len(document)
        if not count:
            raise ValueError("PDF has no pages")
        output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            page = document[index]
            image_path = output_dir / f"page_{index + 1:04d}.png"
            page.get_pixmap(dpi=150, alpha=False).save(str(image_path))
            pages[str(index)] = {
                "text": page.get_text().strip(),
                "page_image": str(image_path),
                "figure_images": [],
            }
    if not any(page["text"] for page in pages.values()):
        raise ValueError(
            "No selectable text found. Use the MinerU workflow for scanned PDFs (docs/mineru.md)."
        )
    manifest = output_dir / f"pages_{count}.json"
    manifest.write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
