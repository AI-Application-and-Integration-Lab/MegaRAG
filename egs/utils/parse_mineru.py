#!/usr/bin/env python3
"""Parse a PDF with MinerU 4.1 in its own environment; emit MegaRAG page assets."""

import argparse
import json
from pathlib import Path


def image_paths(value):
    """Visit nested image/table/chart bodies in the public MiddleJson format."""
    if isinstance(value, dict):
        if value.get("image_path"):
            yield value["image_path"]
        for child in value.values():
            yield from image_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from image_paths(child)


def export_page_assets(result, pdf_path, output_dir, *, dpi=150):
    """Keep MinerU's text rendering and saved crops, and render original PDF pages."""
    import fitz
    from mineru.parser import ParseResult
    from mineru.parser.writer import FileBasedDataWriter
    from mineru.render.contracts import RenderMode

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory is not empty: {output_dir}. Choose a new directory.")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not result.pages:
        raise ValueError("MinerU returned no pages")
    output_dir.mkdir(parents=True, exist_ok=True)
    # save() materializes image bytes and writes their actual paths into MiddleJson.
    result.save(FileBasedDataWriter(str(output_dir)))
    saved = ParseResult.from_json((output_dir / "middle_json.json").read_text(encoding="utf-8"))
    page_images = output_dir / "page_images"
    page_images.mkdir()
    pages = {}
    with fitz.open(pdf_path) as document:
        seen = set()
        for index, page in enumerate(saved.pages):
            if page.page_idx in seen or not 0 <= page.page_idx < len(document):
                raise ValueError(f"Invalid or repeated source page index: {page.page_idx}")
            seen.add(page.page_idx)
            figure_images = []
            for relative in dict.fromkeys(image_paths(page.model_dump(mode="json"))):
                asset = (output_dir / relative).resolve()
                if not asset.is_relative_to(output_dir) or not asset.is_file():
                    raise ValueError(f"Missing or invalid MinerU image: {relative}")
                figure_images.append(str(asset))
            page_image = page_images / f"page_{page.page_idx}.png"
            document[page.page_idx].get_pixmap(dpi=dpi, alpha=False).save(page_image)
            per_page = ParseResult(saved.middle_json.model_copy(update={"pages": [page]}))
            pages[str(index)] = {
                # FULL retains headers/footers that can contain the only text
                # on sparse pages; DEFAULT deliberately omits those blocks.
                "text": per_page.markdown(mode=RenderMode.FULL, asset_base_url=str(output_dir)).strip(),
                "page_image": str(page_image),
                "figure_images": figure_images,
            }
    manifest = output_dir / "pages_content.json"
    manifest.write_text(json.dumps(pages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--tier", choices=("flash", "basic", "standard", "advanced"), default="standard")
    parser.add_argument("--ocr-mode", choices=("auto", "txt", "ocr"), default="auto")
    parser.add_argument("--max-pages", type=int, help="Parse the first N pages; default: all")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if not args.pdf.is_file() or args.pdf.suffix.lower() != ".pdf":
        parser.error("input must be an existing PDF")
    if args.dpi <= 0 or (args.max_pages is not None and args.max_pages <= 0):
        parser.error("--dpi and --max-pages must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory")
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed_version = version("mineru")
    except PackageNotFoundError:
        parser.error("MinerU is not installed. Run this script with the separate MinerU environment's Python.")
    if installed_version != "4.1.3":
        parser.error("This adapter targets MinerU 4.1.3. Install requirements_mineru.txt in a separate environment.")
    from mineru.parser import parse

    print(f"Parsing {args.pdf} with MinerU ({args.tier})…", flush=True)
    result = parse(str(args.pdf.resolve()), tier=args.tier, ocr_mode=args.ocr_mode,
                   page_range=f"1-{args.max_pages}" if args.max_pages else "",
                   image_analysis=args.tier != "flash")
    manifest = export_page_assets(result, args.pdf, args.output, dpi=args.dpi)
    print(f"Saved {len(result.pages)} pages to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
