#!/usr/bin/env python3
"""
build_page_assets.py
--------------------

Merge per-page text and image metadata produced by your PDF preprocessing
pipeline, then save the result as JSON.

Example
-------
python build_page_assets.py \
    --working-dir ./dumps/World_History_Volume_1/auto \
    --output  ./dumps/World_History_Volume_1/pages_content.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from bs4 import BeautifulSoup

def load_json(path: str | Path) -> Dict:
    """Load a JSON file and return its content as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """Clean text by removing null bytes (0x00) and whitespace

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    return text.strip().replace("\x00", "")

def collect_page_images(page: Dict, img_root: Path) -> List[str]:
    """
    Collect every embedded image (including table snapshots) on a single page
    and return their absolute (string) paths.
    """
    paths: set[str] = set()
    for sec in ("images", "tables"):
        for item in page.get(sec, []):
            for blk in item.get("blocks", []):
                if blk.get("type") in {"image_body", "table_body"}:
                    try:
                        rel_path = blk["lines"][0]["spans"][0]["image_path"]
                        paths.add(str(img_root / rel_path))
                    except (KeyError, IndexError, TypeError):
                        print(
                            f"[WARN] image_path missing: page {page.get('page_idx')}, section {sec}"
                        )
    return sorted(paths)

def concat_page_content_by_idx(data: List[Dict], total_pages: int) -> Dict[int, Dict]:
    """
    Merge every textual element that belongs to the same page.

    Returns a dict:
        {page_idx: {"text": <concatenated text>}}
    """
    page_texts: defaultdict[int, str] = defaultdict(str)

    for item in data:
        page_idx = item.get("page_idx")
        if page_idx is None:
            continue

        # 1. Plain text
        if item.get("type") == "text":
            text = item.get("text", "").strip()
            if text:
                page_texts[page_idx] += text + "\n"

        # 2. Captions & footnotes
        for key in ("img_caption", "img_footnote", "table_caption", "table_footnote"):
            for txt in item.get(key, []):
                txt = txt.strip()
                if txt:
                    page_texts[page_idx] += txt + "\n"

        # 3. Table body (HTML → rows)
        if item.get("type") == "table" and "table_body" in item:
            html = item["table_body"]
            soup = BeautifulSoup(html, "html.parser")
            for row in soup.find_all("tr"):
                cells = [
                    cell.get_text(strip=True) for cell in row.find_all(["td", "th"])
                ]
                if cells:
                    page_texts[page_idx] += " | ".join(cells) + "\n"

    # Ensure every page index exists
    for idx in range(total_pages):
        page_texts[idx] = {"text": clean_text(page_texts.get(idx, ""))}

    return dict(page_texts)

def prepare_page_assets(
    txt_json: Path,
    middle_json: Path,
    img_root: Path,
    page_img_root: Path,
    num_pages: int,
) -> Dict[int, Dict]:
    """
    Assemble per-page assets:

        pages_content[page_idx] = {
            "text":          <merged text>,
            "page_image":    <main page PNG/JPG>,
            "figure_images": [<embedded fig/table images>]
        }
    """
    data = load_json(txt_json)
    pages_content = concat_page_content_by_idx(data, num_pages)

    pdf_info = load_json(middle_json).get("pdf_info", [])
    if len(pdf_info) != num_pages:
        print(
            f"[WARN] Middle JSON has {len(pdf_info)} pages, "
            f"but --pages says {num_pages}. Using min(len(pdf_info), --pages)."
        )

    additional_page_image_sets = [
        collect_page_images(page, img_root) for page in pdf_info[:num_pages]
    ]
    # pad if middle_json shorter than num_pages
    additional_page_image_sets += [[]] * (num_pages - len(additional_page_image_sets))

    page_img_root = Path(page_img_root)
    page_img_sets = sorted(
        page_img_root.iterdir(),
        key=lambda p: int(p.stem.split("page_")[-1])  # expects "...page_XX.png"
    )

    if len(page_img_sets) != num_pages:
        print(
            f"[WARN] Found {len(page_img_sets)} page images in {page_img_root}, "
            f"but --pages is {num_pages}. Continuing with min()."
        )

    for idx in pages_content:
        pages_content[idx]["page_image"] = str(page_img_sets[int(idx)]) if int(idx) < len(
            page_img_sets
        ) else ""
        pages_content[idx]["figure_images"] = additional_page_image_sets[int(idx)]

    _pages_content = {}
    for idx in range(len(pages_content)):
        _pages_content[idx] = pages_content[idx]

    print(
        "Total embedded images collected:",
        sum(map(len, additional_page_image_sets)),
    )
    return _pages_content

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-page text & image metadata and save to JSON."
    )
    parser.add_argument(
        "--working-dir",
        required=True,
        type=Path,
        help="Folder that contains *_middle.json, *_content_list.json, "
        "'images/' and 'page_images/'.",
    )
    parser.add_argument(
        "--output",
        default="pages_content.json",
        type=Path,
        help="Destination JSON file (default: pages_content.json).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    wdir = args.working_dir
    if not wdir.exists():
        raise FileNotFoundError(f"--working-dir {wdir} not found")

    filename = wdir.parent.name if wdir.name == "auto" else wdir.name

    middle_json = wdir / f"{filename}_middle.json"
    txt_json = wdir / f"{filename}_content_list.json"
    img_root = wdir / "images"
    page_img_root = wdir / "page_images"

    for p in (middle_json, txt_json, img_root, page_img_root):
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    pages_content = prepare_page_assets(
        txt_json=txt_json,
        middle_json=middle_json,
        img_root=img_root,
        page_img_root=page_img_root,
        num_pages=len(os.listdir(str(page_img_root))),
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pages_content, f, ensure_ascii=False, indent=2)

    print(f"Saved page-asset manifest → {args.output.resolve()}")

if __name__ == "__main__":
    main()
