import os
import fitz
import math
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def _render_pages(pdf_path: str,
                  output_dir: str,
                  page_numbers: list[int],
                  dpi: int,
                  fmt: str) -> None:
    """
    Render *page_numbers* (1‑based) of *pdf_path* to *output_dir*.
    Each worker opens its own document instance → no cross‑process state.
    """
    doc   = fitz.open(pdf_path)
    zoom  = dpi / 72.0
    mat   = fitz.Matrix(zoom, zoom)
    alpha = False if fmt == "jpeg" else True
    stem  = Path(pdf_path).stem

    for pno in page_numbers:
        page = doc.load_page(pno - 1)              # 0‑based internally
        pix  = page.get_pixmap(matrix=mat, alpha=alpha)
        out  = Path(output_dir) / f"{stem}_page_{pno:03d}.{fmt}"
        if fmt == "jpeg":
            # PyMuPDF ≥1.22 lets you set JPEG quality
            try:
                pix.save(out, quality=90)
            except TypeError:
                pix.save(out)                      # fallback for older versions
        else:
            pix.save(out)

    doc.close()

def _chunk_pages(page_numbers: list[int], chunks: int) -> list[list[int]]:
    """Split *page_numbers* into *chunks* balanced sub‑lists."""
    k, m  = divmod(len(page_numbers), chunks)       # k pages per chunk, m chunks get +1
    out   = []
    idx   = 0
    for i in range(chunks):
        step = k + (1 if i < m else 0)
        out.append(page_numbers[idx: idx + step])
        idx += step
    return [chunk for chunk in out if chunk]        # drop empties if < workers

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast PDF→image converter using PyMuPDF + multiprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_pdf",  help="Path to input PDF.")
    p.add_argument("output_dir", help="Directory for output images.")
    p.add_argument("--dpi", type=int, default=200,
                   help="Render resolution (dots per inch).")
    p.add_argument("--jpeg", action="store_true",
                   help="Output JPEG instead of PNG (faster, smaller).")
    p.add_argument("--jobs", type=int, default=os.cpu_count(),
                   help="Number of parallel workers (default: all logical CPUs).")
    # NEW FLAGS
    p.add_argument("--start-page", type=int, default=1,
                   help="First page to render (1‑based, inclusive).")
    p.add_argument("--end-page",   type=int, default=None,
                   help="Last page to render (1‑based, inclusive).")

    return p.parse_args()

def main() -> None:
    args   = parse_args()
    fmt    = "jpeg" if args.jpeg else "png"
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Discover total page count once (very cheap)
    total_pages = fitz.open(args.input_pdf).page_count

    # Clamp & validate the requested range
    start = max(1, args.start_page)
    end   = total_pages if args.end_page is None else min(args.end_page, total_pages)
    if start > end:
        raise ValueError(f"start‑page ({start}) cannot be after end‑page ({end}).")

    # Build the list of pages to render, then split across workers
    pages_to_render = list(range(start, end + 1))
    page_chunks     = _chunk_pages(pages_to_render, min(args.jobs, len(pages_to_render)))

    print(f"Rendering pages {start}–{end} of {total_pages} at {args.dpi} DPI "
          f"to {fmt.upper()} using {len(page_chunks)} processes…")

    with ProcessPoolExecutor(max_workers=len(page_chunks)) as pool:
        futures = [
            pool.submit(
                _render_pages,
                args.input_pdf,
                outdir.as_posix(),
                chunk,
                args.dpi,
                fmt,
            )
            for chunk in page_chunks
        ]
        for fut in as_completed(futures):           # optional: progress / error handling
            fut.result()  # re‑raise any worker exceptions

    print("Done ✔")

if __name__ == "__main__":
    main()
