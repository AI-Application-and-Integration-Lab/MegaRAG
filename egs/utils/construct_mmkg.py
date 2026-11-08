import asyncio
import argparse
from pathlib import Path

from megarag import MegaRAG
from megarag.config import load_config
from megarag.runtime import initialize_rag

async def insert_context(
    rag: MegaRAG,
    file_path: Path,
    *,
    max_retries: int = 3,
    retry_delay: int = 10,
) -> None:
    """Insert the JSON contexts into the RAG store, retrying on failure."""
    with file_path.open("r", encoding="utf-8") as f:
        unique_contexts = f.read()

    for attempt in range(1, max_retries + 1):
        try:
            await rag.ainsert(
                input=unique_contexts,
                file_paths=str(file_path),
            )
            from lightrag.base import DocStatus
            if await rag.doc_status.get_docs_by_status(DocStatus.FAILED):
                raise RuntimeError("Document processing failed; inspect the preceding error.")
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                msg = "Insertion failed after exceeding the maximum number of retries"
                raise RuntimeError(msg) from exc

            print(
                f"Insertion failed, retrying ({attempt}/{max_retries}), error: {exc}",
            )
            await asyncio.sleep(retry_delay)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert unique context JSON into MegaRAG storage, using a YAML‑based addon_params config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        default="addon_params.yaml",
        metavar="FILE",
        help="YAML file containing addon_params configuration.",
    )
    parser.add_argument(
        "--input-dir",
        default="./pages_content.json",
        metavar="FILE",
        help="Path to the prepared pages_content.json file.",
    )
    parser.add_argument(
        "--working-dir",
        default="./exp",
        metavar="DIR",
        help="Working directory for MegaRAG.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of insertion retries on failure.",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=10,
        metavar="SECONDS",
        help="Delay between retries when insertion fails.",
    )
    return parser.parse_args()

async def run(args) -> None:

    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Load addon_params from YAML
    config = load_config(Path(args.config_file))

    # Initialise RAG with the loaded parameters
    rag, token_tracker = await initialize_rag(working_dir, config=config, ingest=True)

    try:
        await insert_context(
            rag, Path(args.input_dir),
            max_retries=args.max_retries, retry_delay=args.retry_delay,
        )
        print(f'Final Token Usage: {token_tracker}.')
    finally:
        await rag.finalize_storages()


def main() -> None:
    args = parse_args()
    if args.max_retries < 1 or args.retry_delay < 0:
        raise ValueError("max-retries must be positive and retry-delay nonnegative")
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
