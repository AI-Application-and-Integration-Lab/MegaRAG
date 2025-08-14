import json
import time
import yaml
import torch
import logging
import asyncio
import argparse
from pathlib import Path

from transformers import AutoModel

from megarag import MegaRAG
from megarag.llms.hf import (
    hf_gme_embed,
)
from megarag.llms.openai import (
    gpt_4o_mini_complete,
) 
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import TokenTracker

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
)

def initialize_model():
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    model_name  = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    embed_model = (
        AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else None,
            trust_remote_code=True
        ).to(device).eval()
    )
    return embed_model

def load_addon_params(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Accept either layout: top‑level or under "addon_params"
    return data.get("addon_params", data)

async def initialize_rag(working_dir: Path, addon_params: dict) -> MegaRAG:
    """Initialise a :class:`MegaRAG` instance for *working_dir* using *addon_params*."""
    embed_model = initialize_model()
    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=32768)
    async def embed_func(texts=[], images=[], is_query=False):
        results = await hf_gme_embed(
            embed_model=embed_model, 
            texts=texts, 
            images=images, 
            is_query=is_query
        )
        return results
    token_tracker = TokenTracker()
    async def llm_func(
        prompt,
        input_images=None,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ):
        results = await gpt_4o_mini_complete(
            prompt=prompt,
            input_images=input_images,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            token_tracker=token_tracker,
            **kwargs,
        )
        return results
    rag = MegaRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        embedding_func=embed_func,
        addon_params=addon_params,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag, token_tracker

def insert_context(
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
            rag.insert(
                input=unique_contexts,
                file_paths=str(file_path),
            )
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries:
                msg = "Insertion failed after exceeding the maximum number of retries"
                raise RuntimeError(msg) from exc

            print(
                f"Insertion failed, retrying ({attempt}/{max_retries}), error: {exc}",
            )
            time.sleep(retry_delay)

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
        default="./dumps",
        metavar="DIR",
        help="Directory where the mineru processed file resides.",
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

def main() -> None:
    args = parse_args()

    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # Load addon_params from YAML
    addon_params = load_addon_params(Path(args.config_file))

    # Initialise RAG with the loaded parameters
    rag, token_tracker = asyncio.run(initialize_rag(working_dir, addon_params))

    input_dir = Path(args.input_dir)
    insert_context(
        rag,
        input_dir,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )
    print(f'Final Token Usage: {token_tracker}.')

if __name__ == "__main__":
    main()
