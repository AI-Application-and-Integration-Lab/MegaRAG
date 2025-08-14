import re
import json
import time
import yaml
import sys
import torch
import pathlib
import asyncio
import argparse

from pathlib import Path
from typing import List, Tuple, Any, Dict
from datetime import datetime, timezone

from transformers import AutoModel

from megarag import MegaRAG
from megarag.llms.hf import hf_gme_embed
from megarag.llms.openai import gpt_4o_mini_complete

from lightrag.base import QueryParam
from lightrag.utils import TokenTracker, wrap_embedding_func_with_attrs
from lightrag.kg.shared_storage import initialize_pipeline_status

# --- progress bar (optional) ---
try:
    import tqdm.auto as tqdmauto  # tqdmauto.tqdm + tqdmauto.tqdm.write
except Exception:
    tqdmauto = None

QUERY_RE = re.compile(r"-\s*Question\s+\d+:\s+(.+)")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _torch_speed_tweaks():
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

def initialize_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name  = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    embed_model = (
        AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        .to(device, non_blocking=True)
        .eval()
    )
    return embed_model

def load_addon_params(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("addon_params", data)

async def initialize_rag(working_dir: Path, addon_params: dict) -> Tuple[MegaRAG, TokenTracker]:
    """Initialise a MegaRAG instance for *working_dir* using *addon_params*."""
    embed_model = initialize_model()

    # Guard the GPU embed model; tune if your GPU can handle more parallel fwd passes.
    embed_parallel_limit = addon_params.get("embed_parallel_limit", 1)
    _embed_sem = asyncio.Semaphore(embed_parallel_limit)

    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=32768)
    async def embed_func(texts=[], images=[], is_query=False):
        async with _embed_sem:
            return await hf_gme_embed(
                embed_model=embed_model,
                texts=texts,
                images=images,
                is_query=is_query,
            )

    token_tracker = TokenTracker()

    async def llm_func(
        prompt,
        input_images=None,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ):
        return await gpt_4o_mini_complete(
            prompt=prompt,
            input_images=input_images,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            token_tracker=token_tracker,
            **kwargs,
        )

    rag = MegaRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        embedding_func=embed_func,
        addon_params=addon_params,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag, token_tracker

def extract_queries(file_path: pathlib.Path) -> List[str]:
    """
    Read queries from either:
      - JSONL: lines with {"question": "..."}  (preferred)
      - Markdown-like lines: "- Question N: text"
    """
    if file_path.suffix.lower() == ".jsonl":
        out = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("question") or obj.get("text")
                    if q:
                        out.append(str(q))
                except json.JSONDecodeError:
                    m = QUERY_RE.search(line.replace("**", ""))
                    if m:
                        out.append(m.group(1).strip())
        return out
    # default markdown/flat text mode
    text = file_path.read_text(encoding="utf-8").replace("**", "")
    return [q.strip() for q in QUERY_RE.findall(text)]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run concurrent RAG queries and save results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-file", default="addon_params.yaml", metavar="FILE")
    parser.add_argument("--input-queries", default="./queries.jsonl", metavar="FILE")
    parser.add_argument("--working-dir", default="./exp", metavar="DIR")
    parser.add_argument("--max-retries", type=int, default=3, metavar="N")
    parser.add_argument("--retry-delay", type=float, default=3.0, metavar="SECONDS")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        metavar="N",
        help="How many queries to run at once (tune to your GPU/API limits).",
    )
    parser.add_argument(
        "--print-as-complete",
        action="store_true",
        help="Print results as each query finishes instead of preserving input order.",
    )
    parser.add_argument(
        "--output-file",
        default="results.jsonl",
        metavar="FILE",
        help="Where to save outputs (default JSONL).",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "json"],
        default="json",
        help="jsonl = one JSON object per line; json = single array with metadata.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file (only for jsonl).",
    )
    # --- progress bar switches ---
    parser.add_argument("--progress", dest="progress", action="store_true",
                        help="Show a progress bar (default if supported).")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Disable the progress bar.")
    parser.set_defaults(progress=True)
    return parser.parse_args()

async def run_query_with_retries(rag: MegaRAG, question: str, param: QueryParam, max_retries: int, retry_delay: float):
    attempt = 0
    while True:
        attempt += 1
        try:
            t0 = time.perf_counter()
            res = await rag.aquery(question, param=param)
            dt = time.perf_counter() - t0
            return res, dt, None
        except Exception as e:
            if attempt >= max_retries:
                return None, None, e
            # Exponential backoff with a tiny jitter
            jitter = min(0.5, retry_delay * 0.1)
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)) + jitter)

async def async_main():
    _torch_speed_tweaks()
    args = parse_args()

    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    addon_params = load_addon_params(Path(args.config_file))
    rag, token_tracker = await initialize_rag(working_dir, addon_params)

    param = QueryParam(
        mode="mix_two_step",
        chunk_top_k=addon_params.get("chunk_top_k", 6),
        enable_rerank=False,
    )

    questions = extract_queries(Path(args.input_queries))
    if not questions:
        raise SystemExit(f"No questions found in {args.input_queries}")

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def worker(idx: int, q: str):
        started_at = _now_iso()
        async with sem:
            res, dt, err = await run_query_with_retries(
                rag, q, param, max_retries=args.max_retries, retry_delay=args.retry_delay
            )
        finished_at = _now_iso()
        record: Dict[str, Any] = {
            "index": idx,
            "question": q,
            "answer": res,          # may be complex; handled by default=str on dump
            "latency_seconds": dt,
            "error": str(err) if err else None,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        return idx, record

    tasks = [asyncio.create_task(worker(i, q)) for i, q in enumerate(questions)]

    # --- progress bar setup ---
    use_bar = (
        args.progress
        and (tqdmauto is not None)
        and (hasattr(sys.stderr, "isatty") and sys.stderr.isatty())
    )
    pbar = tqdmauto.tqdm(total=len(tasks), desc="Processing queries", unit="q") if use_bar else None

    # Collect results as tasks finish so we can update the bar
    finished: List[Any] = [None] * len(tasks)
    err_count = 0

    for fut in asyncio.as_completed(tasks):
        idx, rec = await fut
        finished[idx] = rec
        if pbar:
            if rec["error"]:
                err_count += 1
                pbar.set_postfix_str(f"errors={err_count}")
            pbar.update(1)

        # Optional streaming output without breaking the bar
        if args.print_as_complete:
            line = (
                f"[{idx}] ERROR for question: {rec['question']}\n  {rec['error']}\n"
                if rec["error"]
                else f"[{idx}] ({rec['latency_seconds']:.2f}s) {rec['answer']}\n"
            )
            if pbar:
                tqdmauto.tqdm.write(line)
            else:
                print(line)

    if pbar:
        pbar.close()

    # If not streaming earlier, print in input order now
    if not args.print_as_complete:
        for idx, rec in enumerate(finished):
            if rec["error"]:
                print(f"[{idx}] ERROR for question: {rec['question']}\n  {rec['error']}\n")
            else:
                print(f"[{idx}] ({rec['latency_seconds']:.2f}s) {rec['answer']}\n")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "jsonl":
        mode = "a" if args.append else "w"
        with out_path.open(mode, encoding="utf-8") as f:
            for rec in finished:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"Saved {len(finished)} records to {out_path} (JSONL).")
    else:
        payload = {
            "results": finished,
            "metadata": {
                "final_token_usage": str(token_tracker),
                "query_file": str(args.input_queries),
                "generated_at": _now_iso(),
            },
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved results + metadata to {out_path} (JSON array).")

    print(f"Final Token Usage: {token_tracker}.")

def main():
    try:
        import uvloop  # optional but often faster on Linux/macOS
        uvloop.install()
    except Exception:
        pass
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
