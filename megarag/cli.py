"""Small command-line entry point for the complete PDF workflow."""

import argparse
import asyncio
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

import yaml

from .config import Config, load_config, prepare_workspace
from .inputs import create_demo, extract_queries, prepare_pdf, read_page_manifest, resolve_mineru_python


def parser():
    root = argparse.ArgumentParser(
        prog="megarag",
        description="Build a multimodal graph from PDFs and ask questions.",
    )
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Write an editable configuration")
    init.add_argument("--output", type=Path, default=Path("megarag.yaml"))
    demo = commands.add_parser(
        "demo", help="Create a two-page PDF and example questions"
    )
    demo.add_argument("--output", type=Path, default=Path("demo"))
    for name, description in (
        ("doctor", "Check local dependencies and configuration; no API calls"),
        ("ingest", "Parse a PDF with MinerU and build an index, or use prepared page JSON"),
        ("query", "Ask a question or run a question file"),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument(
            "--config",
            type=Path,
            help="YAML configuration (uses defaults when omitted)",
        )
        command.add_argument("--workspace", type=Path, default=Path("workspace"))
        if name == "ingest":
            command.add_argument("input", type=Path, help="PDF or pages_content.json")
            command.add_argument(
                "--max-pages", type=int, help="Limit PDF pages for a first run"
            )
        elif name == "query":
            command.add_argument("question", nargs="?")
            command.add_argument(
                "--questions", type=Path, help="TXT (one question per line) or JSONL"
            )
            command.add_argument(
                "--output",
                type=Path,
                help="Save each completed answer as JSONL (new file)",
            )
    return root


def doctor(config):
    missing = []
    for module in (
        "lightrag",
        "nano_vectordb",
        "networkx",
        "openai",
        "torch",
        "transformers",
        "accelerate",
        "fitz",
    ):
        exists = importlib.util.find_spec(module) is not None
        print(f"{'OK' if exists else 'MISSING'}  {module}")
        if not exists:
            missing.append(module)
    has_key = not config.llm.api_key_env or bool(os.environ.get(config.llm.api_key_env))
    auth_label = config.llm.api_key_env or "API authentication disabled in configuration"
    print(f"{'OK' if has_key else 'MISSING'}  {auth_label}")
    if missing:
        print('Install the inference extras: pip install -e ".[local,pdf]"')
    try:
        print(f"OK  MinerU Python: {resolve_mineru_python()}")
    except ValueError as exc:
        print(f"MISSING  {exc}")
        missing.append("MinerU Python")
    print(f"LLM: {config.llm.model}")
    print(
        f"Embedding: {config.embedding.model} ({config.embedding.dimension} dimensions)"
    )
    print(
        "This checks local setup only; endpoint access and model inference are not tested."
    )
    return 0 if has_key and not missing else 1


async def run(args, config):
    from .runtime import initialize_rag

    if args.command == "ingest":
        if not args.input.is_file():
            raise ValueError(f"Input file not found: {args.input}")
        suffix = args.input.suffix.lower()
        if suffix not in (".pdf", ".json"):
            raise ValueError("Input must be a PDF or prepared page JSON")
        # Validate the workspace before writing any generated PDF assets.
        prepare_workspace(args.workspace, config, ingest=True)
        manifest = (
            prepare_pdf(args.input, args.workspace / "parsed", max_pages=args.max_pages)
            if suffix == ".pdf"
            else args.input.resolve()
        )
        pages = read_page_manifest(manifest)
        content = json.dumps(pages, ensure_ascii=False)
        rag, tracker = await initialize_rag(args.workspace, config=config, ingest=True)
        try:
            print(f"Building graph from {len(pages)} pages…", flush=True)
            await rag.ainsert(content, file_paths=str(args.input.resolve()))
            from lightrag.base import DocStatus

            failed = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
            if failed:
                raise RuntimeError(
                    "Document processing failed. Check the preceding error and rerun ingest."
                )
            print(f"Index saved to {args.workspace.resolve()}")
            print(tracker)
        finally:
            await rag.finalize_storages()
    else:
        if bool(args.question) == bool(args.questions):
            raise ValueError("Provide either a question or --questions FILE")
        questions = (
            extract_queries(args.questions)
            if args.questions
            else [args.question.strip()]
        )
        if not questions or any(not question for question in questions):
            raise ValueError("No questions found")
        if args.output and args.output.exists():
            raise ValueError("Output file already exists. Choose a new --output path.")
        rag, tracker = await initialize_rag(args.workspace, config=config)
        output = None
        try:
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                output = args.output.open("x", encoding="utf-8")
            from lightrag.base import QueryParam

            for question in questions:
                answer = await rag.aquery(
                    question,
                    param=QueryParam(
                        mode="mix_two_step",
                        chunk_top_k=config.addon_params["chunk_top_k"],
                        enable_rerank=False,
                    ),
                )
                print(f"\n{question}\n{answer}")
                if output:
                    output.write(
                        json.dumps(
                            {"question": question, "answer": answer}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    output.flush()
            print(tracker)
        finally:
            if output:
                output.close()
            await rag.finalize_storages()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            with args.output.open("x", encoding="utf-8") as output:
                yaml.safe_dump(asdict(Config()), output, sort_keys=False)
            print(f"Configuration written to {args.output}")
            return 0
        if args.command == "demo":
            print(create_demo(args.output))
            return 0
        config = load_config(args.config)
        if args.command == "doctor":
            return doctor(config)
        config.llm.resolve_api_key()
        asyncio.run(run(args, config))
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
