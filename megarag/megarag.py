import os
import asyncio
import traceback
import time, json, uuid, contextvars  # timing + correlation
import atexit, threading
from pathlib import Path
from contextlib import ExitStack

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    cast,
    final,
    Literal,
    Optional,
    List,
    Dict,
)
from functools import partial

from dataclasses import (
    asdict,
    dataclass,
    field
)

from datetime import (
    datetime,
    timezone
)

from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    get_graph_db_lock,
)

from lightrag import LightRAG
from lightrag.utils import (
    Tokenizer,
    always_get_an_event_loop,
    logger,
    clean_text,
    compute_mdhash_id,
    get_content_summary,
    lazy_external_import,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
    DeletionResult,
)

from megarag.operate import (
    chunking_by_token_or_page,
    extract_entities,
    extract_entities_refinement,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    kg_two_step_query,
)

from megarag.utils import plot_waterfall_from_jsonl

from megarag.kg import STORAGES

trace_id_var = contextvars.ContextVar("trace_id", default="-")
_span_stack_var = contextvars.ContextVar("span_stack", default=())  # for parent/child spans

_TIMING_JSONL_FILE = None     # type: Optional[Any]
_TIMING_JSONL_LOCK = threading.Lock()

def enable_timing_jsonl(path: str = "logs/timings.jsonl"):
    """Enable JSONL persistence. Call once at startup (we auto-call below)."""
    global _TIMING_JSONL_FILE
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    _TIMING_JSONL_FILE = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
    atexit.register(_TIMING_JSONL_FILE.close)
    return _TIMING_JSONL_FILE

_StageListener = Callable[[dict], None]
_STAGE_LISTENERS: List[_StageListener] = []
_STAGE_LISTENERS_LOCK = threading.Lock()

def add_stage_listener(listener: _StageListener) -> None:
    with _STAGE_LISTENERS_LOCK:
        _STAGE_LISTENERS.append(listener)

def remove_stage_listener(listener: _StageListener) -> None:
    with _STAGE_LISTENERS_LOCK:
        try:
            _STAGE_LISTENERS.remove(listener)
        except ValueError:
            pass

def _notify_stage_listeners(event: dict) -> None:
    # Best-effort, never break the pipeline if UI fails
    with _STAGE_LISTENERS_LOCK:
        listeners = tuple(_STAGE_LISTENERS)
    for fn in listeners:
        try:
            fn(event)
        except Exception:
            pass

class StageTimer:
    """Async context manager for precise wall-clock timing with span hierarchy.
    Emits both: (1) flattened JSON via logger, and (2) full record to JSONL file (if enabled).
    Also notifies listeners so a Rich progress can mirror the stages.
    """
    def __init__(self, name: str, tags: dict | None = None):
        self.name = name
        self.tags = tags or {}
        self.t0_ns = 0
        self.span_id = uuid.uuid4().hex[:12]
        self.parent_span_id: Optional[str] = None

    async def __aenter__(self):
        stack = _span_stack_var.get()
        self.parent_span_id = stack[-1] if stack else None
        _span_stack_var.set(stack + (self.span_id,))
        self.t0_ns = time.perf_counter_ns()

        _notify_stage_listeners({
            "event": "enter",
            "ts": time.time(),
            "trace_id": trace_id_var.get(),
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "stage": self.name,
            "tags": self.tags,
        })
        return self

    async def __aexit__(self, exc_type, exc, tb):
        t1_ns = time.perf_counter_ns()
        dt_ms = (t1_ns - self.t0_ns) / 1e6
        # Full record (kept rich for JSONL)
        record = {
            "type": "stage_timing",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),  # human UTC
            "trace_id": trace_id_var.get(),
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "stage": self.name,
            "elapsed_ms": dt_ms,
            "ok": exc is None,
            "error": repr(exc) if exc else None,
            "start_ns": self.t0_ns,
            "end_ns": t1_ns,
            "tags": self.tags,
        }
        # Flattened view for your existing logs
        flat = {k: v for k, v in record.items() if k != "tags"} | self.tags
        logger.info(json.dumps(flat, ensure_ascii=False))

        # Persist JSONL if enabled
        if _TIMING_JSONL_FILE:
            with _TIMING_JSONL_LOCK:
                _TIMING_JSONL_FILE.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Progress/listeners
        _notify_stage_listeners({
            "event": "exit",
            "ts": time.time(),
            "trace_id": trace_id_var.get(),
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "stage": self.name,
            "elapsed_ms": dt_ms,
            "ok": exc is None,
            "error": repr(exc) if exc else None,
            "tags": self.tags,
        })

        # Pop span from stack
        stack = _span_stack_var.get()
        if stack and stack[-1] == self.span_id:
            _span_stack_var.set(stack[:-1])
        else:
            _span_stack_var.set(tuple(s for s in stack if s != self.span_id))

def stage(name: str, **tags) -> StageTimer:
    """Helper to create a StageTimer with optional key/value tags."""
    return StageTimer(name, tags)

def timed_coro(name: str, coro_fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Wrap a coroutine with timing so it can be scheduled via create_task()."""
    async def _runner():
        async with stage(name, **kwargs.get("_tags", {})):
            real_kwargs = {k: v for k, v in kwargs.items() if k != "_tags"}
            return await coro_fn(*args, **real_kwargs)
    return _runner()

class StageProgress:
    """Drives a Rich Progress from StageTimer enter/exit events."""
    def __init__(self):
        self.progress = None
        self._span2task: Dict[str, int] = {}
        self._span2depth: Dict[str, int] = {}
        self._task_desc: Dict[int, str] = {}
        self._lock = threading.Lock()

        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
            from rich.console import Console
            if os.getenv("RAG_PROGRESS", "1") not in ("0", "false", "False"):
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    transient=True,
                    console=Console(stderr=True),  # keep stdout clean if you log JSON there
                )
        except Exception:
            self.progress = None

    def __enter__(self):
        if self.progress:
            self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.progress:
            try:
                self.progress.stop()
            finally:
                self.progress.__exit__(exc_type, exc, tb)

    def handle(self, ev: dict) -> None:
        if not self.progress:
            return
        with self._lock:
            kind = ev.get("event")
            if kind == "enter":
                parent = ev.get("parent_span_id")
                depth = (self._span2depth.get(parent, -1) + 1) if parent else 0
                self._span2depth[ev["span_id"]] = depth
                indent = "  " * depth
                desc = f"{indent}{ev['stage']}"
                task_id = self.progress.add_task(desc, total=1)
                self._span2task[ev["span_id"]] = task_id
                self._task_desc[task_id] = desc

            elif kind == "exit":
                span_id = ev["span_id"]
                task_id = self._span2task.pop(span_id, None)
                if task_id is not None:
                    base = self._task_desc.get(task_id, "")
                    ok = ev.get("ok", True)
                    ms = ev.get("elapsed_ms")
                    mark = "✓" if ok else "✗"
                    if ms is not None:
                        base = f"{base}  {mark}  ({ms:.0f} ms)"
                    else:
                        base = f"{base}  {mark}"
                    # finish the line
                    self.progress.update(task_id, description=base, completed=1, advance=1)

class MegaRAG(LightRAG):
    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def __post_init__(self):
        self.vector_storage = "NanoMMVectorDBStorage"
        super().__post_init__()
        self.chunking_func  = chunking_by_token_or_page

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        split_by_page: bool = True,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Sync Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: single string of the file path or list of file paths, used for citation
        """
        # Auto-enable JSONL sink (override via env RAG_TIMING_JSONL)
        tim_log_dir = os.path.join(self.working_dir, self.workspace, "logs/timings.jsonl")
        enable_timing_jsonl(os.getenv("RAG_TIMING_JSONL", tim_log_dir))

        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(
                input, split_by_character, split_by_character_only, split_by_page, ids, file_paths
            )
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        split_by_page: bool = True,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Async Insert documents with checkpoint support

        Args:
            input: Single document string or list of document strings
            split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
            chunk_token_size, it will be split again by token size.
            split_by_character_only: if split_by_character_only is True, split the string by character only, when
            split_by_character is None, this parameter is ignored.
            ids: list of unique document IDs, if not provided, MD5 hash IDs will be generated
            file_paths: list of file paths corresponding to each document, used for citation
        """
        await self.apipeline_enqueue_documents(input, ids, file_paths)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only, split_by_page
        )

    async def _process_entity_relation_graph(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        try:
            # Time the whole extraction over all chunks
            chunk_results = await extract_entities(
                chunk,
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
                text_chunks_storage=self.text_chunks,
            )    
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
            raise e

    async def _process_entity_relation_graph_refinement(
        self,
        chunk: dict[str, Any],
        chunk_results: list,
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        global_config: dict[str, str],
        llm_response_cache: BaseKVStorage | None = None,
        pipeline_status=None,
        pipeline_status_lock=None
    ) -> list:
        try:
            # Time the refinement extraction over all chunks
            async with stage("extract_entities_refine", chunks=len(chunk) if hasattr(chunk, "__len__") else None):
                chunk_results = await extract_entities_refinement(
                    chunk,
                    chunk_results,
                    knowledge_graph_inst,
                    entity_vdb,
                    relationships_vdb,
                    global_config=global_config,
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                    llm_response_cache=llm_response_cache,
                    text_chunks_storage=self.text_chunks,
                )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
            raise e

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        split_by_page: bool = True,
    ) -> None:
        """
        Process pending documents by splitting them into chunks, processing
        each chunk for entity and relation extraction, and updating the
        document status.
        """

        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            if not pipeline_status.get("busy", False):
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                pipeline_status.update(
                    {
                        "busy": True,
                        "job_name": "Default Job",
                        "job_start": datetime.now(timezone.utc).isoformat(),
                        "docs": 0,
                        "batchs": 0,  # Total number of files to be processed
                        "cur_batch": 0,  # Number of files already processed
                        "request_pending": False,  # Clear any previous request
                        "latest_message": "",
                    }
                )
                # Clean history_messages without breaking its shared list reference
                del pipeline_status["history_messages"][:]
            else:
                # Another process is busy: set the request flag and return
                pipeline_status["request_pending"] = True
                logger.info(
                    "Another process is already processing the document queue. Request queued."
                )
                return

        # Create a trace_id for this pipeline run to tie all stage logs together
        trace_id_var.set(uuid.uuid4().hex)

        # ---- Stage-driven progress UI --------------------------------------
        sp = StageProgress()
        with ExitStack() as stack:
            stack.enter_context(sp)         # show progress UI (if available)
            add_stage_listener(sp.handle)   # subscribe to StageTimer events
            try:
                # Total wall-clock timing for the whole pipeline (covers the while-loop)
                async with stage("pipeline_total", job_name=pipeline_status.get("job_name", "Default Job")):
                    # Process documents until no more documents or requests
                    while True:
                        if not to_process_docs:
                            log_message = "All documents have been processed or are duplicates"
                            logger.info(log_message)
                            pipeline_status["latest_message"] = log_message
                            pipeline_status["history_messages"].append(log_message)
                            break

                        log_message = f"Processing {len(to_process_docs)} document(s)"
                        logger.info(log_message)

                        # Update pipeline_status: batchs now represents the total number of files to be processed
                        pipeline_status["docs"] = len(to_process_docs)
                        pipeline_status["batchs"] = len(to_process_docs)
                        pipeline_status["cur_batch"] = 0
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                        # Get first document's file path and total count for job name
                        first_doc_id, first_doc = next(iter(to_process_docs.items()))
                        first_doc_path = first_doc.file_path

                        # Handle cases where first_doc_path is None
                        if first_doc_path:
                            path_prefix = first_doc_path[:20] + (
                                "..." if len(first_doc_path) > 20 else ""
                            )
                        else:
                            path_prefix = "unknown_source"

                        total_files = len(to_process_docs)
                        job_name = f"{path_prefix}[{total_files} files]"
                        pipeline_status["job_name"] = job_name

                        # Counter for processed files
                        processed_count = 0
                        # Semaphore to limit concurrent file processing
                        semaphore = asyncio.Semaphore(self.max_parallel_insert)

                        async def process_document(
                            doc_id: str,
                            status_doc: DocProcessingStatus,
                            split_by_character: str | None,
                            split_by_character_only: bool,
                            split_by_page: bool,
                            pipeline_status: dict,
                            pipeline_status_lock: asyncio.Lock,
                            semaphore: asyncio.Semaphore,
                        ) -> None:
                            """Process a single document end-to-end."""
                            file_extraction_stage_ok = False
                            async with semaphore:
                                nonlocal processed_count
                                current_file_number = 0

                                # Get file_path early so it's available in top-level timing tags
                                file_path = getattr(status_doc, "file_path", "unknown_source")

                                # Per-file total timing
                                async with stage("file_total", doc_id=doc_id, file_path=file_path):
                                    first_stage_tasks = []
                                    entity_relation_task = None
                                    try:
                                        async with pipeline_status_lock:
                                            # Update processed file count and save current ordinal
                                            processed_count += 1
                                            current_file_number = processed_count
                                            pipeline_status["cur_batch"] = processed_count

                                            log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                                            logger.info(log_message)
                                            pipeline_status["history_messages"].append(log_message)
                                            log_message = f"Processing d-id: {doc_id}"
                                            logger.info(log_message)
                                            pipeline_status["latest_message"] = log_message
                                            pipeline_status["history_messages"].append(log_message)

                                        # Generate chunks from the document
                                        async with stage(
                                            "chunking",
                                            doc_id=doc_id,
                                            split_by_character=str(split_by_character),
                                            split_by_page=split_by_page
                                        ):
                                            chunk_args = {
                                                "tokenizer": self.tokenizer,
                                                "content": status_doc.content,
                                                "split_by_character": split_by_character,
                                                "split_by_character_only": split_by_character_only,
                                                "split_by_page": split_by_page,
                                                "overlap_token_size": self.chunk_overlap_token_size,
                                                "max_token_size": self.chunk_token_size,
                                            }

                                            chunks: dict[str, Any] = {
                                                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                                    **dp,
                                                    "full_doc_id": doc_id,
                                                    "file_path": file_path,  # Attach the originating path to each chunk
                                                    "llm_cache_list": [],    # Initialize per-chunk LLM cache list
                                                }
                                                for dp in self.chunking_func(**chunk_args)
                                            }

                                        if not chunks:
                                            logger.warning("No document chunks to process")

                                        # Stage 1: Upsert text chunks and docs (in parallel)
                                        doc_status_task = asyncio.create_task(
                                            timed_coro(
                                                "upsert.doc_status",
                                                self.doc_status.upsert,
                                                {
                                                    doc_id: {
                                                        "status": DocStatus.PROCESSING,
                                                        "chunks_count": len(chunks),
                                                        "chunks_list": list(chunks.keys()),  # Keep the list of chunk ids
                                                        "content": status_doc.content,
                                                        "content_summary": status_doc.content_summary,
                                                        "content_length": status_doc.content_length,
                                                        "created_at": status_doc.created_at,
                                                        "updated_at": datetime.now(timezone.utc).isoformat(),
                                                        "file_path": file_path,
                                                    }
                                                },
                                                _tags={"doc_id": doc_id}
                                            )
                                        )
                                        chunks_vdb_task = asyncio.create_task(
                                            timed_coro("upsert.chunks_vdb", self.chunks_vdb.upsert, chunks, _tags={"doc_id": doc_id})
                                        )
                                        full_docs_task = asyncio.create_task(
                                            timed_coro(
                                                "upsert.full_docs",
                                                self.full_docs.upsert,
                                                {doc_id: {"content": status_doc.content}},
                                                _tags={"doc_id": doc_id}
                                            )
                                        )
                                        text_chunks_task = asyncio.create_task(
                                            timed_coro("upsert.text_chunks", self.text_chunks.upsert, chunks, _tags={"doc_id": doc_id})
                                        )

                                        # Track first-stage tasks (parallel execution)
                                        first_stage_tasks = [
                                            doc_status_task,
                                            chunks_vdb_task,
                                            full_docs_task,
                                            text_chunks_task,
                                        ]

                                        # Await the first stage as a group (wall-clock; not the sum of subtasks)
                                        async with stage("stage1_upserts_gather", doc_id=doc_id):
                                            await asyncio.gather(*first_stage_tasks)

                                        # Stage 2: Entity/relationship extraction (after text_chunks are saved)
                                        async with stage("extract_entities_all_chunks", doc_id=doc_id, chunks=len(chunks)):
                                            entity_relation_task = asyncio.create_task(
                                                self._process_entity_relation_graph(
                                                    chunks, pipeline_status, pipeline_status_lock
                                                )
                                            )
                                            await entity_relation_task
                                        file_extraction_stage_ok = True

                                    except Exception as e:
                                        # Log error and update pipeline status
                                        logger.error(traceback.format_exc())
                                        error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                                        logger.error(error_msg)
                                        async with pipeline_status_lock:
                                            pipeline_status["latest_message"] = error_msg
                                            pipeline_status["history_messages"].append(
                                                traceback.format_exc()
                                            )
                                            pipeline_status["history_messages"].append(error_msg)

                                            # Cancel tasks that are not completed yet
                                            all_tasks = first_stage_tasks + (
                                                [entity_relation_task] if entity_relation_task else []
                                            )
                                            for task in all_tasks:
                                                if task and not task.done():
                                                    task.cancel()

                                        # Persist LLM cache if available
                                        if self.llm_response_cache:
                                            await self.llm_response_cache.index_done_callback()

                                        # Mark the document as failed
                                        await self.doc_status.upsert(
                                            {
                                                doc_id: {
                                                    "status": DocStatus.FAILED,
                                                    "error": str(e),
                                                    "content": status_doc.content,
                                                    "content_summary": status_doc.content_summary,
                                                    "content_length": status_doc.content_length,
                                                    "created_at": status_doc.created_at,
                                                    "updated_at": datetime.now(
                                                        timezone.utc
                                                    ).isoformat(),
                                                    "file_path": file_path,
                                                }
                                            }
                                        )

                                    # After extraction succeeds, perform merge and optional refinement
                                    if file_extraction_stage_ok:
                                        try:
                                            # Retrieve results from the extraction task
                                            chunk_results = await entity_relation_task

                                            # Merge nodes/edges into graph and vector stores
                                            async with stage("merge_nodes_and_edges", doc_id=doc_id):
                                                await merge_nodes_and_edges(
                                                    chunk_results=chunk_results,
                                                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                                                    entity_vdb=self.entities_vdb,
                                                    relationships_vdb=self.relationships_vdb,
                                                    global_config=asdict(self),
                                                    pipeline_status=pipeline_status,
                                                    pipeline_status_lock=pipeline_status_lock,
                                                    llm_response_cache=self.llm_response_cache,
                                                    current_file_number=current_file_number,
                                                    total_files=total_files,
                                                    file_path=file_path,
                                                )

                                            # Optional: iterative refinement rounds
                                            for r in range(self.addon_params['entity_refine_max_times']):
                                                await self._insert_refine_done(refine_round=r)

                                                async with stage(f"refine_round_{r}", round=r, doc_id=doc_id):
                                                    chunk_results = await process_document_refinement(
                                                        doc_id=doc_id,
                                                        status_doc=status_doc,
                                                        chunks=chunks,
                                                        chunk_results=chunk_results,
                                                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                                                        entity_vdb=self.entities_vdb,
                                                        relationships_vdb=self.relationships_vdb,
                                                        global_config=asdict(self),
                                                        pipeline_status=pipeline_status,
                                                        pipeline_status_lock=pipeline_status_lock,
                                                        llm_response_cache=self.llm_response_cache,
                                                        current_file_number=current_file_number,
                                                        total_files=total_files,
                                                        file_path=file_path,
                                                    )

                                            # Final status update for the document
                                            async with stage("finalize_doc_status", doc_id=doc_id):
                                                await self.doc_status.upsert(
                                                    {
                                                        doc_id: {
                                                            "status": DocStatus.PROCESSED,
                                                            "chunks_count": len(chunks),
                                                            "chunks_list": list(
                                                                chunks.keys()
                                                            ),  # Keep chunks_list for debugging/traceability
                                                            "content": status_doc.content,
                                                            "content_summary": status_doc.content_summary,
                                                            "content_length": status_doc.content_length,
                                                            "created_at": status_doc.created_at,
                                                            "updated_at": datetime.now(
                                                                timezone.utc
                                                            ).isoformat(),
                                                            "file_path": file_path,
                                                        }
                                                    }
                                                )

                                            # Fire post-insert hook
                                            async with stage("_insert_done", doc_id=doc_id):
                                                await self._insert_done()

                                            async with pipeline_status_lock:
                                                log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                                                logger.info(log_message)
                                                pipeline_status["latest_message"] = log_message
                                                pipeline_status["history_messages"].append(
                                                    log_message
                                                )

                                        except Exception as e:
                                            # Log error and update pipeline status
                                            logger.error(traceback.format_exc())
                                            error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                                            logger.error(error_msg)
                                            async with pipeline_status_lock:
                                                pipeline_status["latest_message"] = error_msg
                                                pipeline_status["history_messages"].append(
                                                    traceback.format_exc()
                                                )
                                                pipeline_status["history_messages"].append(
                                                    error_msg
                                                )

                                            # Persist LLM cache if available
                                            if self.llm_response_cache:
                                                await self.llm_response_cache.index_done_callback()

                                            # Mark the document as failed
                                            await self.doc_status.upsert(
                                                {
                                                    doc_id: {
                                                        "status": DocStatus.FAILED,
                                                        "error": str(e),
                                                        "content": status_doc.content,
                                                        "content_summary": status_doc.content_summary,
                                                        "content_length": status_doc.content_length,
                                                        "created_at": status_doc.created_at,
                                                        "updated_at": datetime.now().isoformat(),
                                                        "file_path": file_path,
                                                    }
                                                }
                                            )

                        async def process_document_refinement(
                            doc_id: str,
                            status_doc: DocProcessingStatus,
                            chunks: dict[str, Any],
                            chunk_results: list,
                            knowledge_graph_inst: BaseGraphStorage,
                            entity_vdb: BaseVectorStorage,
                            relationships_vdb: BaseVectorStorage,
                            global_config: dict[str, str],
                            pipeline_status: dict = None,
                            pipeline_status_lock=None,
                            llm_response_cache: BaseKVStorage | None = None,
                            current_file_number: int = 0,
                            total_files: int = 0,
                            file_path: str = "unknown_source",
                        ):
                            """Run a single refinement round: extract again then merge."""
                            try:
                                entity_relation_refine_task = None
                                file_extraction_stage_ok = False
                                entity_relation_refine_task = asyncio.create_task(
                                    self._process_entity_relation_graph_refinement(
                                        chunks,
                                        chunk_results,
                                        knowledge_graph_inst,
                                        entity_vdb,
                                        relationships_vdb,
                                        global_config,
                                        llm_response_cache,
                                        pipeline_status,
                                        pipeline_status_lock
                                    )
                                )
                                file_extraction_stage_ok = True

                                if file_extraction_stage_ok:
                                    try:
                                        # Wait for refined extraction results
                                        chunk_results = await entity_relation_refine_task

                                        # Merge refined results
                                        async with stage("merge_nodes_and_edges_refine", file_path=file_path):
                                            await merge_nodes_and_edges(
                                                chunk_results=chunk_results,
                                                knowledge_graph_inst=knowledge_graph_inst,
                                                entity_vdb=entity_vdb,
                                                relationships_vdb=relationships_vdb,
                                                global_config=global_config,
                                                pipeline_status=pipeline_status,
                                                pipeline_status_lock=pipeline_status_lock,
                                                llm_response_cache=llm_response_cache,
                                                current_file_number=current_file_number,
                                                total_files=total_files,
                                                file_path=file_path,
                                            )
                                        return chunk_results
                                    except Exception as e:
                                        # Log error and update pipeline status
                                        logger.error(traceback.format_exc())
                                        error_msg = f"Merging (refinement) stage failed in document {current_file_number}/{total_files}: {file_path}"
                                        logger.error(error_msg)
                                        async with pipeline_status_lock:
                                            pipeline_status["latest_message"] = error_msg
                                            pipeline_status["history_messages"].append(
                                                traceback.format_exc()
                                            )
                                            pipeline_status["history_messages"].append(
                                                error_msg
                                            )

                                        # Persist LLM cache if available
                                        if self.llm_response_cache:
                                            await self.llm_response_cache.index_done_callback()

                                        # Mark the document as failed
                                        await self.doc_status.upsert(
                                            {
                                                doc_id: {
                                                    "status": DocStatus.FAILED,
                                                    "error": str(e),
                                                    "content": status_doc.content,
                                                    "content_summary": status_doc.content_summary,
                                                    "content_length": status_doc.content_length,
                                                    "created_at": status_doc.created_at,
                                                    "updated_at": datetime.now().isoformat(),
                                                    "file_path": file_path,
                                                }
                                            }
                                        )

                            except Exception as e:
                                # Log error and update pipeline status
                                logger.error(traceback.format_exc())
                                error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                                logger.error(error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    pipeline_status["history_messages"].append(
                                        traceback.format_exc()
                                    )
                                    pipeline_status["history_messages"].append(error_msg)

                                    # Cancel tasks that are not completed yet
                                    all_tasks = (
                                        [entity_relation_refine_task]
                                        if entity_relation_refine_task
                                        else []
                                    )
                                    for task in all_tasks:
                                        if task and not task.done():
                                            task.cancel()

                                # Persist LLM cache if available
                                if self.llm_response_cache:
                                    await self.llm_response_cache.index_done_callback()

                                # Mark the document as failed
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.FAILED,
                                            "error": str(e),
                                            "content": status_doc.content,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                        }
                                    }
                                )

                        # Create processing tasks for all documents
                        doc_tasks = []
                        for doc_id, status_doc in to_process_docs.items():
                            doc_tasks.append(
                                process_document(
                                    doc_id,
                                    status_doc,
                                    split_by_character,
                                    split_by_character_only,
                                    split_by_page,
                                    pipeline_status,
                                    pipeline_status_lock,
                                    semaphore,
                                )
                            )

                        # Wait for all document processing to complete
                        await asyncio.gather(*doc_tasks)

                        # Check if there's a pending request to process more documents (with lock)
                        has_pending_request = False
                        async with pipeline_status_lock:
                            has_pending_request = pipeline_status.get("request_pending", False)
                            if has_pending_request:
                                # Clear the request flag before checking for more documents
                                pipeline_status["request_pending"] = False

                        if not has_pending_request:
                            break

                        log_message = "Processing additional documents due to pending request"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                        # Check again for pending documents
                        processing_docs, failed_docs, pending_docs = await asyncio.gather(
                            self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                            self.doc_status.get_docs_by_status(DocStatus.FAILED),
                            self.doc_status.get_docs_by_status(DocStatus.PENDING),
                        )

                        to_process_docs = {}
                        to_process_docs.update(processing_docs)
                        to_process_docs.update(failed_docs)
                        to_process_docs.update(pending_docs)
            finally:
                remove_stage_listener(sp.handle)
        # ---------------------------------------------------------------------

        log_message = "Document processing pipeline completed"
        logger.info(log_message)
        # Always reset busy status when done or if an exception occurs (with lock)
        async with pipeline_status_lock:
            pipeline_status["busy"] = False
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)
        
        # Optional: produce a waterfall chart for the latest trace
        try:
            tim_log_dir = os.path.join(self.working_dir, self.workspace, "logs/timings.jsonl")
            wat_log_dir = os.path.join(self.working_dir, self.workspace, "logs/waterfall.png")
            plot_waterfall_from_jsonl(tim_log_dir, outfile=wat_log_dir)
        except Exception as _:
            pass

    async def _insert_refine_done(
        self, refine_round
    ) -> None:
        new_filename2old_filename = {}
        work_dir = os.path.join(self.working_dir, self.workspace, 'iterative_refinement', f'r{refine_round}')
        os.makedirs(work_dir, exist_ok=True)

        def _save_vdb(storage_inst):
            _client_file_name = storage_inst._client_file_name
            client_file_name  = os.path.join(
                work_dir, f"vdb_{storage_inst.namespace}.json"
            )
            new_filename2old_filename[client_file_name] = _client_file_name
            storage_inst._client.storage_file = client_file_name
            return storage_inst.index_done_callback()

        tasks = [
            _save_vdb(cast(StorageNameSpace, storage_inst))
            for storage_inst in [  # type: ignore
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
        ]:
            storage_inst._client.storage_file = new_filename2old_filename[
                storage_inst._client.storage_file
            ]

        # Persist graphml for the graph namespace
        storage_inst = cast(StorageNameSpace, self.chunk_entity_relation_graph)
        _graphml_xml_file = storage_inst._graphml_xml_file
        graphml_xml_file = os.path.join(
            work_dir, f"graph_{storage_inst.namespace}.graphml"
        )
        storage_inst._graphml_xml_file = graphml_xml_file
        await storage_inst.index_done_callback()
        storage_inst._graphml_xml_file = _graphml_xml_file

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform an async query.
        """
        # Create a new trace per query to correlate all logs
        trace_id_var.set(uuid.uuid4().hex)

        async with stage("query_total", mode=param.mode or "default"):
            # If a custom model is provided in param, temporarily update global config
            global_config = asdict(self)
            # Save original query for vector search
            param.original_query = query

            if param.mode in ["local", "global", "hybrid", "mix"]:
                async with stage("kg_query"):
                    response = await kg_query(
                        query.strip(),
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        param,
                        global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                        chunks_vdb=self.chunks_vdb,
                    )
            elif param.mode == "naive":
                async with stage("naive_query"):
                    response = await naive_query(
                        query.strip(),
                        self.chunks_vdb,
                        self.text_chunks,
                        param,
                        global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                    )
            elif param.mode == "bypass":
                # Bypass mode: directly use LLM without knowledge retrieval
                use_llm_func = param.model_func or global_config["llm_model_func"]
                # Apply higher priority (8) to entity/relation summary tasks
                use_llm_func = partial(use_llm_func, _priority=8)

                param.stream = True if param.stream is None else param.stream
                async with stage("bypass_llm", stream=param.stream):
                    response = await use_llm_func(
                        query.strip(),
                        system_prompt=system_prompt,
                        history_messages=param.conversation_history,
                        stream=param.stream,
                    )
            elif param.mode == "mix_two_step":
                async with stage("kg_two_step_query"):
                    response = await kg_two_step_query(
                        query.strip(),
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        param,
                        global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                        chunks_vdb=self.chunks_vdb,
                    )
            else:
                raise ValueError(f"Unknown mode {param.mode}")

        await self._query_done()
        return response
