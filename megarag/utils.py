import json
import logging
import logging.handlers
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

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

from lightrag.utils import (
    compute_args_hash,
    generate_cache_key,
    handle_cache,
    logger,
    statistic_data,
    remove_think_tags,
    save_to_cache,
    CacheData
)

async def use_llm_func_with_cache(
    input_text: str,
    use_llm_func: callable,
    input_images: list[str] = [],
    llm_response_cache: "BaseKVStorage | None" = None,
    max_tokens: int = None,
    history_messages: list[dict[str, str]] = None,
    cache_type: str = "extract",
    chunk_id: str | None = None,
    cache_keys_collector: list = None,
) -> str:
    """Call LLM function with cache support

    If cache is available and enabled (determined by handle_cache based on mode),
    retrieve result from cache; otherwise call LLM function and save result to cache.

    Args:
        input_text: Input text to send to LLM
        use_llm_func: LLM function with higher priority
        llm_response_cache: Cache storage instance
        max_tokens: Maximum tokens for generation
        history_messages: History messages list
        cache_type: Type of cache
        chunk_id: Chunk identifier to store in cache
        text_chunks_storage: Text chunks storage to update llm_cache_list
        cache_keys_collector: Optional list to collect cache keys for batch processing

    Returns:
        LLM response text
    """
    if llm_response_cache:
        if history_messages:
            history = json.dumps(history_messages, ensure_ascii=False)
            _prompt = history + "\n" + input_text
        else:
            _prompt = input_text

        if input_images:
            _prompt = _prompt + "\n" + ", ".join(input_images)

        arg_hash = compute_args_hash(_prompt)
        # Generate cache key for this LLM call
        cache_key = generate_cache_key("default", cache_type, arg_hash)

        cached_return, _1, _2, _3 = await handle_cache(
            llm_response_cache,
            arg_hash,
            _prompt,
            "default",
            cache_type=cache_type,
        )
        if cached_return:
            print(f"Found cache for {arg_hash}")
            logger.debug(f"Found cache for {arg_hash}")
            statistic_data["llm_cache"] += 1

            # Add cache key to collector if provided
            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

            return cached_return
        statistic_data["llm_call"] += 1

        # Call LLM
        kwargs = {}
        if history_messages:
            kwargs["history_messages"] = history_messages
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        res: str = await use_llm_func(input_text, input_images, **kwargs)
        res = remove_think_tags(res)

        if llm_response_cache.global_config.get("enable_llm_cache_for_entity_extract"):
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type=cache_type,
                    chunk_id=chunk_id,
                ),
            )

            # Add cache key to collector if provided
            if cache_keys_collector is not None:
                cache_keys_collector.append(cache_key)

        return res

    # When cache is disabled, directly call LLM
    kwargs = {}
    if history_messages:
        kwargs["history_messages"] = history_messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    logger.info(f"Call LLM function with query text length: {len(input_text)}")
    res = await use_llm_func(input_text, input_images, **kwargs)
    return remove_think_tags(res)

def plot_waterfall_from_jsonl(jsonl_path: str, trace_id: Optional[str] = None,
                              outfile: str = "waterfall.png",
                              only_top_level: bool = False) -> str:
    """Render a waterfall chart for a given trace_id from the JSONL timing file.

    Requirements: pandas, matplotlib. Install: pip install pandas matplotlib
    """

    # Stream read JSONL to keep memory small; collect only the chosen trace
    rows = []
    latest_trace = None
    latest_end = -1
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type") != "stage_timing":
                continue
            tid = rec.get("trace_id")
            if trace_id is None:
                # Track the latest trace by max end_ns
                end_ns = rec.get("end_ns", -1)
                if end_ns is not None and end_ns > latest_end:
                    latest_end = end_ns
                    latest_trace = tid
            if trace_id is not None and tid != trace_id:
                continue
            rows.append(rec)

    if trace_id is None:
        trace_id = latest_trace

    if not rows or trace_id is None:
        raise ValueError(f"No timing records found for trace_id={trace_id} in {jsonl_path}")

    # Filter rows to the chosen trace_id
    rows = [r for r in rows if r.get("trace_id") == trace_id]
    if not rows:
        raise ValueError(f"No timing records found for trace_id={trace_id} in {jsonl_path}")

    df = pd.DataFrame(rows)
    t0 = df["start_ns"].min()
    df["start_ms"] = (df["start_ns"] - t0) / 1e6
    df["elapsed_ms"] = df["elapsed_ms"].astype(float)

    # Compute depth for indentation using parent_span_id
    parents = dict(zip(df["span_id"], df["parent_span_id"]))

    def depth_of(span_id: Optional[str]) -> int:
        d, seen = 0, set()
        s = span_id
        while s and s in parents and parents[s]:
            if s in seen:
                break
            seen.add(s)
            s = parents[s]
            d += 1
        return d

    df["depth"] = df["span_id"].apply(depth_of)
    if only_top_level:
        df = df[df["depth"] == 0].copy()

    # Indented labels
    df["label"] = df.apply(lambda r: ("  " * int(r["depth"])) + str(r["stage"]), axis=1)

    # Sort by start (so the y-axis is roughly chronological)
    df = df.sort_values("start_ms", ascending=True).reset_index(drop=True)

    # Plot
    fig_h = max(3.5, 0.45 * len(df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_pos = list(range(len(df)))
    ax.barh(y_pos, df["elapsed_ms"], left=df["start_ms"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"Waterfall (trace_id={trace_id})")
    ax.invert_yaxis()  # earliest at top
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    return outfile
