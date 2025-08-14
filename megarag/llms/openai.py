import logging
import numpy as np
import mimetypes
import base64

from pathlib import Path
from typing import (
    Any, 
    AsyncIterator, 
    List, 
    Optional, 
    Union
)
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
    VERBOSE_DEBUG,
    verbose_debug,
)
from lightrag.llm.openai import (
    create_openai_async_client,
    InvalidResponseError
)
from lightrag.types import GPTKeywordExtractionFormat

async def gpt_4o_mini_complete(
    prompt,
    input_images=None,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        input_images=input_images,
        **kwargs,
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[dict[str, Any]]] = None,
    *,
    input_images: Optional[List[str]] = None,  # ← NEW
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Complete *prompt* with OpenAI, supporting optional image inputs.

    Args:
        model:          Vision‑capable model (e.g. ``"gpt-4o-mini"``)
        prompt:         User text prompt.
        system_prompt:  System message.
        history_messages: Prior conversation turns.
        input_images:   List of **local paths** *or* **remote URLs** to images.
        base_url / api_key / token_tracker / kwargs: unchanged.

    Returns:
        Either the full string completion or an **async iterator** of
        streamed text chunks, matching the behaviour of the original code.
    """
    if history_messages is None:
        history_messages = []

    # Reduce OpenAI log noise unless VERBOSE_DEBUG is on
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)

    # Pull out and honour any per‑client kwargs
    client_configs = kwargs.pop("openai_client_configs", {})

    # Instantiate the async client (helper remains unchanged)
    openai_async_client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        client_configs=client_configs,
    )

    # Strip kwargs meant only for upstream callers
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    messages: List[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)

    # Helper: turn a path / URL into the vision content item
    def _img_item(path: str) -> dict[str, Any]:
        if Path(path).is_file():  # local ⇒ convert to data: URL
            mime, _ = mimetypes.guess_type(path)
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            url = f"data:{mime or 'image/png'};base64,{b64}"
        else:                      # already remote
            url = path
        return {"type": "image_url", "image_url": {"url": url}}

    if input_images:
        # Vision models expect *array* content when mixing text & images
        image_items = [_img_item(p) for p in input_images]
        text_item = {"type": "text", "text": prompt}
        messages.append({"role": "user", "content": [*image_items, text_item]})
    else:
        messages.append({"role": "user", "content": prompt})

    # Allow caller to override the whole messages list via kwargs
    messages = kwargs.pop("messages", messages)

    logger.debug("===== Entering func of LLM =====")
    logger.debug("Model: %s   Base URL: %s", model, base_url)
    logger.debug("Additional kwargs: %s", kwargs)
    logger.debug("Num of history messages: %d", len(history_messages))
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LLM =====")

    try:
        if "response_format" in kwargs:  # e.g. json mode
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
    except (APIConnectionError, RateLimitError, APITimeoutError) as e:
        logger.error("%s: %s", e.__class__.__name__, e)
        await openai_async_client.close()
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(
            "OpenAI API Call Failed,\nModel: %s,\nParams: %s,\nGot: %s",
            model,
            kwargs,
            e,
        )
        await openai_async_client.close()
        raise

    if hasattr(response, "__aiter__"):
        async def inner() -> AsyncIterator[str]:
            iteration_started = False
            final_chunk_usage = None
            try:
                iteration_started = True
                async for chunk in response:
                    # Capture token usage if present
                    if hasattr(chunk, "usage") and chunk.usage:
                        final_chunk_usage = chunk.usage
                        logger.debug(
                            "Received usage info in streaming chunk: %s", chunk.usage
                        )

                    if not getattr(chunk, "choices", None):
                        logger.warning("Received chunk without choices: %s", chunk)
                        continue

                    delta = chunk.choices[0].delta
                    if not getattr(delta, "content", None):
                        continue  # no content yet (maybe final usage‑only chunk)

                    content = delta.content
                    if r"\u" in content:
                        content = safe_unicode_decode(content.encode())
                    yield content

                # Record usage after stream finishes
                if token_tracker and final_chunk_usage:
                    token_counts = {
                        "prompt_tokens": getattr(
                            final_chunk_usage, "prompt_tokens", 0
                        ),
                        "completion_tokens": getattr(
                            final_chunk_usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
                    }
                    token_tracker.add_usage(token_counts)
                    logger.debug("Streaming token usage (from API): %s", token_counts)
            finally:
                # Ensure resources are released
                if (
                    iteration_started
                    and hasattr(response, "aclose")
                    and callable(response.aclose)
                ):
                    try:
                        await response.aclose()
                    except Exception:  # noqa: BLE001
                        pass
                await openai_async_client.close()
        return inner()

    try:
        if (
            not response
            or not response.choices
            or not hasattr(response.choices[0], "message")
            or not getattr(response.choices[0].message, "content", None)
        ):
            await openai_async_client.close()
            raise InvalidResponseError("Invalid response from OpenAI API")

        content = response.choices[0].message.content
        if not content.strip():
            await openai_async_client.close()
            raise InvalidResponseError("Received empty content from OpenAI API")

        if r"\u" in content:
            content = safe_unicode_decode(content.encode())

        if token_tracker and hasattr(response, "usage"):
            token_counts = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            token_tracker.add_usage(token_counts)

        logger.debug("Response content len: %d", len(content))
        verbose_debug(f"Response: {response}")
        return content
    finally:
        await openai_async_client.close()

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
    ),
)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    client_configs: dict[str, Any] = None,
) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI's API.

    Args:
        texts: List of texts to embed.
        model: The OpenAI embedding model to use.
        base_url: Optional base URL for the OpenAI API.
        api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        client_configs: Additional configuration options for the AsyncOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        APIConnectionError: If there is a connection error with the OpenAI API.
        RateLimitError: If the OpenAI API rate limit is exceeded.
        APITimeoutError: If the OpenAI API request times out.
    """
    # Create the OpenAI client
    openai_async_client = create_openai_async_client(
        api_key=api_key, base_url=base_url, client_configs=client_configs
    )

    async with openai_async_client:
        response = await openai_async_client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )
        return np.array([dp.embedding for dp in response.data])
