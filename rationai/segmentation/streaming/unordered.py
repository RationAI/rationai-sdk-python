import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterable

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray

from rationai.segmentation.core import AsyncNucleiSegmentation
from rationai.segmentation.types import Result


logger = logging.getLogger(__name__)


async def stream_tiles(
    session: ClientSession,
    tile_generator: AsyncIterable[NDArray[np.uint8]],
    endpoint: str = "/lsp-detr",
    max_concurrent: int = 5,
    base_url: str = "",
    format: str = "raw",
) -> AsyncGenerator[Result, None]:
    """Stream tiles to the segmentation server asynchronously with concurrent processing.

    Results are yielded as soon as they're available (not necessarily in order).

    Args:
        session: aiohttp ClientSession
        tile_generator: async iterable of image tiles
        endpoint: API endpoint path (e.g., "/prostate" or "/lsp-detr")
        max_concurrent: maximum number of concurrent requests
        base_url: base URL for the API (without trailing slash)
        format: request format - "raw" for bytes, "json" for JSON payload

    Yields:
        Result objects as they complete (not in input order)

    Raises:
        Exception: if any tile processing fails
    """
    segmenter = AsyncNucleiSegmentation(
        base_url=base_url, max_concurrent=max_concurrent
    )
    segmenter._session = session
    segmenter._owns_session = False

    pending: list[asyncio.Task] = []  # List of active tasks

    try:
        async for tile in tile_generator:
            task = asyncio.create_task(
                segmenter._process_tile_with_retry(tile, endpoint, format)
            )
            pending.append(task)

            # Limit concurrent tasks - yield results as they complete
            if len(pending) >= max_concurrent:
                done, _ = await asyncio.wait(
                    set(pending), return_when=asyncio.FIRST_COMPLETED
                )
                pending = [t for t in pending if t not in done]
                for completed_task in done:
                    result = completed_task.result()
                    yield result  # Yield immediately (unordered)

        # Process remaining tasks after generator is exhausted
        while pending:
            done, _ = await asyncio.wait(
                set(pending), return_when=asyncio.FIRST_COMPLETED
            )
            pending = [t for t in pending if t not in done]
            for completed_task in done:
                result = completed_task.result()
                yield result

    except (asyncio.CancelledError, Exception):
        logger.debug("Stream processing interrupted, cleaning up...")
        for task in pending:
            task.cancel()
        raise
