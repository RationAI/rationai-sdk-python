import asyncio
import itertools
import logging
from collections.abc import AsyncGenerator, AsyncIterable

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray

from rationai.segmentation.core import AsyncNucleiSegmentation
from rationai.segmentation.types import Result


logger = logging.getLogger(__name__)


async def stream_tiles_ordered(
    session: ClientSession,
    tile_generator: AsyncIterable[NDArray[np.uint8]],
    endpoint: str = "/lsp-detr",
    max_concurrent: int = 5,
    base_url: str = "",
    format: str = "raw",
) -> AsyncGenerator[Result, None]:
    """Stream tiles with ordered output (preserves input order).

    Args:
        session: aiohttp ClientSession
        tile_generator: async iterable of image tiles
        endpoint: API endpoint path (e.g., "/prostate" or "/lsp-detr")
        max_concurrent: maximum number of concurrent requests
        base_url: base URL for the API (without trailing slash)
        format: request format - "raw" for bytes, "json" for JSON payload

    Yields:
        Result objects in the same order as input tiles
    """
    segmenter = AsyncNucleiSegmentation(
        base_url=base_url, max_concurrent=max_concurrent
    )
    segmenter._session = session
    segmenter._owns_session = False

    # Queue for results: (index, result, error)
    result_queue: asyncio.Queue[tuple[int, Result | None, Exception | None]] = (
        asyncio.Queue()
    )
    next_index = 0  # Next index we want to yield
    pending_results: dict[int, Result] = {}  # Results that arrived out of order

    async def process_tile(tile: NDArray[np.uint8], index: int):
        """Process a single tile and put result in queue with its index."""
        try:
            result = await segmenter._process_tile_with_retry(tile, endpoint, format)
            result_queue.put_nowait((index, result, None))
        except Exception as e:
            logger.warning("Tile %d failed after all retries: %s", index, e)
            result_queue.put_nowait((index, None, e))

    active_tasks: list[asyncio.Task] = []
    generator_done = asyncio.Event()
    index_counter = itertools.count()

    async def producer():
        """Read tiles from generator and spawn processing tasks."""
        try:
            async for tile in tile_generator:
                idx = next(index_counter)
                task = asyncio.create_task(process_tile(tile, idx))
                active_tasks.append(task)

                # Limit concurrent processing
                if len(active_tasks) >= max_concurrent:
                    done, _ = await asyncio.wait(
                        set(active_tasks), return_when=asyncio.FIRST_COMPLETED
                    )
                    # Remove completed tasks from the active list
                    active_tasks[:] = [t for t in active_tasks if t not in done]
        finally:
            generator_done.set()

    producer_task = asyncio.create_task(producer())

    try:
        while True:
            # Exit when all tiles processed and no pending work
            if (
                generator_done.is_set()
                and not active_tasks
                and result_queue.empty()
                and not pending_results
            ):
                break

            # Try to get a result with short timeout to allow periodic exit checks
            try:
                index, result, error = await asyncio.wait_for(
                    result_queue.get(), timeout=0.1
                )
            except TimeoutError:
                continue

            if error:
                logger.warning("Skipping tile %d due to error", index)
                next_index += 1
                continue

            # Result arrived out of order - store it
            if index != next_index:
                if result is not None:
                    pending_results[index] = result
            # Result is next in sequence - yield it and any pending results
            else:
                if result is not None:
                    yield result
                next_index += 1

                # Yield any buffered results that are now in order
                while next_index in pending_results:
                    yield pending_results.pop(next_index)
                    next_index += 1

    finally:
        if not producer_task.done():
            producer_task.cancel()
        for task in active_tasks:
            if not task.done():
                task.cancel()
