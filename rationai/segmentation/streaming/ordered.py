import asyncio
import contextlib
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
    model: str = "lsp-detr",
    max_concurrent: int = 5,
) -> AsyncGenerator[Result, None]:
    """Stream tiles with ordered output (preserves input order).

    Args:
        session: aiohttp ClientSession
        tile_generator: async iterable of image tiles
        model: model name
        max_concurrent: maximum number of concurrent requests

    Yields:
        Result objects in the same order as input tiles
    """
    segmenter = AsyncNucleiSegmentation(max_concurrent=max_concurrent)
    segmenter._session = session

    result_queue: asyncio.Queue[tuple[int, Result | None, Exception | None]] = (
        asyncio.Queue()
    )
    next_index = 0
    pending_results: dict[int, Result] = {}

    async def process_tile(tile: NDArray[np.uint8], index: int):
        try:
            result = await segmenter._process_tile_with_retry(tile, model)
            await result_queue.put((index, result, None))
        except Exception as e:
            logger.warning("Tile %d failed after all retries: %s", index, e)
            await result_queue.put((index, None, e))

    tile_index = 0
    active_tasks = set()
    generator_exhausted = False

    async def producer():
        nonlocal tile_index, generator_exhausted
        try:
            async for tile in tile_generator:
                task = asyncio.create_task(process_tile(tile, tile_index))
                active_tasks.add(task)
                tile_index += 1

                if len(active_tasks) >= max_concurrent:
                    done, _ = await asyncio.wait(
                        active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    active_tasks.difference_update(done)
        except asyncio.CancelledError:
            logger.debug("Producer cancelled")
            raise
        except Exception as e:
            logger.error("Error in tile generator: %s", e)
        finally:
            generator_exhausted = True

    producer_task = asyncio.create_task(producer())

    try:
        while True:
            if (
                generator_exhausted
                and not active_tasks
                and result_queue.empty()
                and next_index >= tile_index
            ):
                break

            try:
                index, result, error = await asyncio.wait_for(
                    result_queue.get(), timeout=0.1
                )

                if error:
                    logger.warning("Skipping tile %d due to error", index)
                    next_index += 1
                    continue

                if index != next_index:
                    if result is not None:
                        pending_results[index] = result
                else:
                    if result is not None:
                        yield result
                    next_index += 1

                    while next_index in pending_results:
                        yield pending_results.pop(next_index)
                        next_index += 1

            except TimeoutError:
                continue

    except asyncio.CancelledError:
        logger.debug("Ordered stream cancelled, cleaning up...")
    finally:
        if not producer_task.done():
            producer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer_task
        for task in active_tasks:
            if not task.done():
                task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
