import asyncio
from typing import AsyncGenerator, AsyncIterable, Literal, Optional

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray

from ..core import AsyncNucleiSegmentation
from ..types import Result


async def stream_tiles_ordered(
    session: ClientSession,
    tile_generator: AsyncIterable[NDArray[np.uint8]],
    model: Literal["lsp-detr"] = "lsp-detr",
    max_concurrent: int = 5,
) -> AsyncGenerator[Result, None]:
    """
    Stream tiles with ordered output (preserves input order).

    Args:
        session: aiohttp ClientSession
        tile_generator: async iterable of image tiles
        model: model name
        max_concurrent: maximum number of concurrent requests

    Yields:
        Result objects in the same order as input tiles
    """
    segmenter = AsyncNucleiSegmentation(session, max_concurrent=max_concurrent)

    # Queue to maintain order
    result_queue: asyncio.Queue[tuple[int, Optional[Result], Optional[Exception]]] = (
        asyncio.Queue()
    )
    next_index = 0
    pending_results: dict[int, Result] = {}

    async def process_tile(tile: NDArray[np.uint8], index: int):
        try:
            result = await segmenter._process_tile_with_retry(tile, model)
            await result_queue.put((index, result, None))
        except Exception as e:
            print(f"Tile {index} failed after all retries: {e}")
            await result_queue.put((index, None, e))

    # Start processing tiles
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

                # Limit concurrent tasks
                if len(active_tasks) >= max_concurrent:
                    done, _ = await asyncio.wait(
                        active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    active_tasks.difference_update(done)
        except asyncio.CancelledError:
            print("Producer cancelled")
            raise
        except Exception as e:
            print(f"Error in tile generator: {e}")
        finally:
            generator_exhausted = True

    # Start producer task
    producer_task = asyncio.create_task(producer())

    # Yield results in order
    try:
        while True:
            # Check if we're done
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
                    print(f"Skipping tile {index} due to error")
                    next_index += 1
                    continue

                # Store result if not next in sequence
                if index != next_index:
                    if result is not None:
                        pending_results[index] = result
                else:
                    # Yield this result and any consecutive ones
                    if result is not None:
                        yield result
                    next_index += 1

                    while next_index in pending_results:
                        yield pending_results.pop(next_index)
                        next_index += 1

            except asyncio.TimeoutError:
                continue

    except asyncio.CancelledError:
        print("Ordered stream cancelled, cleaning up...")
        raise
    finally:
        # Clean up
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass
        for task in active_tasks:
            if not task.done():
                task.cancel()
        # Wait for all tasks to complete cancellation
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
