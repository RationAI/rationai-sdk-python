import asyncio
from typing import AsyncGenerator, AsyncIterable, Literal

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray

from .segmentation import AsyncNucleiSegmentation, Result


async def stream_tiles(
    session: ClientSession,
    tile_generator: AsyncIterable[NDArray[np.uint8]],
    model: Literal["lsp-detr"] = "lsp-detr",
    max_concurrent: int = 5,
) -> AsyncGenerator[Result, None]:
    """
    Stream tiles to the segmentation server asynchronously with concurrent processing.

    Results are yielded as soon as they're available (not necessarily in order).

    Args:
        session: aiohttp ClientSession
        tile_generator: async iterable of image tiles
        model: model name
        max_concurrent: maximum number of concurrent requests

    Yields:
        Result objects as they complete (not in input order)
    """
    segmenter = AsyncNucleiSegmentation(session, max_concurrent=max_concurrent)

    # Queue to hold pending tasks
    pending = set()

    async for tile in tile_generator:
        # Create task for this tile
        task = asyncio.create_task(segmenter._process_tile_with_retry(tile, model))
        pending.add(task)

        # If we've hit the concurrency limit, wait for at least one to complete
        if len(pending) >= max_concurrent:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for completed_task in done:
                try:
                    yield await completed_task
                except Exception as e:
                    print(f"Error processing tile: {e}")

    # Process remaining tasks
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for completed_task in done:
            try:
                yield await completed_task
            except Exception as e:
                print(f"Error processing tile: {e}")


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
    result_queue = asyncio.Queue()
    next_index = 0
    pending_results = {}

    async def process_tile(tile: NDArray[np.uint8], index: int):
        try:
            result = await segmenter._process_tile_with_retry(tile, model)
            await result_queue.put((index, result, None))
        except Exception as e:
            await result_queue.put((index, None, e))

    # Start processing tiles
    tile_index = 0
    active_tasks = set()
    generator_exhausted = False

    async def producer():
        nonlocal tile_index, generator_exhausted
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
                    print(f"Error processing tile {index}: {error}")
                    continue

                # Store result if not next in sequence
                if index != next_index:
                    pending_results[index] = result
                else:
                    # Yield this result and any consecutive ones
                    yield result
                    next_index += 1

                    while next_index in pending_results:
                        yield pending_results.pop(next_index)
                        next_index += 1

            except asyncio.TimeoutError:
                continue

    finally:
        # Clean up
        if not producer_task.done():
            producer_task.cancel()
        for task in active_tasks:
            task.cancel()


async def stream_tiles_batched(
    session: ClientSession,
    tile_generator: AsyncIterable[NDArray[np.uint8]],
    model: Literal["lsp-detr"] = "lsp-detr",
    batch_size: int = 8,
    max_concurrent_batches: int = 3,
) -> AsyncGenerator[Result, None]:
    """
    Stream tiles with batching for better throughput.

    Accumulates tiles into batches before sending to the server.

    Args:
        session: aiohttp ClientSession
        tile_generator: async iterable of image tiles (must be same size!)
        model: model name
        batch_size: number of tiles per batch
        max_concurrent_batches: maximum number of concurrent batch requests

    Yields:
        Result objects as batches complete
    """
    from .batch import _send_batch

    current_batch = []
    tile_size = None
    pending_batches = set()

    async for tile in tile_generator:
        # Validate tile size consistency
        h, w = tile.shape[:2]
        if tile_size is None:
            tile_size = h
        elif h != tile_size or w != tile_size:
            raise ValueError(
                f"All tiles must be the same size. Expected {tile_size}, got {h}x{w}"
            )

        current_batch.append(tile)

        # Send batch when full
        if len(current_batch) >= batch_size:
            assert tile_size is not None  # For type checker
            task = asyncio.create_task(
                _send_batch(session, current_batch, model, tile_size)
            )
            pending_batches.add(task)
            current_batch = []

            # Limit concurrent batches
            if len(pending_batches) >= max_concurrent_batches:
                done, pending_batches = await asyncio.wait(
                    pending_batches, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    try:
                        results = await completed_task
                        for result in results:
                            yield result
                    except Exception as e:
                        print(f"Error processing batch: {e}")

    # Send remaining partial batch
    if current_batch and tile_size is not None:
        try:
            results = await _send_batch(session, current_batch, model, tile_size)
            for result in results:
                yield result
        except Exception as e:
            print(f"Error processing final batch: {e}")

    # Process remaining batches
    while pending_batches:
        done, pending_batches = await asyncio.wait(
            pending_batches, return_when=asyncio.FIRST_COMPLETED
        )
        for completed_task in done:
            try:
                results = await completed_task
                for result in results:
                    yield result
            except Exception as e:
                print(f"Error processing batch: {e}")
