import asyncio
from typing import AsyncGenerator, AsyncIterable, Literal

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray

from ..segmentation import AsyncNucleiSegmentation, Result


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

    try:
        async for tile in tile_generator:
            # Create task for this tile (uses built-in retry logic)
            task = asyncio.create_task(segmenter._process_tile_with_retry(tile, model))
            pending.add(task)

            # If we've hit the concurrency limit, wait for at least one to complete
            if len(pending) >= max_concurrent:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    try:
                        result = await completed_task
                        yield result
                    except Exception as e:
                        print(f"Error processing tile (skipping): {e}")

        # Process remaining tasks
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for completed_task in done:
                try:
                    result = await completed_task
                    yield result
                except Exception as e:
                    print(f"Error processing tile (skipping): {e}")

    except asyncio.CancelledError:
        print("Stream processing cancelled, cleaning up...")
        # Cancel all pending tasks
        for task in pending:
            task.cancel()
        # Wait for cancellation to complete
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise
