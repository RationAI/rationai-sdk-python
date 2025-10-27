import asyncio
from typing import AsyncGenerator, AsyncIterable

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray

from rationai.segmentation.core import AsyncNucleiSegmentation
from rationai.segmentation.types import Result


async def stream_tiles(
    session: ClientSession,
    tile_generator: AsyncIterable[NDArray[np.uint8]],
    model: str = "lsp-detr",
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

    Raises:
        Exception: if any tile processing fails
    """
    segmenter = AsyncNucleiSegmentation(max_concurrent=max_concurrent)
    segmenter._session = session

    pending = set()
    error: Exception | None = None

    try:
        async for tile in tile_generator:
            if error:
                break

            task = asyncio.create_task(segmenter._process_tile_with_retry(tile, model))
            pending.add(task)

            if len(pending) >= max_concurrent:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    try:
                        result = await completed_task
                        yield result
                    except Exception as e:
                        error = e
                        for task in pending:
                            task.cancel()
                        raise

        if not error:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    try:
                        result = await completed_task
                        yield result
                    except Exception:
                        for task in pending:
                            task.cancel()
                        raise

    except asyncio.CancelledError:
        print("Stream processing cancelled, cleaning up...")
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise
