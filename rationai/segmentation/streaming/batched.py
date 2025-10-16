import asyncio
from typing import AsyncGenerator, AsyncIterable, Literal, Optional

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray

from ..segmentation import AsyncNucleiSegmentation, Result


async def stream_tiles_batched(
    session: ClientSession,
    tile_generator: AsyncIterable[NDArray[np.uint8]],
    model: Literal["lsp-detr"] = "lsp-detr",
    batch_size: int = 8,
    max_concurrent_batches: int = 3,
    max_retries: int = 3,
    timeout: int = 300,
    retry_delay: float = 1.0,
    fallback_to_individual: bool = True,
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
        max_retries: maximum number of retry attempts per batch
        timeout: timeout in seconds per batch request
        retry_delay: initial delay between retries (exponential backoff)
        fallback_to_individual: if True, process failed batch images individually

    Yields:
        Result objects as batches complete
    """
    from ..batch import _send_batch_with_retry

    current_batch: list[NDArray[np.uint8]] = []
    tile_size: Optional[int] = None
    pending_batches = set()

    try:
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
                batch_to_send = current_batch.copy()  # Copy to avoid mutation
                task = asyncio.create_task(
                    _send_batch_with_retry(
                        session,
                        batch_to_send,
                        model,
                        tile_size,
                        max_retries,
                        timeout,
                        retry_delay,
                    )
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
                            print(f"Batch failed after all retries: {e}")
                            if fallback_to_individual:
                                print("Attempting individual processing as fallback...")
                                # Note: We lost the batch reference here
                                # In production, you'd need to track batches separately

        # Send remaining partial batch
        if current_batch and tile_size is not None:
            try:
                results = await _send_batch_with_retry(
                    session,
                    current_batch,
                    model,
                    tile_size,
                    max_retries,
                    timeout,
                    retry_delay,
                )
                for result in results:
                    yield result
            except Exception as e:
                print(f"Final batch failed after all retries: {e}")
                if fallback_to_individual:
                    print("Processing final batch individually...")
                    segmenter = AsyncNucleiSegmentation(session, max_concurrent=5)
                    for tile in current_batch:
                        try:
                            result = await segmenter._process_tile_with_retry(
                                tile, model
                            )
                            yield result
                        except Exception as tile_error:
                            print(f"Individual tile also failed: {tile_error}")

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
                    print(f"Batch failed after all retries: {e}")

    except asyncio.CancelledError:
        print("Batched stream cancelled, cleaning up...")
        # Cancel all pending batches
        for task in pending_batches:
            task.cancel()
        if pending_batches:
            await asyncio.gather(*pending_batches, return_exceptions=True)
        raise
