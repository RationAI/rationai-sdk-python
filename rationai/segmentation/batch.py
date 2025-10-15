import asyncio
from typing import List, Literal

import numpy as np
from aiohttp import ClientResponseError, ClientSession
from numpy.typing import NDArray

from rationai.segmentation.segmentation import Result


async def batch_process(
    session: ClientSession,
    images: List[NDArray[np.uint8]],
    model: Literal["lsp-detr"] = "lsp-detr",
    batch_size: int = 8,
) -> List[Result]:
    """
    Send images to the segmentation server in batches.

    This function combines multiple images into batches and sends them as single requests,
    allowing the Ray Serve backend to perform efficient batch processing.

    Args:
        session: aiohttp ClientSession
        images: list of image tiles (all must be the same size!)
        model: model name
        batch_size: number of images to send in each batch

    Returns:
        List of Result objects in the same order as input images.
    """
    if not images:
        return []

    h, w = images[0].shape[:2]
    if any(img.shape[:2] != (h, w) for img in images):
        raise ValueError("All images in the batch must have the same shape")

    # Split images into batches
    batches = [images[i : i + batch_size] for i in range(0, len(images), batch_size)]

    # Create tasks for each batch
    batch_tasks = [_send_batch(session, batch, model, h) for batch in batches]

    all_results = []
    try:
        # Send all batches concurrently
        batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for batch_result in batch_responses:
            if isinstance(batch_result, list):
                # Successfully processed batch
                all_results.extend(batch_result)
            elif isinstance(batch_result, Exception):
                print(f"Batch processing error: {batch_result}")
                # Return empty results for failed batch to maintain order
                # You might want to handle this differently based on your needs
                return []
            else:
                print(f"Unexpected response type: {type(batch_result)}")
                return []

    except asyncio.CancelledError:
        print("Batch processing was cancelled.")
        raise
    except Exception as e:
        print(f"An error occurred during batch processing: {e}")
        return []

    return all_results


async def _send_batch(
    session: ClientSession,
    batch: List[NDArray[np.uint8]],
    model: str,
    tile_size: int,
) -> List[Result]:
    """
    Send a single batch of images to the server.

    Args:
        session: aiohttp ClientSession
        batch: list of images to send in this batch
        model: model name
        tile_size: size of the tiles (assumed square)

    Returns:
        List of Result objects for this batch
    """
    # Stack images along batch dimension: (batch_size, h, w, c)
    batch_array = np.stack(batch, axis=0)

    try:
        async with session.post(
            f"/{model}/{tile_size}",
            data=batch_array.tobytes(),
            headers={
                "Content-Type": "application/octet-stream",
                "X-Batch-Size": str(len(batch)),
            },
        ) as response:
            response.raise_for_status()
            results = await response.json()

            # Ensure we got results for all images in the batch
            if not isinstance(results, list) or len(results) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} results, got {len(results) if isinstance(results, list) else 'non-list'}"
                )

            return results

    except ClientResponseError as e:
        print(f"HTTP error during batch processing: {e.status} - {e.message}")
        raise
    except Exception as e:
        print(f"Error processing batch: {e}")
        raise
