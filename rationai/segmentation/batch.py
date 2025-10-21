import asyncio
from typing import List, Literal, Optional

import numpy as np
from aiohttp import ClientResponseError, ClientSession, ClientTimeout
from numpy.typing import NDArray

from rationai.segmentation.types import Result


async def batch_process(
    session: ClientSession,
    images: List[NDArray[np.uint8]],
    model: Literal["lsp-detr"] = "lsp-detr",
    batch_size: int = 8,
    max_retries: int = 3,
    timeout: int = 300,
    retry_delay: float = 1.0,
    fallback_to_individual: bool = True,
) -> List[Result]:
    """
    Send images to the segmentation server in batches with robust error handling.

    This function combines multiple images into batches and sends them as single requests,
    allowing the Ray Serve backend to perform efficient batch processing.

    Args:
        session: aiohttp ClientSession
        images: list of image tiles (all must be the same size!)
        model: model name
        batch_size: number of images to send in each batch
        max_retries: maximum number of retry attempts per batch
        timeout: timeout in seconds for each batch request
        retry_delay: initial delay between retries (exponential backoff)
        fallback_to_individual: if True, process images individually when batch fails

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

    # Create tasks for each batch with retry logic
    batch_tasks = [
        _send_batch_with_retry(
            session, batch, model, h, max_retries, timeout, retry_delay
        )
        for batch in batches
    ]

    all_results = []
    failed_batches = []

    try:
        # Send all batches concurrently
        batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for batch_idx, batch_result in enumerate(batch_responses):
            if isinstance(batch_result, list):
                # Successfully processed batch
                all_results.extend(batch_result)
            elif isinstance(batch_result, Exception):
                print(
                    f"Batch {batch_idx + 1}/{len(batches)} failed after retries: {batch_result}"
                )
                failed_batches.append((batch_idx, batches[batch_idx]))
                # Placeholder for failed batch to maintain order
                all_results.extend([None] * len(batches[batch_idx]))
            else:
                print(f"Unexpected response type: {type(batch_result)}")
                failed_batches.append((batch_idx, batches[batch_idx]))
                all_results.extend([None] * len(batches[batch_idx]))

    except asyncio.CancelledError:
        print("Batch processing was cancelled.")
        raise
    except Exception as e:
        print(f"Critical error during batch processing: {e}")
        raise

    # Try to recover failed batches by processing individually
    if failed_batches and fallback_to_individual:
        print(
            f"Attempting to recover {len(failed_batches)} failed batches by processing individually..."
        )
        for batch_idx, failed_batch in failed_batches:
            try:
                individual_results = await _process_individually(
                    session, failed_batch, model, h, max_retries, timeout, retry_delay
                )
                # Replace None placeholders with actual results
                start_idx = batch_idx * batch_size
                for i, result in enumerate(individual_results):
                    all_results[start_idx + i] = result
                print(f"Successfully recovered batch {batch_idx + 1}")
            except Exception as e:
                print(f"Failed to recover batch {batch_idx + 1}: {e}")

    # Check if we have any None results (unrecoverable failures)
    if None in all_results:
        failed_count = all_results.count(None)
        print(
            f"Warning: {failed_count}/{len(all_results)} images failed to process and could not be recovered"
        )

    return all_results


async def _send_batch_with_retry(
    session: ClientSession,
    batch: List[NDArray[np.uint8]],
    model: str,
    tile_size: int,
    max_retries: int,
    timeout: int,
    retry_delay: float,
) -> List[Result]:
    """
    Send a batch with exponential backoff retry logic.

    Args:
        session: aiohttp ClientSession
        batch: list of images to send in this batch
        model: model name
        tile_size: size of the tiles (assumed square)
        max_retries: maximum number of retry attempts
        timeout: timeout in seconds
        retry_delay: initial delay between retries

    Returns:
        List of Result objects for this batch
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return await _send_batch(session, batch, model, tile_size, timeout)
        except asyncio.TimeoutError as e:
            last_exception = e
            delay = retry_delay * (2**attempt)
            print(
                f"Batch timeout (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
        except ClientResponseError as e:
            last_exception = e
            # Don't retry on client errors (4xx), only server errors (5xx)
            if 400 <= e.status < 500:
                print(f"Client error {e.status}, not retrying: {e.message}")
                raise
            delay = retry_delay * (2**attempt)
            print(
                f"Server error {e.status} (attempt {attempt + 1}/{max_retries}). Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
        except Exception as e:
            last_exception = e
            delay = retry_delay * (2**attempt)
            print(
                f"Error processing batch (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    # All retries exhausted - raise the last exception or a RuntimeError if somehow none was caught
    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError("All retries exhausted with no exception captured")


async def _send_batch(
    session: ClientSession,
    batch: List[NDArray[np.uint8]],
    model: str,
    tile_size: int,
    timeout: int,
) -> List[Result]:
    """
    Send a single batch of images to the server.

    Args:
        session: aiohttp ClientSession
        batch: list of images to send in this batch
        model: model name
        tile_size: size of the tiles (assumed square)
        timeout: timeout in seconds

    Returns:
        List of Result objects for this batch
    """
    # Stack images along batch dimension: (batch_size, h, w, c)
    batch_array = np.stack(batch, axis=0)

    request_timeout = ClientTimeout(total=timeout)

    async with session.post(
        f"/{model}/{tile_size}",
        data=batch_array.tobytes(),
        headers={
            "Content-Type": "application/octet-stream",
            "X-Batch-Size": str(len(batch)),
        },
        timeout=request_timeout,
    ) as response:
        response.raise_for_status()
        results = await response.json()

        # Ensure we got results for all images in the batch
        if not isinstance(results, list) or len(results) != len(batch):
            raise ValueError(
                f"Expected {len(batch)} results, got {len(results) if isinstance(results, list) else 'non-list'}"
            )

        return results


async def _process_individually(
    session: ClientSession,
    images: List[NDArray[np.uint8]],
    model: str,
    tile_size: int,
    max_retries: int,
    timeout: int,
    retry_delay: float,
) -> List[Optional[Result]]:
    """
    Process images individually as a fallback when batch processing fails.

    Args:
        session: aiohttp ClientSession
        images: list of images to process
        model: model name
        tile_size: size of the tiles
        max_retries: maximum number of retry attempts per image
        timeout: timeout in seconds per image
        retry_delay: initial delay between retries

    Returns:
        List of Result objects (or None for failed images)
    """
    tasks = [
        _send_batch_with_retry(
            session, [img], model, tile_size, max_retries, timeout, retry_delay
        )
        for img in images
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert results, handling exceptions
    processed_results = []
    for idx, result in enumerate(results):
        if isinstance(result, list) and len(result) == 1:
            processed_results.append(result[0])
        elif isinstance(result, Exception):
            print(f"Image {idx} failed individually: {result}")
            processed_results.append(None)
        else:
            print(f"Unexpected result for image {idx}: {type(result)}")
            processed_results.append(None)

    return processed_results
