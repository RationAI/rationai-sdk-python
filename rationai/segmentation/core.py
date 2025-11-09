import asyncio
import logging
from collections.abc import AsyncIterable, Iterable
from typing import cast

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray
from openslide import OpenSlide
from PIL import Image
from ratiopath.tiling import grid_tiles

from rationai.segmentation.types import Result, Tile


logger = logging.getLogger(__name__)


class AsyncNucleiSegmentation:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30,
        max_concurrent: int = 5,
    ):
        self._base_url = base_url.rstrip("/")  # Remove trailing slash
        self._session: ClientSession | None = None  # Lazy initialization
        self._owns_session = True  # Track if we created the session (for cleanup)

        # Supported tile sizes for the segmentation model
        self.tile_sizes = [256, 512, 1024, 2048]
        self.max_concurrent = max_concurrent  # Store for later access
        self.semaphore = asyncio.Semaphore(max_concurrent)  # Limit concurrent requests
        self.retry_delays = [1, 2, 5]  # Exponential backoff delays in seconds
        self.timeout = timeout

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager."
            )
        return self._session

    async def __aenter__(self):
        if self._session is None:
            self._session = ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def __call__(
        self,
        input: OpenSlide
        | NDArray[np.uint8]
        | Iterable[Tile]
        | list[NDArray[np.uint8]]
        | AsyncIterable[NDArray[np.uint8]],
        endpoint: str = "/lsp-detr",
        *,
        format: str = "raw",  # "raw" for bytes, "json" for JSON payload
        stream_mode: str = "auto",
    ) -> Result | list[Result] | AsyncIterable[Result]:
        from rationai.segmentation.streaming import (
            stream_tiles,
            stream_tiles_ordered,
        )

        # Handle streaming input (async iterator)
        if hasattr(input, "__aiter__") or stream_mode in {"unordered", "ordered"}:
            tile_gen = cast("AsyncIterable[NDArray[np.uint8]]", input)
            if stream_mode == "ordered":
                return stream_tiles_ordered(
                    self.session,
                    tile_gen,
                    endpoint=endpoint,
                    base_url=self._base_url,
                    format=format,
                )
            else:
                return stream_tiles(
                    self.session,
                    tile_gen,
                    endpoint=endpoint,
                    base_url=self._base_url,
                    format=format,
                )

        # Handle list of tiles (either Tile dicts or NDArrays)
        if isinstance(input, list):
            if input and isinstance(input[0], dict):
                ndarray_list = [tile["data"] for tile in input]  # type: ignore[index]
            else:
                ndarray_list = cast("list[NDArray[np.uint8]]", input)

            # Process all tiles concurrently
            return await asyncio.gather(
                *[
                    self._process_tile_with_retry(img, endpoint, format)
                    for img in ndarray_list
                ]
            )

        is_openslide = isinstance(input, OpenSlide) or (
            hasattr(input, "__class__") and input.__class__.__name__ == "OpenSlide"
        )
        if is_openslide:
            raise NotImplementedError("OpenSlide input not yet supported")

        if isinstance(input, np.ndarray):
            return await self._process_ndarray(input, endpoint, format)

        if isinstance(input, (list, tuple)) and all(isinstance(x, dict) for x in input):
            return await self._process_tiles(input, endpoint, format)

        raise TypeError(f"Unsupported input type: {type(input)}")

    async def _process_ndarray(
        self, image: NDArray[np.uint8], endpoint: str, format: str
    ) -> Result | list[Result]:
        """Process a numpy array image (small images directly, large images via tiling)."""
        h, w = image.shape[:2]
        max_tile_size = self.tile_sizes[-1]  # 2048

        # Small enough to process directly
        if h <= max_tile_size and w <= max_tile_size:
            return await self._process_tile_with_retry(image, endpoint, format)

        # Large image - split into tiles
        tiles: list[Tile] = []
        for y, x in grid_tiles(
            slide_extent=(h, w),
            tile_extent=(max_tile_size, max_tile_size),
            stride=(max_tile_size, max_tile_size),
            last="shift",  # Shift last tile to avoid partial tiles
        ):
            tile_data = image[y : y + max_tile_size, x : x + max_tile_size]
            tiles.append(Tile(data=tile_data, x=x, y=y))

        results = await asyncio.gather(
            *[
                self._process_tile_with_retry(tile["data"], endpoint, format)
                for tile in tiles
            ]
        )
        return results

    async def _process_tiles(
        self, tiles: Iterable[Tile], endpoint: str, format: str
    ) -> list[Result]:
        return await asyncio.gather(
            *[
                self._process_tile_with_retry(tile["data"], endpoint, format)
                for tile in tiles
            ]
        )

    async def _process_tile_with_retry(
        self, tile: NDArray[np.uint8], endpoint: str, format: str
    ) -> Result:
        """Process a single tile with retries and error handling."""
        async with self.semaphore:  # Limit concurrent requests
            last_exception = None
            max_attempts = len(self.retry_delays) + 1  # Initial attempt + retries
            # Retry with exponential backoff
            for attempt, delay in enumerate([0, *self.retry_delays]):
                if attempt > 0:
                    logger.warning(
                        "Retrying after %ss delay (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        max_attempts,
                    )
                    await asyncio.sleep(delay)
                try:
                    return await asyncio.wait_for(
                        self._process_tile(tile, endpoint, format), timeout=self.timeout
                    )
                except TimeoutError as e:
                    last_exception = e
                    if attempt == len(self.retry_delays):  # Last attempt
                        raise Exception(
                            f"Request timed out after {max_attempts} attempts"
                        ) from e
                    continue
                except Exception as e:
                    last_exception = e
                    if attempt == len(self.retry_delays):  # Last attempt
                        raise Exception(f"Failed after {max_attempts} attempts") from e
                    logger.warning("Attempt %d failed: %s", attempt + 1, e)
                    continue
            raise Exception(f"Failed after {max_attempts} attempts") from last_exception

    def _resize_tile_to_target(
        self, tile: NDArray[np.uint8], target_size: int
    ) -> NDArray[np.uint8]:
        """Resize tile to target size by cropping and/or padding.

        Args:
            tile: Input image tile
            target_size: Target dimension (square)

        Returns:
            Resized tile with shape (target_size, target_size, channels)
        """
        h, w = tile.shape[:2]

        # Crop if tile is larger than target
        if h > target_size or w > target_size:
            tile = tile[:target_size, :target_size]
            h, w = tile.shape[:2]

        # Pad if tile is smaller than target
        if h < target_size or w < target_size:
            tile = np.pad(
                tile,
                ((0, target_size - h), (0, target_size - w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        return tile

    async def _process_tile(
        self, tile: NDArray[np.uint8], endpoint: str, format: str
    ) -> Result:
        """Process a single tile through the segmentation API.

        Supports two formats:
        - raw: Raw bytes with tile size in URL (for DETR models)
        - json: JSON payload with preprocessing (for models like prostate)
        """
        # Ensure endpoint has leading slash
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"

        if format == "raw":
            # Raw bytes format: dynamic tile size selection
            h, w = tile.shape[:2]
            max_dim = max(h, w)

            # Find smallest tile size that fits the image
            tile_size = self.tile_sizes[-1]  # Default to largest
            for size in self.tile_sizes:
                if size >= max_dim:
                    tile_size = size
                    break

            tile = self._resize_tile_to_target(tile, tile_size)

            async with self.session.post(
                f"{self._base_url}{endpoint}/{tile_size}",
                data=tile.tobytes(),
            ) as response:
                response.raise_for_status()
                return await response.json()

        elif format == "json":
            # JSON format: preprocess and send as JSON
            payload = self._preprocess_for_json(tile)

            async with self.session.post(
                f"{self._base_url}{endpoint}",
                json=payload,
            ) as response:
                response.raise_for_status()
                return await response.json()

        else:
            raise ValueError(f"Unsupported format: {format}. Must be 'raw' or 'json'")

    def _preprocess_for_json(self, tile: NDArray[np.uint8]) -> dict:
        """Preprocess image for JSON-based models (e.g., prostate).

        Resizes to 224x224, normalizes to [0,1], and converts to NCHW format.
        """
        # Resize to 224x224 (expected by prostate model)
        if tile.shape[:2] != (224, 224):
            tile = np.array(
                Image.fromarray(tile).resize((224, 224), Image.Resampling.BILINEAR)
            )

        # Convert to NCHW format (batch, channels, height, width)
        tile_nchw = tile.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

        # Normalize to [0, 1]
        tile_nchw /= 255.0

        return {"input": tile_nchw.tolist()}
