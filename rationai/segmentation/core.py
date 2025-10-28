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
        base_url: str = "http://127.0.0.1:8001",
        *,
        timeout: float = 30,
        max_concurrent: int = 5,
    ):
        self._base_url = base_url.rstrip("/")  # Remove trailing slash
        self._session: ClientSession | None = None  # Lazy initialization
        self._owns_session = True  # Track if we created the session (for cleanup)

        # Supported tile sizes for the segmentation model
        self.tile_sizes = [256, 512, 1024, 2048]
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
        model: str = "lsp-detr",
        *,
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
                return stream_tiles_ordered(self.session, tile_gen, model=model)
            else:
                return stream_tiles(self.session, tile_gen, model=model)

        # Handle list of tiles (either Tile dicts or NDArrays)
        if isinstance(input, list):
            if input and isinstance(input[0], dict):
                ndarray_list = [tile["data"] for tile in input]  # type: ignore[index]
            else:
                ndarray_list = cast("list[NDArray[np.uint8]]", input)

            # Process all tiles concurrently
            return await asyncio.gather(
                *[self._process_tile_with_retry(img, model) for img in ndarray_list]
            )

        is_openslide = isinstance(input, OpenSlide) or (
            hasattr(input, "__class__") and input.__class__.__name__ == "OpenSlide"
        )
        if is_openslide:
            raise NotImplementedError("OpenSlide input not yet supported")

        if isinstance(input, np.ndarray):
            return await self._process_ndarray(input, model)

        if isinstance(input, (list, tuple)) and all(isinstance(x, dict) for x in input):
            return await self._process_tiles(input, model)

        raise TypeError(f"Unsupported input type: {type(input)}")

    async def _process_ndarray(
        self, image: NDArray[np.uint8], model: str
    ) -> Result | list[Result]:
        """Process a numpy array image (small images directly, large images via tiling)."""
        h, w = image.shape[:2]
        max_tile_size = self.tile_sizes[-1]  # 2048

        # Small enough to process directly
        if h <= max_tile_size and w <= max_tile_size:
            return await self._process_tile_with_retry(image, model)

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
            *[self._process_tile_with_retry(tile["data"], model) for tile in tiles]
        )
        return results

    async def _process_tiles(self, tiles: Iterable[Tile], model: str) -> list[Result]:
        return await asyncio.gather(
            *[self._process_tile_with_retry(tile["data"], model) for tile in tiles]
        )

    async def _process_tile_with_retry(
        self, tile: NDArray[np.uint8], model: str
    ) -> Result:
        """Process a single tile with retries and error handling."""
        async with self.semaphore:  # Limit concurrent requests
            last_exception = None
            # Retry with exponential backoff
            for attempt, delay in enumerate([0, *self.retry_delays]):
                if attempt > 0:
                    logger.warning(
                        "Retrying after %ss delay (attempt %d)", delay, attempt + 1
                    )
                    await asyncio.sleep(delay)
                try:
                    return await asyncio.wait_for(
                        self._process_tile(tile, model), timeout=self.timeout
                    )
                except TimeoutError as e:
                    last_exception = e
                    if attempt == len(self.retry_delays):
                        raise Exception(
                            f"Request timed out after {len(self.retry_delays)} retries"
                        ) from e
                    continue
                except Exception as e:
                    last_exception = e
                    if attempt == len(self.retry_delays):
                        raise Exception(
                            f"Failed after {len(self.retry_delays)} retries"
                        ) from e
                    logger.warning("Attempt %d failed: %s", attempt + 1, e)
                    continue
            raise Exception(
                f"Failed after {len(self.retry_delays)} retries"
            ) from last_exception

    async def _process_tile(self, tile: NDArray[np.uint8], model: str) -> Result:
        """Process a single tile through the segmentation API.

        Supports two formats:
        - DETR models (nuclei): raw bytes with tile size in URL
        - JSON models (prostate): JSON payload with preprocessing
        """
        # Check if this is a DETR model (uses raw bytes)
        is_detr = "detr" in model.lower()

        if is_detr:
            # DETR format: raw bytes with tile size in URL
            h, w = tile.shape[:2]
            # Select smallest tile size that fits the image dimensions
            tile_size = min(
                next(
                    (size for size in self.tile_sizes if size >= max(h, w)),
                    self.tile_sizes[-1],
                ),
                self.tile_sizes[-1],
            )

            # Crop if tile is larger than selected tile_size
            if h > tile_size or w > tile_size:
                tile = tile[:tile_size, :tile_size]
                h, w = tile.shape[:2]

            # Pad if tile is smaller than selected tile_size
            if h < tile_size or w < tile_size:
                tile = np.pad(
                    tile,
                    ((0, tile_size - h), (0, tile_size - w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            try:
                async with self.session.post(
                    f"{self._base_url}/{model.lstrip('/')}/{tile_size}",
                    data=tile.tobytes(),
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception:
                raise

        else:
            # JSON format: preprocess and send as JSON
            payload = self._preprocess_for_json(tile)

            try:
                async with self.session.post(
                    f"{self._base_url}/{model.lstrip('/')}",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception:
                raise

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
