import asyncio
import logging
from collections.abc import AsyncIterable, Iterable
from typing import cast

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray
from openslide import OpenSlide
from ratiopath.tiling import grid_tiles

from rationai.segmentation.types import Result, Tile


logger = logging.getLogger(__name__)


class AsyncNucleiSegmentation:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 30,
        max_concurrent: int = 5,
    ):
        self._base_url = base_url.rstrip("/")
        self._session: ClientSession | None = None
        self._owns_session = True

        self.tile_sizes = [256, 512, 1024, 2048]
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.retry_delays = [1, 2, 5]
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
            self._session = ClientSession(base_url=self._base_url)
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

        if hasattr(input, "__aiter__") or stream_mode in {"unordered", "ordered"}:
            tile_gen = cast("AsyncIterable[NDArray[np.uint8]]", input)
            if stream_mode == "ordered":
                return stream_tiles_ordered(self.session, tile_gen, model=model)
            else:
                return stream_tiles(self.session, tile_gen, model=model)

        if isinstance(input, list):
            if input and isinstance(input[0], dict):
                ndarray_list = [tile["data"] for tile in input]  # type: ignore[index]
            else:
                ndarray_list = cast("list[NDArray[np.uint8]]", input)

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
        h, w = image.shape[:2]
        max_tile_size = self.tile_sizes[-1]
        if h <= max_tile_size and w <= max_tile_size:
            return await self._process_tile_with_retry(image, model)

        tiles: list[Tile] = []
        for y, x in grid_tiles(
            slide_extent=(h, w),
            tile_extent=(max_tile_size, max_tile_size),
            stride=(max_tile_size, max_tile_size),
            last="shift",
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
        async with self.semaphore:
            last_exception = None
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
        """Process a single tile through the segmentation API."""
        h, w = tile.shape[:2]
        tile_size = min(
            next(
                (size for size in self.tile_sizes if size >= max(h, w)),
                self.tile_sizes[-1],
            ),
            self.tile_sizes[-1],
        )

        if h > tile_size or w > tile_size:
            tile = tile[:tile_size, :tile_size]
            h, w = tile.shape[:2]

        if h < tile_size or w < tile_size:
            tile = np.pad(
                tile,
                ((0, tile_size - h), (0, tile_size - w), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        try:
            async with self.session.post(
                f"/{model}/{tile_size}",
                data=tile.tobytes(),
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception:
            raise
