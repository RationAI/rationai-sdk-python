import asyncio
from typing import AsyncIterable, Iterable, List, Literal, TypedDict, Union, cast

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray
from openslide import OpenSlide

from rationai.resources.tilers import grid_tiles
from rationai.segmentation.batch import batch_process
from rationai.segmentation.types import Result


class Tile(TypedDict):
    data: NDArray[np.uint8]
    x: int
    y: int


class AsyncNucleiSegmentation:
    def __init__(self, session: ClientSession, max_concurrent: int = 5):
        self.session = session
        self.tile_sizes = [256, 512, 1024, 2048]
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.retry_delays = [10, 20, 30]  # seconds
        self.timeout = 60  # seconds

    async def __call__(
        self,
        input: Union[
            OpenSlide,
            NDArray[np.uint8],
            Iterable[Tile],
            List[NDArray[np.uint8]],
            AsyncIterable[NDArray[np.uint8]],
        ],
        model: Literal["lsp-detr"] = "lsp-detr",
        *,
        stream_mode: Literal["auto", "unordered", "ordered"] = "auto",
    ) -> Union[Result, List[Result], AsyncIterable[Result]]:
        # --- Streaming modes ---
        from rationai.segmentation.streaming import (
            stream_tiles,
            stream_tiles_ordered,
        )

        if hasattr(input, "__aiter__") or stream_mode in {
            "unordered",
            "ordered",
        }:
            tile_gen = cast(AsyncIterable[NDArray[np.uint8]], input)
            if stream_mode == "ordered":
                return stream_tiles_ordered(self.session, tile_gen, model=model)
            else:  # unordered or auto
                return stream_tiles(self.session, tile_gen, model=model)
        # --- Batched mode (explicit or auto-detect) ---
        if isinstance(input, list):
            # Convert list[Tile] → list[NDArray]
            if input and isinstance(input[0], dict):
                ndarray_list = [tile["data"] for tile in input]  # type: ignore
            else:
                ndarray_list = cast(List[NDArray[np.uint8]], input)
            return await batch_process(self.session, ndarray_list, model=model)
        # --- OpenSlide input ---
        is_openslide = isinstance(input, OpenSlide) or (
            hasattr(input, "__class__") and input.__class__.__name__ == "OpenSlide"
        )
        if is_openslide:
            raise NotImplementedError("OpenSlide input not yet supported")
        # --- Single NDArray ---
        if isinstance(input, np.ndarray):
            return await self._process_ndarray(input, model)
        # --- Iterable of Tile dicts ---
        if isinstance(input, (list, tuple)) and all(isinstance(x, dict) for x in input):
            return await self._process_tiles(input, model)
        raise TypeError(f"Unsupported input type: {type(input)}")

    async def _process_ndarray(
        self, image: NDArray[np.uint8], model: Literal["lsp-detr"]
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

    async def _process_tiles(
        self, tiles: Iterable[Tile], model: Literal["lsp-detr"]
    ) -> list[Result]:
        return await asyncio.gather(
            *[self._process_tile_with_retry(tile["data"], model) for tile in tiles]
        )

    async def _process_tile_with_retry(
        self, tile: NDArray[np.uint8], model: Literal["lsp-detr"]
    ) -> Result:
        """Process a single tile with retries and error handling."""
        async with self.semaphore:
            for attempt, delay in enumerate([0] + self.retry_delays):
                if attempt > 0:
                    print(f"Retrying after {delay}s delay (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                try:
                    return await asyncio.wait_for(
                        self._process_tile(tile, model), timeout=self.timeout
                    )
                except asyncio.TimeoutError:
                    if attempt == len(self.retry_delays):
                        raise Exception(
                            f"Request timed out after {len(self.retry_delays)} retries"
                        )
                    continue
                except Exception as e:
                    if attempt == len(self.retry_delays):
                        raise  # Re-raise the original error for better debugging
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    continue
            raise Exception(f"Failed after {len(self.retry_delays)} retries")

    async def _process_tile(
        self, tile: NDArray[np.uint8], model: Literal["lsp-detr"]
    ) -> Result:
        """Process a single tile through the segmentation API."""
        h, w = tile.shape[:2]
        # Use minimum tile size needed or max available
        tile_size = min(
            next(
                (size for size in self.tile_sizes if size >= max(h, w)),
                self.tile_sizes[-1],
            ),
            self.tile_sizes[-1],
        )

        # If tile is larger than max size, crop it
        if h > tile_size or w > tile_size:
            tile = tile[:tile_size, :tile_size]
            h, w = tile.shape[:2]

        # Now pad to the target size if needed
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
                response.raise_for_status()  # This can raise ClientResponseError
                return await response.json()
        except Exception:
            # Pass through the original error for better error handling
            raise
