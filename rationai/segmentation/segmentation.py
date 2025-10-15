import asyncio
from typing import AsyncIterable, Iterable, List, Literal, TypedDict, Union, cast

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray
from openslide import OpenSlide

from rationai.resources.tilers import grid_tiles
from rationai.segmentation.batch import batch_process
from rationai.segmentation.streaming import (
    stream_tiles,
    stream_tiles_batched,
    stream_tiles_ordered,
)


class Tile(TypedDict):
    data: NDArray[np.uint8]
    x: int
    y: int


class Result(TypedDict):
    polygons: list[list[list[float]]]
    embeddings: list[list[float]]


class AsyncNucleiSegmentation:
    """
    Asynchronous nuclei segmentation for in-memory data.
    Supports:
      - Single NDArray images
      - Large images split into tiles
      - Iterable of Tile dictionaries
    """

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
        stream_mode: Literal["auto", "unordered", "ordered", "batched"] = "auto",
    ) -> Union[Result, List[Result], AsyncIterable[Result]]:
        """
        Auto-selects segmentation mode based on input type or explicit stream_mode.
        """

        # --- Streaming modes ---
        if hasattr(input, "__aiter__") or stream_mode in {
            "unordered",
            "ordered",
            "batched",
        }:
            tile_gen = cast(AsyncIterable[NDArray[np.uint8]], input)

            if stream_mode == "ordered":
                return stream_tiles_ordered(self.session, tile_gen, model=model)
            elif stream_mode == "batched":
                return stream_tiles_batched(self.session, tile_gen, model=model)
            else:
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
        if isinstance(input, OpenSlide):
            raise NotImplementedError("OpenSlide input not yet supported.")

        # --- Single NDArray ---
        if isinstance(input, np.ndarray):
            return await self._process_ndarray(input, model)

        # --- Iterable of Tile dicts ---
        if isinstance(input, Iterable):
            return await self._process_tiles(input, model)

        raise TypeError(f"Unsupported input type: {type(input)}")

    async def _process_ndarray(
        self, image: NDArray[np.uint8], model: Literal["lsp-detr"]
    ) -> Result | list[Result]:
        """Process a single image or split into tiles if too large."""
        h, w = image.shape[:2]
        max_tile_size = self.tile_sizes[-1]

        if h <= max_tile_size and w <= max_tile_size:
            return await self._process_tile_with_retry(image, model)

        # Split large image into tiles
        tiles: list[Tile] = []
        for y, x in grid_tiles(
            slide_extent=(h, w),
            tile_extent=(max_tile_size, max_tile_size),
            stride=(max_tile_size, max_tile_size),
            last="shift",
        ):
            tile_data = image[y : y + max_tile_size, x : x + max_tile_size]
            tiles.append({"data": tile_data, "x": x, "y": y})

        # Process all tiles concurrently with semaphore
        results = await asyncio.gather(
            *[self._process_tile_with_retry(tile["data"], model) for tile in tiles]
        )
        return results

    async def _process_tiles(
        self, tiles: Iterable[Tile], model: Literal["lsp-detr"]
    ) -> list[Result]:
        """Process an iterable of Tile dictionaries."""
        return await asyncio.gather(
            *[self._process_tile_with_retry(tile["data"], model) for tile in tiles]
        )

    async def _process_tile_with_retry(
        self, tile: NDArray[np.uint8], model: Literal["lsp-detr"]
    ) -> Result:
        """Process tile with retry logic and error handling."""
        async with self.semaphore:
            for attempt, delay in enumerate([0] + self.retry_delays):
                if attempt > 0:
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
                        raise Exception(
                            f"Failed after {len(self.retry_delays)} retries: {str(e)}"
                        )
                    continue

            raise Exception("Unexpected error in retry logic")

    async def _process_tile(
        self, tile: NDArray[np.uint8], model: Literal["lsp-detr"]
    ) -> Result:
        """Pad tile and send to server for segmentation."""
        h, w = tile.shape[:2]

        tile_size = next(
            (size for size in self.tile_sizes if size >= max(h, w)),
            self.tile_sizes[-1],
        )

        tile = np.pad(
            tile,
            ((0, tile_size - h), (0, tile_size - w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        async with self.session.post(
            f"/{model}/{tile_size}",
            data=tile.tobytes(),
        ) as response:
            response.raise_for_status()
            return await response.json()
