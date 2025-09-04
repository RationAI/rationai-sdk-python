from typing import Iterable, Literal, TypedDict, overload

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray
from openslide import OpenSlide

from rationai.resources.tilers import grid_tiles


class Tile(TypedDict):
    data: NDArray[np.uint8]
    x: int
    y: int


class Result(TypedDict):
    polygons: list[list[list[float]]]
    embeddings: list[list[float]]


class AsyncNucleiSegmentation:
    def __init__(self, session: ClientSession):
        self.session = session
        self.tile_sizes = [256, 512, 1024, 2048]

    @overload
    async def __call__(
        self, input: OpenSlide, model: Literal["lsp-detr"]
    ) -> Result: ...

    @overload
    async def __call__(
        self, input: NDArray[np.uint8], model: Literal["lsp-detr"]
    ) -> Result | list[Result]: ...

    @overload
    async def __call__(
        self, input: Iterable[Tile], model: Literal["lsp-detr"]
    ) -> list[Result]: ...

    async def __call__(
        self,
        input: OpenSlide | NDArray[np.uint8] | Iterable[Tile],
        model: Literal["lsp-detr"],
    ) -> Result | list[Result]:
        if isinstance(input, np.ndarray):
            h, w = input.shape[:2]
            max_tile_size = self.tile_sizes[-1]

            if h <= max_tile_size and w <= max_tile_size:
                return await self._process_tile(input, model)  # compatible with model

            tiles: list[Tile] = []  # picture larger than max tile size
            for coord in grid_tiles(
                slide_extent=(h, w),
                tile_extent=(max_tile_size, max_tile_size),
                stride=(max_tile_size, max_tile_size),
                last="shift",
            ):
                y, x = coord
                tile_data = input[y : y + max_tile_size, x : x + max_tile_size]
                tiles.append({"data": tile_data, "x": x, "y": y})

            # Process all tiles and return a list of results
            results: list[Result] = []
            for tile in tiles:
                results.append(await self._process_tile(tile["data"], model))
            return results

        if isinstance(input, OpenSlide):
            raise NotImplementedError("Processing OpenSlide input is not implemented.")

        # iterable of tiles
        results: list[Result] = []
        for tile in input:
            results.append(await self._process_tile(tile["data"], model))
        return results

    async def _process_tile(
        self, tile: NDArray[np.uint8], model: Literal["lsp-detr"]
    ) -> Result:
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
            f"/{model}/{tile_size}", data=tile.tobytes()
        ) as response:
            response.raise_for_status()
            return await response.json()
