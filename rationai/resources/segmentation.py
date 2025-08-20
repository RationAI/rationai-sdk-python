from typing import Iterable, Literal, TypedDict, overload

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray
from openslide import OpenSlide


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
    ) -> Result: ...

    @overload
    async def __call__(
        self, input: Iterable[Tile], model: Literal["lsp-detr"]
    ) -> Result: ...

    async def __call__(
        self, input: NDArray[np.uint8] | Iterable[Tile], model: Literal["lsp-detr"]
    ) -> Result:
        if isinstance(input, np.ndarray):
            return await self._process_tile(input, model)

        for tile in input:
            await self._process_tile(tile["data"], model)

    async def _process_tile(
        self, tile: NDArray[np.uint8], model: Literal["lsp-detr"]
    ) -> Result:
        h, w = tile.shape[:2]

        tile_size = next(
            (size for size in self.tile_sizes if size >= max(h, w)), self.tile_sizes[-1]
        )

        tile = np.pad(
            tile,
            ((0, tile_size - h), (0, tile_size - w), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        response = await self.session.post(f"/{model}/{tile_size}", data=tile.tobytes())
        response.raise_for_status()
        return response.json()
