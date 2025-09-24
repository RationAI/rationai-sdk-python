# streaming.py
from typing import AsyncGenerator, AsyncIterable, Literal

import numpy as np
from aiohttp import ClientSession
from numpy.typing import NDArray

from .segmentation import AsyncNucleiSegmentation, Result


async def stream_tiles(
    session: ClientSession,
    tile_generator: AsyncIterable[NDArray[np.uint8]],
    model: Literal["lsp-detr"] = "lsp-detr",
) -> AsyncGenerator[Result, None]:
    """
    Stream tiles to the segmentation server asynchronously.
    """
    segmenter = AsyncNucleiSegmentation(session)

    async for tile in tile_generator:
        yield await segmenter._process_tile(tile, model)
