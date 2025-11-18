import numpy as np
from aiohttp import ClientTimeout
from numpy.typing import NDArray

from rationai._resource import AsyncAPIResource


class AsyncModels(AsyncAPIResource):
    async def classify_image(
        self, model: str, image: NDArray[np.uint8], timeout: ClientTimeout | None = None
    ):
        async with self._post(
            model, json={"input": image}, timeout=timeout
        ) as response:
            return await response.json()

    async def embed_image(
        self, model: str, image: NDArray[np.uint8], timeout: ClientTimeout | None = None
    ):
        async with self._post(
            model, json={"input": image}, timeout=timeout
        ) as response:
            return await response.json()
