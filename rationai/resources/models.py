import numpy as np
from aiohttp import ClientTimeout
from numpy.typing import NDArray

from rationai._resource import APIResource, AsyncAPIResource


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


class Models(APIResource):
    def classify_image(
        self, model: str, image: NDArray[np.uint8], timeout: float | None = None
    ):
        response = self._post(model, json={"input": image.tolist()}, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def embed_image(
        self, model: str, image: NDArray[np.uint8], timeout: float | None = None
    ):
        response = self._post(model, json={"input": image.tolist()}, timeout=timeout)
        response.raise_for_status()
        return response.json()
