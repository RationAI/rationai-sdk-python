import json

import numpy as np
from aiohttp import ClientTimeout
from numpy.typing import NDArray
from PIL import Image

from rationai._resource import APIResource, AsyncAPIResource


def preprocess_image(image: NDArray[np.uint8]) -> list:
    if image.shape[:2] != (224, 224):
        pil_img = Image.fromarray(image)
        image = np.array(pil_img.resize((224, 224), Image.Resampling.BILINEAR))

    tile_nchw = image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
    tile_nchw /= 255.0

    return tile_nchw.tolist()


class AsyncModels(AsyncAPIResource):
    async def classify_image(
        self,
        model: str,
        image: NDArray[np.uint8],
        timeout: ClientTimeout | None = None,
    ):
        processed_data = preprocess_image(image)

        payload = {"input": processed_data}

        async with self._post(
            model,
            json=payload,
            timeout=timeout,
            raise_for_status=False,
        ) as response:
            text = await response.text()

            response.raise_for_status()
            return json.loads(text)

    async def embed_image(
        self,
        model: str,
        image: NDArray[np.uint8],
        timeout: ClientTimeout | None = None,
    ):
        processed_data = preprocess_image(image)
        payload = {"input": processed_data}

        async with self._post(
            model,
            json=payload,
            timeout=timeout,
            raise_for_status=False,
        ) as response:
            text = await response.text()

            response.raise_for_status()
            return json.loads(text)


class Models(APIResource):
    def classify_image(
        self, model: str, image: NDArray[np.uint8], timeout: float | None = None
    ):
        processed_data = preprocess_image(image)
        payload = {"input": processed_data}

        response = self._post(
            model, json=payload, timeout=timeout, raise_for_status=False
        )

        return response.json()

    def embed_image(
        self, model: str, image: NDArray[np.uint8], timeout: float | None = None
    ):
        processed_data = preprocess_image(image)
        payload = {"input": processed_data}

        response = self._post(
            model, json=payload, timeout=timeout, raise_for_status=False
        )

        return response.json()
