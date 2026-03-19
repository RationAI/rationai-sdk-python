from typing import cast

import lz4.frame
import numpy as np
from httpx import USE_CLIENT_DEFAULT
from httpx._client import UseClientDefault
from httpx._types import TimeoutTypes
from numpy.typing import NDArray
from PIL.Image import Image

from rationai._resource import APIResource, AsyncAPIResource


class Models(APIResource):
    def classify_image(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> float | dict[str, float]:
        """Classify an image using the specified model.

        Args:
            model: The name of the model to use for classification.
            image: The image to classify. It must be uint8 RGB image.
            timeout: Optional timeout for the request.

        Returns:
            (float | dict[str, float]): The classification result as a single float
                (for binary classification) or probabilities for each class.
        """
        compressed_data = lz4.frame.compress(image.tobytes())
        response = self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def segment_image(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> NDArray[np.float16]:
        """Segment an image using the specified model.

        Args:
            model: The name of the model to use for segmentation.
            image: The image to segment. It must be uint8 RGB image.
            timeout: Optional timeout for the request.

        Returns:
            NDArray[np.float16]: The segmentation result as a numpy array of float16 values.
                The shape of the array is (num_classes, height, width).
        """
        if isinstance(image, Image):
            w, h = image.size
        else:
            h, w = image.shape[:2]

        compressed_data = lz4.frame.compress(image.tobytes())
        response = self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()

        return np.frombuffer(
            lz4.frame.decompress(response.content), dtype=np.float16
        ).reshape(-1, h, w)

    def embed_image[DType: np.generic](
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        output_dtype: type[DType] = np.float32,  # type: ignore[assignment]
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> NDArray[DType]:
        """Compute an embedding vector for an image using the specified model.

        Args:
            model: The name of the model to use for embedding.
            image: The image to embed. It must be uint8 RGB image.
            output_dtype: Output numpy dtype for embeddings (e.g. np.float16, np.float32).
            timeout: Optional timeout for the request.

        Returns:
            NDArray[DType]: The embedding vector as a 1-D numpy array.
        """
        compressed_data = lz4.frame.compress(image.tobytes())
        response = self._post(
            model,
            data=compressed_data,
            headers={"x-output-dtype": np.dtype(output_dtype).name},
            timeout=timeout,
        )
        response.raise_for_status()

        payload = lz4.frame.decompress(response.content)
        embedding = np.frombuffer(payload, dtype=output_dtype)

        response_shape = response.headers.get("x-output-shape")
        if response_shape:
            embedding = embedding.reshape(eval(response_shape))

        return cast("NDArray[DType]", embedding)


class AsyncModels(AsyncAPIResource):
    async def classify_image(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> float | dict[str, float]:
        """Classify an image using the specified model.

        Args:
            model: The name of the model to use for classification.
            image: The image to classify. It must be uint8 RGB image.
            timeout: Optional timeout for the request.

        Returns:
            (float | dict[str, float]): The classification result as a single float
                (for binary classification) or probabilities for each class.
        """
        compressed_data = lz4.frame.compress(image.tobytes())
        response = await self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()
        return response.json()

    async def segment_image(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> NDArray[np.float16]:
        """Segment an image using the specified model.

        Args:
            model: The name of the model to use for segmentation.
            image: The image to segment. It must be uint8 RGB image.
            timeout: Optional timeout for the request.

        Returns:
            NDArray[np.float16]: The segmentation result as a numpy array of float16 values.
                The shape of the array is (num_classes, height, width).
        """
        if isinstance(image, Image):
            w, h = image.size
        else:
            h, w = image.shape[:2]

        compressed_data = lz4.frame.compress(image.tobytes())
        response = await self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()

        return np.frombuffer(
            lz4.frame.decompress(response.content), dtype=np.float16
        ).reshape(-1, h, w)

    async def embed_image[DType: np.generic](
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        output_dtype: type[DType] = np.float32,  # type: ignore[assignment]
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> NDArray[DType]:
        """Compute an embedding vector for an image using the specified model.

        Args:
            model: The name of the model to use for embedding.
            image: The image to embed. It must be uint8 RGB image.
            output_dtype: Output numpy dtype for embeddings (e.g. np.float16, np.float32).
            timeout: Optional timeout for the request.

        Returns:
            NDArray[DType]: The embedding vector as a 1-D numpy array.
        """
        compressed_data = lz4.frame.compress(image.tobytes())
        response = await self._post(
            model,
            data=compressed_data,
            headers={"x-output-dtype": np.dtype(output_dtype).name},
            timeout=timeout,
        )
        response.raise_for_status()

        payload = lz4.frame.decompress(response.content)
        embedding = np.frombuffer(payload, dtype=output_dtype)

        response_shape = response.headers.get("x-output-shape")
        if response_shape:
            embedding = embedding.reshape(eval(response_shape))

        return cast("NDArray[DType]", embedding)
