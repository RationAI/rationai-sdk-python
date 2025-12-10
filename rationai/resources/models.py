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
        data = image.tobytes()
        compressed_data = lz4.frame.compress(data)
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

        data = image.tobytes()
        compressed_data = lz4.frame.compress(data)
        response = self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()

        return np.frombuffer(
            lz4.frame.decompress(response.content), dtype=np.float16
        ).reshape(-1, h, w)


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
        data = image.tobytes()
        compressed_data = lz4.frame.compress(data)
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

        data = image.tobytes()
        compressed_data = lz4.frame.compress(data)
        response = await self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()

        return np.frombuffer(
            lz4.frame.decompress(response.content), dtype=np.float16
        ).reshape(-1, h, w)
