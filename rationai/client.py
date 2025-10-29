from __future__ import annotations

import logging
from pathlib import Path

from rationai.segmentation.core import AsyncNucleiSegmentation


logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for a specific model endpoint.

    Note: input_size, num_channels, and pixel_range are reserved for future use.
    Currently only endpoint is used by the client.
    """

    def __init__(
        self,
        endpoint: str,
        input_size: tuple[int, int] = (512, 512),
        num_channels: int = 3,
        pixel_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.endpoint = endpoint
        self.input_size = input_size
        self.num_channels = num_channels
        self.pixel_range = pixel_range


MODELS: dict[str, ModelConfig] = {
    "prostate": ModelConfig(
        endpoint="/prostate",
        input_size=(512, 512),
        num_channels=3,
        pixel_range=(0.0, 1.0),
    ),
    "nuclei": ModelConfig(
        endpoint="/lsp-detr",
        input_size=(512, 512),
        num_channels=3,
        pixel_range=(0.0, 255.0),
    ),
}


class Model:
    """Wrapper for a specific model endpoint.

    Delegates all processing to core.py.
    """

    def __init__(self, core_client: AsyncNucleiSegmentation, endpoint: str):
        self._core = core_client
        self._endpoint = endpoint

    async def predict(self, input):
        """Send image to model. Delegates to core.py for all processing."""
        if isinstance(input, (str, Path)):
            import numpy as np
            from PIL import Image

            input = np.array(Image.open(input))

        return await self._core(input, model=self._endpoint)

    async def stream(self, input, stream_mode: str = "unordered"):
        """Stream predictions. Delegates to core.py."""
        return await self._core(input, model=self._endpoint, stream_mode=stream_mode)


class RationAIClient:
    """Async client for multiple RationAI models.

    Thin wrapper for connection management only.
    All processing logic is in core.py.

    Usage:
        async with RationAIClient("http://127.0.0.1:8001") as client:
            prostate = client.model("prostate")
            result = await prostate.predict(image)
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        *,
        timeout: int = 30,
        max_concurrent: int = 5,
    ):
        self._core = AsyncNucleiSegmentation(
            base_url=base_url,
            timeout=timeout,
            max_concurrent=max_concurrent,
        )

    async def __aenter__(self) -> RationAIClient:
        await self._core.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._core.__aexit__(exc_type, exc, tb)

    async def close(self) -> None:
        """Close the HTTP session."""
        await self._core.close()

    def model(self, name: str) -> Model:
        """Return a Model object for a given model name or endpoint.

        Args:
            name: Pre-defined model name (from MODELS) or custom endpoint path

        Returns:
            Model object that delegates to core.py for all processing

        Examples:
            # Use pre-configured model
            prostate = client.model("prostate")

            # Use custom endpoint
            custom = client.model("/my-custom-model")
        """
        endpoint = MODELS[name].endpoint if name in MODELS else name
        return Model(self._core, endpoint)


if __name__ == "__main__":
    import asyncio

    async def main():
        """Example usage of RationAIClient."""
        import numpy as np

        # Example: prostate segmentation
        async with RationAIClient("http://127.0.0.1:8001") as client:
            prostate = client.model("prostate")
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            print(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
            result = await prostate.predict(test_image)
            print(f"Prostate result: {result}")

    asyncio.run(main())
