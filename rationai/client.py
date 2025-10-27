from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Union

import aiohttp
import numpy as np
from PIL import Image


class ModelConfig:
    """Configuration for a specific model."""

    def __init__(
        self,
        endpoint: str,
        input_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        preprocessor: Callable[[np.ndarray], Any] | None = None,
    ):
        self.endpoint = endpoint
        self.input_size = input_size
        self.normalize = normalize
        self.preprocessor = preprocessor


MODELS: dict[str, ModelConfig] = {
    "prostate": ModelConfig("/prostate", input_size=(224, 224)),
    "nuclei": ModelConfig("/lsp-detr", input_size=(256, 256), normalize=False),
}


class Model:
    """Wrapper for a specific model endpoint."""

    def __init__(self, client: RationAIClient, config: ModelConfig):
        self._client = client
        self._config = config

    async def predict(
        self, image: Union[np.ndarray, str, Path], tile_size: int | None = None
    ) -> dict:
        """Send image to the model endpoint."""
        return await self._client._predict(image, self._config, tile_size=tile_size)


class RationAIClient:
    """
    Async client for multiple RationAI models.

    Usage:
        async with RationAIClient("http://127.0.0.1:8001") as client:
            prostate_model = client.model("prostate")
            result = await prostate_model.predict(image)
    """

    DEFAULT_TIMEOUT = 30
    DEFAULT_RETRIES = [1, 2, 5]

    def __init__(
        self, base_url: str, *, timeout: int = DEFAULT_TIMEOUT, max_concurrent: int = 5
    ):
        self.base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._retries = self.DEFAULT_RETRIES

    async def __aenter__(self) -> RationAIClient:
        self._session = aiohttp.ClientSession(base_url=self.base_url)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def model(self, name: str, config: ModelConfig | None = None) -> Model:
        """
        Return a Model object for a given model name or endpoint.

        Args:
            name: Pre-defined model name (from MODELS) or custom endpoint path
            config: Optional custom ModelConfig (overrides MODELS lookup and defaults)

        Returns:
            Model object ready for prediction

        Examples:
            # Use pre-configured model
            nuclei = client.model("nuclei")

            # Use custom endpoint with defaults
            custom = client.model("/my-custom-model")

            # Use custom endpoint with specific config
            advanced = client.model("custom", ModelConfig("/endpoint", input_size=(512, 512)))
        """
        if not self._session:
            raise RuntimeError("Client session not initialized. Use `async with`.")

        if config is not None:
            return Model(self, config)

        if name in MODELS:
            return Model(self, MODELS[name])

        return Model(self, ModelConfig(endpoint=name))

    async def _predict(
        self,
        image: Union[np.ndarray, str, Path],
        config: ModelConfig,
        *,
        tile_size: int | None = None,
    ) -> dict:
        if not self._session:
            raise RuntimeError("Client session not initialized. Use `async with`.")

        img = self._load_image(image)
        payload = self._preprocess_image(img, config)

        endpoint = config.endpoint
        if tile_size and "detr" in endpoint.lower():
            endpoint = f"{endpoint}/{tile_size}"

        async with self._semaphore:
            last_error = None

            for attempt, delay in enumerate([0] + self._retries):
                if delay > 0:
                    await asyncio.sleep(delay)

                try:
                    async with self._session.post(
                        endpoint, json=payload, timeout=self._timeout
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()

                except aiohttp.ClientConnectorError as e:
                    raise RuntimeError(
                        f"Cannot connect to {self.base_url}. Is the server running?"
                    ) from e

                except aiohttp.ClientResponseError as e:
                    if 400 <= e.status < 500:
                        raise RuntimeError(
                            f"Client error {e.status}: {e.message}"
                        ) from e

                    last_error = e
                    if attempt < len(self._retries):
                        print(f"[Warning] Attempt {attempt + 1} failed: {e}")
                    continue

                except Exception as e:
                    last_error = e
                    if attempt < len(self._retries):
                        print(f"[Warning] Attempt {attempt + 1} failed: {e}")
                    continue

            raise RuntimeError(
                f"All {len(self._retries) + 1} attempts failed"
            ) from last_error

    @staticmethod
    def _load_image(image: Union[np.ndarray, str, Path]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {image.shape}")
        return image

    @staticmethod
    def _preprocess_image(image: np.ndarray, config: ModelConfig) -> dict:
        if config.preprocessor:
            return {"input": config.preprocessor(image)}

        # Default preprocessing
        # NOTE: image.shape[:2] = (H, W), config.input_size = (W, H)
        if (
            image.shape[:2] != config.input_size[::-1]
        ):  # Reverse to compare (H,W) with (H,W)
            image = np.array(
                Image.fromarray(image).resize(
                    config.input_size, Image.Resampling.BILINEAR
                )
            )
        image = image.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
        if config.normalize:
            image /= 255.0
        return {"input": image.tolist()}


async def main():
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)
    test_image[50:150, 50:150] = 255

    async with RationAIClient("http://127.0.0.1:8001") as client:
        prostate = client.model("prostate")
        result = await prostate.predict(test_image)
        print("Prostate result:", result)


if __name__ == "__main__":
    asyncio.run(main())
