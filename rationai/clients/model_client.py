from __future__ import annotations

import logging
import os
from pathlib import Path

from rationai.clients.model_configs import MODELS
from rationai.segmentation.core import AsyncNucleiSegmentation


logger = logging.getLogger(__name__)


class Model:
    """Wrapper for a specific model endpoint.

    Delegates all processing to core.py.
    """

    def __init__(
        self, core_client: AsyncNucleiSegmentation, endpoint: str, format: str = "raw"
    ):
        self._core = core_client
        self._endpoint = endpoint
        self._format = format

    async def predict(self, input):
        """Send image to model. Delegates to core.py for all processing."""
        if isinstance(input, (str, Path)):
            import numpy as np
            from PIL import Image

            input = np.array(Image.open(input))

        return await self._core(input, endpoint=self._endpoint, format=self._format)

    async def stream(self, input, stream_mode: str = "unordered"):
        """Stream predictions. Delegates to core.py."""
        return await self._core(
            input, endpoint=self._endpoint, format=self._format, stream_mode=stream_mode
        )


class RationAIClient:
    """Async client for multiple RationAI models.

    Thin wrapper for connection management only.
    All processing logic is in core.py.

    Each pre-configured model (prostate, nuclei) uses its own service URL.
    The client's base_url is only used for custom endpoints.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: int = 30,
        max_concurrent: int = 5,
    ):
        if base_url is None:
            base_url = os.getenv(
                "RATIONAI_MODEL_URL",
                "http://rayservice-prostate-serve-svc.rationai-notebooks-ns.svc.cluster.local:8000",
            )
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
            # Use pre-configured model (uses model-specific service URL)
            prostate = client.model("prostate")
            nuclei = client.model("nuclei")

            # Use custom endpoint (uses client's base_url)
            custom = client.model("/my-custom-model")
        """
        if name in MODELS:
            model_config = MODELS[name]
            # Check for model-specific environment variable override
            env_var = f"RATIONAI_{name.upper()}_URL"
            base_url = os.getenv(env_var, model_config.base_url)

            # Create a new core client with model-specific base_url
            model_core = AsyncNucleiSegmentation(
                base_url=base_url,
                timeout=self._core.timeout,
                max_concurrent=self._core.max_concurrent,
            )
            model_core._session = self._core._session
            model_core._owns_session = False
            return Model(model_core, model_config.endpoint, model_config.format)
        else:
            # Custom endpoint uses client's base_url (assume raw format)
            return Model(self._core, name, "raw")
