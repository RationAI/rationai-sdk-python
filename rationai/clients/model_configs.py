"""Model configurations for RationAI services.

This module contains the configuration for all pre-defined models available
through the RationAI client. Each model configuration specifies:
- endpoint: The API endpoint path
- base_url: The default Kubernetes service URL
- format: "raw" for raw bytes or "json" for JSON payloads
- input_size, num_channels, pixel_range: Reserved for future use
"""

from __future__ import annotations


class ModelConfig:
    """Configuration for a specific model endpoint.

    Note: input_size, num_channels, and pixel_range are reserved for future use.
    Currently only endpoint, base_url, and format are used by the client.
    """

    def __init__(
        self,
        endpoint: str,
        base_url: str,
        format: str = "raw",  # "raw" for bytes, "json" for JSON payload
        input_size: tuple[int, int] = (512, 512),
        num_channels: int = 3,
        pixel_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.endpoint = endpoint
        self.base_url = base_url
        self.format = format
        self.input_size = input_size
        self.num_channels = num_channels
        self.pixel_range = pixel_range


MODELS: dict[str, ModelConfig] = {
    "prostate": ModelConfig(
        endpoint="/prostate",
        base_url="http://rayservice-prostate-serve-svc.rationai-notebooks-ns.svc.cluster.local:8000",
        format="json",
        input_size=(224, 224),
        num_channels=3,
        pixel_range=(0.0, 1.0),
    ),
    "nuclei": ModelConfig(
        endpoint="/lsp-detr",
        base_url="http://rayservice-models-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        format="raw",
        input_size=(
            512,
            512,
        ),  # Supports 256/512/1024/2048 dynamically
        num_channels=3,
        pixel_range=(0.0, 255.0),
    ),
}
