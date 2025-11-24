"""Model configurations for RationAI services.

This module contains the configuration for all pre-defined services available
through the RationAI client. Each service configuration specifies:
- endpoint: The API endpoint path (optional for services like QC)
- base_url: The default Kubernetes service URL
- service_type: "model" for inference models, "qc" for quality control
- format: "raw" for raw bytes or "json" for JSON payloads
- input_size, num_channels, pixel_range: Reserved for future use
"""

from __future__ import annotations


class ServiceConfig:
    """Configuration for a specific service endpoint."""

    def __init__(
        self,
        base_url: str,
        service_type: str = "model",  # "model" or "qc"
        endpoint: str | None = None,
        format: str = "raw",  # "raw" for bytes, "json" for JSON payload
        input_size: tuple[int, int] = (512, 512),
        num_channels: int = 3,
        pixel_range: tuple[float, float] = (0.0, 1.0),
    ):
        self.base_url = base_url
        self.service_type = service_type
        self.endpoint = endpoint
        self.format = format
        self.input_size = input_size
        self.num_channels = num_channels
        self.pixel_range = pixel_range


SERVICES: dict[str, ServiceConfig] = {
    "prostate": ServiceConfig(
        base_url="http://rayservice-models-serve-svc.rationai-notebooks-ns.svc.cluster.local:8000",
        service_type="model",
        endpoint="/prostate",
        format="json",
        input_size=(224, 224),
        num_channels=3,
        pixel_range=(0.0, 1.0),
    ),
    "nuclei": ServiceConfig(
        base_url="http://rayservice-models-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        service_type="model",
        endpoint="/lsp-detr",
        format="raw",
        input_size=(512, 512),
        num_channels=3,
        pixel_range=(0.0, 255.0),
    ),
    "qc": ServiceConfig(
        base_url="http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        service_type="qc",
        endpoint=None,
        format="json",
    ),
}

MODELS: dict[str, ServiceConfig] = {
    k: v for k, v in SERVICES.items() if v.service_type == "model"
}
