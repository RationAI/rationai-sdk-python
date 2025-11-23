from aiohttp import ClientSession
from requests import Session

from rationai.resources.models import AsyncModels, Models
from rationai.resources.qc import AsyncQualityControl, QualityControl
from rationai.resources.service_configs import SERVICES


class AsyncClient(ClientSession):
    """Async client for RationAI services.

    Usage:
        # Use predefined service
        async with AsyncClient("prostate") as client:
            result = await client.models.predict("/prostate", image)

        # Use QC service
        async with AsyncClient("qc") as client:
            result = await client.qc.check_slide(...)
    """

    def __init__(
        self,
        service: str | None = None,
        *,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize AsyncClient.

        Args:
            service: Predefined service name ("prostate", "nuclei", "qc", ...)
                    Takes precedence over base_url if both provided
            base_url: Custom base URL (used if service not provided)
            **kwargs: Additional arguments passed to ClientSession
        """
        if service:
            if service not in SERVICES:
                available = ", ".join(SERVICES.keys())
                raise ValueError(
                    f"Unknown service '{service}'. Available services: {available}"
                )
            resolved_url = SERVICES[service].base_url
        elif base_url:
            resolved_url = base_url
        else:
            resolved_url = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000"

        super().__init__(*kwargs, base_url=resolved_url, **kwargs)
        self._qc: AsyncQualityControl | None = None
        self._models: AsyncModels | None = None

    @property
    def models(self) -> AsyncModels:
        """Access the models resource."""
        if self._models is None:
            self._models = AsyncModels(self)
        return self._models

    @property
    def qc(self) -> AsyncQualityControl:
        """Access the quality control resource."""
        if self._qc is None:
            self._qc = AsyncQualityControl(self)
        return self._qc


class Client(Session):
    """Sync client for RationAI services.

    Usage:
        # Use predefined service
        with Client("prostate") as client:
            result = client.models.predict("/prostate", image)

        # Use QC service
        with Client("qc") as client:
            result = client.qc.check_slide(...)

    """

    def __init__(
        self,
        service: str | None = None,
        *,
        base_url: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize Client.

        Args:
            service: Predefined service name ("prostate", "nuclei", "qc", ...)
                    Takes precedence over base_url if both provided
            base_url: Custom base URL (used if service not provided)
            **kwargs: Additional arguments passed to Session
        """
        super().__init__(**kwargs)

        if service:
            if service not in SERVICES:
                available = ", ".join(SERVICES.keys())
                raise ValueError(
                    f"Unknown service '{service}'. Available services: {available}"
                )
            self.base_url = SERVICES[service].base_url.rstrip("/")
        elif base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000"

        self._models: Models | None = None
        self._qc: QualityControl | None = None

    @property
    def models(self) -> Models:
        """Access the models resource."""
        if self._models is None:
            self._models = Models(self)
        return self._models

    @property
    def qc(self) -> QualityControl:
        """Access the quality control resource."""
        if self._qc is None:
            self._qc = QualityControl(self)
        return self._qc
