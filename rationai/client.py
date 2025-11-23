from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from aiohttp import ClientSession
from requests import Session


if TYPE_CHECKING:
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

    @cached_property
    def models(self) -> AsyncModels:
        from rationai.resources.models import AsyncModels

        return AsyncModels(self)

    @cached_property
    def qc(self) -> AsyncQualityControl:
        from rationai.resources.qc import AsyncQualityControl

        return AsyncQualityControl(self)


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

    @cached_property
    def models(self) -> Models:
        from rationai.resources.models import Models

        return Models(self)

    @cached_property
    def qc(self) -> QualityControl:
        from rationai.resources.qc import QualityControl

        return QualityControl(self)
