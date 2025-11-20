from aiohttp import ClientSession
from requests import Session

from rationai.resources.models import AsyncModels, SyncModels
from rationai.resources.qc import AsyncQualityControl, SyncQualityControl


class AsyncClient(ClientSession):
    """Async client for RationAI services."""

    def __init__(
        self,
        base_url: str = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, base_url=base_url, **kwargs)
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


class SyncClient(Session):
    """Sync client for RationAI services."""

    def __init__(
        self,
        base_url: str = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._models: SyncModels | None = None
        self._qc: SyncQualityControl | None = None

    @property
    def models(self) -> SyncModels:
        """Access the models resource."""
        if self._models is None:
            self._models = SyncModels(self)
        return self._models

    @property
    def qc(self) -> SyncQualityControl:
        """Access the quality control resource."""
        if self._qc is None:
            self._qc = SyncQualityControl(self)
        return self._qc
