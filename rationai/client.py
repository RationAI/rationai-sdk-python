from aiohttp import ClientSession
from requests import Session

from rationai.resources.models import AsyncModels, SyncModels
from rationai.resources.qc import AsyncQualityControl, SyncQualityControl


class AsyncClient(ClientSession):
    def __init__(
        self,
        base_url: str = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, base_url=base_url, **kwargs)

    @property
    def models(self) -> AsyncModels:
        return AsyncModels(self)

    @property
    def qc(self) -> AsyncQualityControl:
        return AsyncQualityControl(self)


class SyncClient(Session):
    def __init__(
        self,
        base_url: str = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = base_url.rstrip("/")

    @property
    def models(self) -> SyncModels:
        return SyncModels(self)

    @property
    def qc(self) -> SyncQualityControl:
        return SyncQualityControl(self)
