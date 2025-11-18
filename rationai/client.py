from aiohttp import ClientSession

from rationai.resources.models import AsyncModels
from rationai.resources.qc import AsyncQualityControl


class AsyncClient(ClientSession):
    """Async client for multiple RationAI models.

    Thin wrapper for connection management only.
    All processing logic is in core.py.

    Each pre-configured model (prostate, nuclei) uses its own service URL.
    The client's base_url is only used for custom endpoints.
    """

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
