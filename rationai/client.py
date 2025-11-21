from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from aiohttp import ClientSession
from requests import Session


if TYPE_CHECKING:
    from rationai.resources.models import AsyncModels, Models
    from rationai.resources.qc import AsyncQualityControl, QualityControl


class Client(Session):
    def __init__(
        self,
        base_url: str = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.base_url = base_url.rstrip("/")

    @cached_property
    def models(self) -> Models:
        from rationai.resources.models import Models

        return Models(self)

    @cached_property
    def qc(self) -> QualityControl:
        from rationai.resources.qc import QualityControl

        return QualityControl(self)


class AsyncClient(ClientSession):
    def __init__(
        self,
        base_url: str = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, base_url=base_url, **kwargs)

    @cached_property
    def models(self) -> AsyncModels:
        from rationai.resources.models import AsyncModels

        return AsyncModels(self)

    @cached_property
    def qc(self) -> AsyncQualityControl:
        from rationai.resources.qc import AsyncQualityControl

        return AsyncQualityControl(self)
