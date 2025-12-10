from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

import httpx
from httpx import URL
from httpx._types import TimeoutTypes


if TYPE_CHECKING:
    from rationai.resources.models import AsyncModels, Models
    from rationai.resources.qc import AsyncQualityControl, QualityControl


class Client(httpx.Client):
    def __init__(
        self,
        *,
        models_base_url: URL
        | str = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        qc_base_url: URL
        | str = "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        timeout: TimeoutTypes = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(timeout=timeout, follow_redirects=True, **kwargs)
        self.models_base_url = models_base_url
        self.qc_base_url = qc_base_url

    def __enter__(self) -> Self:
        super().__enter__()
        return self

    @cached_property
    def models(self) -> Models:
        from rationai.resources.models import Models

        return Models(self, base_url=self.models_base_url)

    @cached_property
    def qc(self) -> QualityControl:
        from rationai.resources.qc import QualityControl

        return QualityControl(self, base_url=self.qc_base_url)


class AsyncClient(httpx.AsyncClient):
    def __init__(
        self,
        *,
        models_base_url: URL
        | str = "http://rayservice-model-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        qc_base_url: URL
        | str = "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        timeout: TimeoutTypes = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(timeout=timeout, follow_redirects=True, **kwargs)
        self.models_base_url = models_base_url
        self.qc_base_url = qc_base_url

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        return self

    @cached_property
    def models(self) -> AsyncModels:
        from rationai.resources.models import AsyncModels

        return AsyncModels(self, base_url=self.models_base_url)

    @cached_property
    def qc(self) -> AsyncQualityControl:
        from rationai.resources.qc import AsyncQualityControl

        return AsyncQualityControl(self, base_url=self.qc_base_url)
