from functools import cached_property

import numpy as np
from aiohttp import ClientSession

from rationai.segmentation.segmentation import AsyncNucleiSegmentation


class AsyncClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        self._session: ClientSession | None = None

    async def __aenter__(self):
        self._session = ClientSession(
            base_url=self._base_url,
            headers={"Content-Type": "application/octet-stream"},
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session:
            await self._session.close()

    @cached_property
    def nuclei_segmentation(self) -> AsyncNucleiSegmentation:
        if not self._session:
            raise RuntimeError(
                "Client session not initialized. Use `async with AsyncClient(...)`."
            )
        return AsyncNucleiSegmentation(self._session)


async def main():
    async with AsyncClient("http://127.0.0.1:8000") as client:
        result = await client.nuclei_segmentation(
            input=np.zeros((256, 256, 3), dtype=np.uint8),
            model="lsp-detr",
        )
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
