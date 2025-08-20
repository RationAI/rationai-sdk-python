from functools import cached_property

from aiohttp import ClientSession

from rationai.resources.segmentation import AsyncNucleiSegmentation


class AyncClient:
    def __init__(self, base_url: str) -> None:
        self.session = ClientSession(
            base_url=base_url, headers={"Content-Type": "application/octet-stream"}
        )

    @cached_property
    def nuclei_segmentation(self) -> AsyncNucleiSegmentation:
        return AsyncNucleiSegmentation(self.session)


import numpy as np


async def main():
    client = AyncClient("http://127.0.0.1:8000")

    x = await client.nuclei_segmentation(
        input=np.zeros((256, 256, 3), dtype=np.uint8), model="lsp-detr"
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
