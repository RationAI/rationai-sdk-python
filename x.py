import asyncio

import numpy as np

import rationai


max_concurrent = 5


async def main(images: list[np.ndarray]):
    client = rationai.AsyncClient()
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    for image in images:
        async with semaphore:
            tasks.append(client.models.classify_image("prostate", image))

    results = await asyncio.gather(*tasks)
