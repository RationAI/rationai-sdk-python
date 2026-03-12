# Quick start

## Sync vs Async clients

This SDK provides two clients:

- `rationai.Client` (sync): Uses blocking HTTP requests. Best for scripts, notebooks, CLIs, or when your code is already synchronous.
- `rationai.AsyncClient` (async): Uses non-blocking HTTP requests (`await`). Best when you already have an `asyncio` event loop (FastAPI, async workers) or you want to run many requests concurrently.

Both clients expose the same high-level resources:

- `client.models` for image classification/segmentation
- `client.qc` for quality control endpoints
- `client.slide` for slide-level operations (e.g. heatmaps)

### What’s the actual difference?

- **Sync** calls (e.g. `client.models.classify_image(...)`) block the current thread until the request completes.
- **Async** calls (e.g. `await client.models.classify_image(...)`) yield control back to the event loop while the network request is in flight, so other tasks can run.

### Lifecycle (important)

- Prefer using context managers so connections are closed:
  - sync: `with rationai.Client(...) as client: ...`
  - async: `async with rationai.AsyncClient(...) as client: ...`
- If you don’t use `with`, call `client.close()` (sync) / `await client.aclose()` (async).

For details on what is sent over the wire (compression, payloads), see: [How it works](../how-it-works.md).

For the full API reference, see the [reference documentation](../../reference/client.md).

## Synchronous client

```python
from PIL import Image
import rationai

image = Image.open("path/to/image.jpg").convert("RGB")

with rationai.Client() as client:
    result = client.models.classify_image("model-name", image)
    print(result)
```

## Asynchronous client

```python
import asyncio
from PIL import Image
import rationai

image = Image.open("path/to/image.jpg").convert("RGB")

async def main():
    async with rationai.AsyncClient() as client:
        result = await client.models.classify_image("model-name", image)
        print(result)

asyncio.run(main())
```

### Concurrency with the async client

Use `asyncio` concurrency when you need to process many images. A semaphore is the simplest way to cap concurrency so you don’t overload the server.

```python
import asyncio
from PIL import Image
import rationai

async def classify_many(paths: list[str], model: str, *, max_concurrent: int = 8) -> list[float | dict[str, float]]:
    sem = asyncio.Semaphore(max_concurrent)

    async def one(client: rationai.AsyncClient, path: str) -> float | dict[str, float]:
        async with sem:
            image = Image.open(path).convert("RGB")
            return await client.models.classify_image(model, image)

    async with rationai.AsyncClient() as client:
        return await asyncio.gather(*(one(client, p) for p in paths))
```

## Common pitfalls

- **PIL image mode**: ensure RGB.

```python
image = Image.open(path).convert("RGB")
```

- **NumPy dtype/shape**: the services expect `uint8` RGB images.

```python
import numpy as np

assert arr.dtype == np.uint8
assert arr.ndim == 3 and arr.shape[2] == 3
```

- **Forgetting to close clients**: prefer `with ...` / `async with ...`.

- **Too much async concurrency**: cap with a semaphore (start small like 4–16) to avoid server overload/timeouts.

- **Timeouts**: segmentation/QC can take longer. Increase per-request timeout if needed.

```python
result = client.models.segment_image("model", image, timeout=300)
```

- **QC paths are server-side**: `wsi_path` / `output_path` must exist where the QC service runs.

## Configuration

You can override service URLs and timeouts:

```python
from rationai import Client

client = Client(
    models_base_url="http://localhost:8000",
    qc_base_url="http://localhost:8001",
    timeout=300,
)
```
