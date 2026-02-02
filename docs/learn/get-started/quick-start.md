# Quick start

## Sync vs Async clients

This SDK provides two clients:

- `rationai.Client` (sync): Uses blocking HTTP requests. Best for scripts, notebooks, CLIs, or when your code is already synchronous.
- `rationai.AsyncClient` (async): Uses non-blocking HTTP requests (`await`). Best when you already have an `asyncio` event loop (FastAPI, async workers) or you want to run many requests concurrently.

Both clients expose the same high-level resources:

- `client.models` for image classification/segmentation
- `client.qc` for quality control endpoints

### What’s the actual difference?

- **Sync** calls (e.g. `client.models.classify_image(...)`) block the current thread until the request completes.
- **Async** calls (e.g. `await client.models.classify_image(...)`) yield control back to the event loop while the network request is in flight, so other tasks can run.

### Lifecycle (important)

- Prefer using context managers so connections are closed:
  - sync: `with rationai.Client(...) as client: ...`
  - async: `async with rationai.AsyncClient(...) as client: ...`
- If you don’t use `with`, call `client.close()` (sync) / `await client.aclose()` (async).

For details on what is sent over the wire (compression, payloads), see: [How it works](../how-it-works.md).

## API at a glance

### Models

#### `client.models.classify_image`

Signature:

`classify_image(model: str, image: PIL.Image.Image | numpy.typing.NDArray[numpy.uint8], timeout=...) -> float | dict[str, float]`

- `model`: Model name / path appended to `models_base_url`.
- `image`: **uint8 RGB** image (PIL or NumPy array of shape `(H, W, 3)`).
- `timeout`: Optional request timeout (defaults to the client’s timeout).
- Returns: classification result from JSON (often `float` for binary, or `dict[class, prob]`).

#### `client.models.segment_image`

Signature:

`segment_image(model: str, image: PIL.Image.Image | numpy.typing.NDArray[numpy.uint8], timeout=...) -> numpy.typing.NDArray[numpy.float16]`

- `model`: Model name / path appended to `models_base_url`.
- `image`: **uint8 RGB** image (PIL or NumPy array of shape `(H, W, 3)`).
- `timeout`: Optional request timeout (defaults to the client’s timeout).
- Returns: `float16` NumPy array with shape `(num_classes, height, width)`.

### Quality control (QC)

#### `client.qc.check_slide`

Signature:

`check_slide(wsi_path: os.PathLike[str] | str, output_path: os.PathLike[str] | str, config: SlideCheckConfig | None = None, timeout=3600) -> str`

- `wsi_path`: Path to a whole-slide image (evaluated by the QC service).
- `output_path`: Directory where the QC service should write masks (evaluated by the QC service).
- `config`: Optional `SlideCheckConfig` (see reference types).
- `timeout`: Request timeout (default is 3600 seconds).
- Returns: xOpat URL as plain text.

#### `client.qc.generate_report`

Signature:

`generate_report(backgrounds: Iterable[os.PathLike[str] | str], mask_dir: os.PathLike[str] | str, save_location: os.PathLike[str] | str, compute_metrics: bool = True, timeout=...) -> None`

- `backgrounds`: Iterable of slide/background image paths.
- `mask_dir`: Directory containing generated masks.
- `save_location`: Path where the report HTML should be written.
- `compute_metrics`: Whether to compute aggregated metrics (default: `True`).
- Returns: nothing.

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

async with rationai.AsyncClient() as client:
    result = await client.models.classify_image("model-name", image)
    print(result)
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
