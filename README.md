# RationAI Python SDK

A Python SDK for interacting with RationAI pathology image analysis services. This library provides both synchronous and asynchronous clients for image classification, segmentation, and quality control operations.

## Installation

```bash
pip install git+https://github.com/RationAI/rationai-sdk-python.git
```

## Quick Start

### Synchronous Client

```python
import rationai
import numpy as np
from PIL import Image

# Initialize the client
client = rationai.Client()

# Load an image
image = Image.open("path/to/image.jpg")

# Classify the image
result = client.models.classify_image("model-name", image)
print(result)

# Segment the image
segmentation = client.models.segment_image("segmentation-model", image)
print(segmentation.shape)  # (num_classes, height, width)

client.close()
```

### Asynchronous Client

```python
import asyncio
import rationai

async def main():
    async with rationai.AsyncClient() as client:
        # Your async operations here
        result = await client.models.classify_image("model-name", image)
        print(result)

asyncio.run(main())
```

## API Reference

### Models (`client.models`)

#### `classify_image(model: str, image: Image | NDArray[np.uint8]) -> float | dict[str, float]`

Classify an image using the specified model.

**Parameters:**

- `model`: The name of the model to use for classification
- `image`: The image to classify (must be uint8 RGB image)
- `timeout`: Optional timeout for the request (defaults to 100 seconds)

**Returns:** Classification result as a float (binary) or dict of probabilities per class

#### `segment_image(model: str, image: Image | NDArray[np.uint8]) -> NDArray[np.float16]`

Segment an image using the specified model.

**Parameters:**

- `model`: The name of the model to use for segmentation
- `image`: The image to segment (must be uint8 RGB image)
- `timeout`: Optional timeout for the request (defaults to 100 seconds)

**Returns:** Segmentation mask as numpy array with shape `(num_classes, height, width)`


### Slide (`client.slide`)

#### `heatmap(model: str, slide_path: str, tissue_mask_path: str, output_path: str, stride_fraction: float = 0.5, output_bigtiff_tile_height: int = 512, output_bigtiff_tile_width: int = 512, timeout: int = 1000) -> str`

Generate a heatmap for a whole slide image using the specified model.

**Parameters:**
- `model`: The name of the model to use for heatmap generation
- `slide_path`: Path to the whole slide image
- `tissue_mask_path`: Path to the tissue mask for the slide
- `output_path`: Directory to save the generated heatmap tiles
- `stride_fraction`: Fraction of tile size to use as stride between tiles (default: 0.5)
- `output_bigtiff_tile_height`: Height of output heatmap tiles in pixels (default: 512)
- `output_bigtiff_tile_width`: Width of output heatmap tiles in pixels (default: 512)
- `timeout`: Optional timeout for the request (defaults to 1000 seconds)

**Returns:** The path to the generated heatmap. Should match the output_path provided.

### Quality Control (`client.qc`)

#### `check_slide(wsi_path, output_path, config=None, timeout=3600) -> str`

Check quality of a whole slide image.

**Parameters:**

- `wsi_path`: Path to the whole slide image
- `output_path`: Directory to save output masks
- `config`: Optional `SlideCheckConfig` for the quality check
- `timeout`: Optional timeout for the request (defaults to 3600 seconds)

**Returns:** xOpat link containing the processed WSI for visual inspection of generated masks

#### `check_slides(wsi_paths, output_path, config=None, timeout=3600, max_concurrent=4)` (async only)

Check quality of multiple slides concurrently.

**Parameters:**

- `wsi_paths`: List of paths to whole slide images
- `output_path`: Directory to save output masks
- `config`: Optional `SlideCheckConfig` for the quality check
- `timeout`: Optional timeout for each request (defaults to 3600 seconds)
- `max_concurrent`: Maximum number of concurrent slide checks (defaults to 4)

**Yields:** `SlideCheckResult` for each slide containing the path, xOpat URL (if successful), and any error information

#### `generate_report(backgrounds, mask_dir, save_location, compute_metrics=True, timeout=None) -> None`

Generate a QC report from processed slides.

**Parameters:**

- `backgrounds`: List of paths to background (slide) images
- `mask_dir`: Directory containing generated masks
- `save_location`: Path where the report HTML will be saved
- `compute_metrics`: Whether to compute quality metrics (default: True)
- `timeout`: Optional timeout for the request

For more details, refer to the [QC documentation](https://quality-control-rationai-digital-pathology-quali-82f7255ed88b44.gitlab-pages.ics.muni.cz).

## Managing Concurrency

To avoid overloading the server, it's important to limit concurrent requests. Here are recommended approaches:

### Using `asyncio.Semaphore`

Limit the number of concurrent requests:

```python
import asyncio
import rationai

async def process_images_with_semaphore(image_paths, model_name, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_segment(client, path):
        async with semaphore:
            image = Image.open(path).convert("RGB")
            return await client.models.segment_image(model_name, image)

    async with rationai.AsyncClient() as client:
        tasks = [bounded_segment(client, path) for path in image_paths]
        results = await asyncio.gather(*tasks)

    return results

# Process up to 16 images concurrently
results = asyncio.run(process_images_with_semaphore(image_paths, "model-name", max_concurrent=16))
```

### Using `asyncio.as_completed()`

Process results as they complete:

```python
import asyncio
from rationai import AsyncClient

async def process_with_as_completed(image_paths, model_name, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request(client, path):
        async with semaphore:
            image = Image.open(path).convert("RGB")
            return path, await client.models.segment_image(model_name, image)

    async with AsyncClient(models_base_url="http://localhost:8000") as client:
        tasks = {asyncio.create_task(bounded_request(client, path)): path
                 for path in image_paths}

        for future in asyncio.as_completed(tasks):
            path, result = await future
            print(f"Processed {path}")
            # Process result immediately without waiting for all tasks

asyncio.run(process_with_as_completed(image_paths, "model-name", max_concurrent=16))
```

Start with a conservative limit and monitor server resources to find the optimal value for your setup.

## Configuration

### Custom Timeouts

```python
from rationai import AsyncClient

async with AsyncClient(timeout=300) as client:  # 300 second timeout
    result = await client.models.segment_image("model", image, timeout=60)
```

### Custom Base URLs

```python
from rationai import Client

client = Client(
    models_base_url="http://custom-models-server:8000",
    qc_base_url="http://custom-qc-server:8000"
)
```
