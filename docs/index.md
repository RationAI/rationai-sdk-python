# RationAI Python SDK

Python SDK for interacting with RationAI pathology image analysis services (classification, segmentation, and QC).

[Quick start](learn/get-started/quick-start.md)

[How it works](learn/how-it-works.md)

[API reference](reference/client.md)

## What you can do

- Run image classification and segmentation via `client.models`.
- Run quality-control workflows via `client.qc`.
- Choose sync (`Client`) or async (`AsyncClient`) depending on your app.

## Minimal examples

### Model example

```python
from PIL import Image

import rationai

image = Image.open("path/to/image.jpg").convert("RGB")

with rationai.Client() as client:
	result = client.models.classify_image("model-name", image)
	print(result)
```

### QC example

```python
import rationai

with rationai.Client() as client:
	xopat_url = client.qc.check_slide(
		wsi_path="/data/slides/slide.svs",
		output_path="/data/qc-output/slide-001",
	)
	print(xopat_url)
```
