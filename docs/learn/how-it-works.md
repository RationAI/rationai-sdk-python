# How it works

This SDK is a thin client around RationAI services. It does not run ML models locally — it **sends requests** to remote services and returns their responses.

At a high level:

- You pass an image (PIL or NumPy) or file paths for QC.
- The SDK builds an HTTP request using `httpx`.
- For model inference, the SDK sends LZ4-compressed raw image bytes.
- It decodes the response into JSON (classification) or a NumPy array (segmentation).

## Clients and base URLs

- `rationai.Client` wraps `httpx.Client` (synchronous, blocking I/O)
- `rationai.AsyncClient` wraps `httpx.AsyncClient` (asynchronous, `await`-based)

Each client holds two service base URLs:

- `models_base_url`: classification + segmentation endpoints
- `qc_base_url`: quality-control endpoints

The `client.models` and `client.qc` properties return small resource objects that build requests relative to those base URLs.

Internally, requests are constructed by joining relative paths onto the configured base URLs (using `httpx.URL(...).join(...)`).

## Image classification and segmentation

### What the SDK sends

For `client.models.classify_image(model, image)` and `client.models.segment_image(model, image)`:

1. The input `image` is serialized with `image.tobytes()`.
2. The raw bytes are compressed using **LZ4 frame** (`lz4.frame.compress`).
3. The compressed bytes are sent as the raw HTTP request body (binary) using `POST`.

The request path is literally the `model` string you pass in, joined to `models_base_url`.

Important: the SDK sends only the raw pixel bytes. It does **not** send image metadata such as width/height/shape, color space, file name, or format.

For `client.models.embed_image(model, image, ...)`:

1. The input image is serialized with `image.tobytes()`.
2. Bytes are compressed with **LZ4 frame**.
3. The request includes `x-output-dtype` to let the service return the desired numeric type.
4. Additional keyword headers are supported and sent as `x-*` headers (e.g. `pool_tokens="false"` becomes `x-pool-tokens: false`; do not include the `x_` prefix in the argument name).

### What the SDK expects back

- **Classification**: JSON (`response.json()`), typically a float (binary) or a mapping of class → probability.
- **Segmentation**: a binary payload (response body) that is LZ4-compressed float16 data.
  The SDK decompresses it, interprets it as `np.float16`, and reshapes it to `(num_classes, height, width)`.
- **Embedding**: an LZ4-compressed binary payload plus an `x-output-shape` header,
  used to reshape the output array.

The SDK determines `height` and `width` from the input image:

- PIL: `image.size` → `(width, height)`
- NumPy: `image.shape[:2]` → `(height, width)`

So the server response is expected to contain exactly $N \cdot H \cdot W$ float16 values (after decompression), where $N$ is the number of output classes.

### Input image requirements

- The server expects an **uint8 RGB** image.
- For NumPy arrays: shape should be `(H, W, 3)` and dtype `uint8`.
- For PIL: ensure it’s RGB (e.g. `image = image.convert("RGB")`).

Because the SDK sends raw bytes without metadata, the server must already “know” how to interpret the byte stream (expected dtype/channel order). If you send the wrong dtype or channel count/order, you’ll typically get incorrect outputs or server-side errors.

## Quality control (QC)

### `check_slide`

`client.qc.check_slide(wsi_path, output_path, config=...)` sends a JSON payload to the QC service containing:

- `wsi_path`: path to the WSI
- `output_path`: where masks should be written
- configuration flags from `SlideCheckConfig`

The SDK retries up to 3 times **only for HTTP 500** responses (using `tenacity`).

"Paths are evaluated by the server"
`wsi_path` and `output_path` must make sense in the environment where the QC service runs (not necessarily your local machine).

### `generate_report`

`client.qc.generate_report(...)` sends a JSON payload to the `/report` endpoint and does not return a value.

### `check_slides` (async only)

`await client.qc.check_slides(...)` runs multiple `check_slide` requests concurrently and yields `SlideCheckResult` objects as each finishes.

- Concurrency is capped by `max_concurrent` (default: 4).
- Exceptions are captured per-slide and returned as `SlideCheckResult(..., success=False, error=str(e))`.

## Timeouts

- The client defaults to a global `timeout=100` seconds.
- QC slide checks use a much larger default (`timeout=3600` seconds) because slide processing can take time.

You can override:

- global per-client timeout in the client constructor
- per-request timeout in method calls

## Errors and status codes

- The SDK calls `response.raise_for_status()` for all endpoints, so non-2xx responses raise `httpx.HTTPStatusError`.
- QC `check_slide` / async `check_slide` are retried only for HTTP 500 (up to 3 attempts with exponential backoff).
