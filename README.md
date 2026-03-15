# RationAI Python SDK

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://rationai.github.io/rationai-sdk-python/)

Python SDK for interacting with RationAI pathology image analysis services. This library provides both synchronous and asynchronous clients for image classification, segmentation, and quality control operations.

## Documentation

Comprehensive documentation is available at **[rationai.github.io/rationai-sdk-python](https://rationai.github.io/rationai-sdk-python/)**.

## Features

- **Image Analysis**: Classify and segment pathology images (WSI patches) using deployed models.
- **Quality Control**: Automated QC workflows for Whole Slide Images (WSI).
- **Concurrency**: Built-in async support for high-throughput processing.
- **Efficiency**: Optimized data transfer with LZ4 compression.

## Requirements

- Python 3.12 or higher

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/RationAI/rationai-sdk-python.git
```

## Quick Start

```python
from PIL import Image
import rationai

# Load an image (must be RGB)
image = Image.open("path/to/image.jpg").convert("RGB")

# Initialize the client and run classification
with rationai.Client() as client:
    result = client.models.classify_image("model-name", image)
    print(f"Classification result: {result}")
```

For asynchronous usage, configuration options, and advanced features, please refer to the [documentation](https://rationai.github.io/rationai-sdk-python/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
