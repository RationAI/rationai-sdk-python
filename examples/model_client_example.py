"""Example usage of RationAI model client.

For local testing with port-forward:
1. Run port-forward in a separate terminal:
   kubectl port-forward svc/rayservice-prostate-serve-svc 8001:8000 -n rationai-notebooks-ns

2. Set environment variable for the specific model:
   export RATIONAI_PROSTATE_URL="http://localhost:8001"
   # or PowerShell:
   $Env:RATIONAI_PROSTATE_URL = "http://localhost:8001"

3. Run this example:
   python examples/model_client_example.py
"""

import asyncio

import numpy as np

from rationai.clients.model_client import RationAIClient


async def main():
    """Example: prostate segmentation with RationAIClient."""
    async with RationAIClient() as client:
        prostate = client.model("prostate")
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"Test image shape: {test_image.shape}, dtype: {test_image.dtype}")
        result = await prostate.predict(test_image)
        print(f"Prostate result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
