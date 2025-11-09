"""Example usage of RationAI LSP-DETR nuclei segmentation model.

For local testing with port-forward:
1. Run port-forward in a separate terminal:
   kubectl port-forward svc/rayservice-models-serve-svc 8002:8000 -n rationai-jobs-ns

2. Set environment variable for the nuclei model:
   export RATIONAI_NUCLEI_URL="http://localhost:8002"
   # or PowerShell:
   $Env:RATIONAI_NUCLEI_URL = "http://localhost:8002"

3. Run this example:
   python examples/nuclei_example.py
"""

import asyncio
import os

import numpy as np

from rationai.clients.model_client import RationAIClient


async def main():
    """Example: nuclei segmentation with LSP-DETR model."""
    print(f"Connecting to: {os.getenv('RATIONAI_NUCLEI_URL', 'cluster DNS (default)')}")

    async with RationAIClient() as client:
        nuclei = client.model("nuclei")

        # Example 1: Small image (will use 256 tile size)
        print("\n=== Example 1: Small image (200x200) ===")
        small_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        print(f"Image shape: {small_image.shape}, dtype: {small_image.dtype}")
        result = await nuclei.predict(small_image)
        print(
            f"Result keys: {result.keys() if isinstance(result, dict) else type(result)}"
        )

        # Example 2: Medium image (will use 512 tile size)
        print("\n=== Example 2: Medium image (500x500) ===")
        medium_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        print(f"Image shape: {medium_image.shape}, dtype: {medium_image.dtype}")
        result = await nuclei.predict(medium_image)
        print(
            f"Result keys: {result.keys() if isinstance(result, dict) else type(result)}"
        )

        # Example 3: Large image (will use 1024 tile size)
        print("\n=== Example 3: Large image (900x900) ===")
        large_image = np.random.randint(0, 255, (900, 900, 3), dtype=np.uint8)
        print(f"Image shape: {large_image.shape}, dtype: {large_image.dtype}")
        result = await nuclei.predict(large_image)
        print(
            f"Result keys: {result.keys() if isinstance(result, dict) else type(result)}"
        )

        # Example 4: Very large image (will be split into multiple tiles)
        print("\n=== Example 4: Very large image (3000x3000) - multiple tiles ===")
        xlarge_image = np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)
        print(f"Image shape: {xlarge_image.shape}, dtype: {xlarge_image.dtype}")
        results = await nuclei.predict(xlarge_image)
        if isinstance(results, list):
            print(f"Number of tiles processed: {len(results)}")
            print(
                f"First tile result keys: {results[0].keys() if isinstance(results[0], dict) else type(results[0])}"
            )
        else:
            print(f"Single result: {type(results)}")


if __name__ == "__main__":
    asyncio.run(main())
