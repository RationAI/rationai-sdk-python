"""Example usage of QualityControl client for RationAI QC service.

For local testing with port-forward:
1. Run port-forward in a separate terminal:
   kubectl port-forward svc/rayservice-qc-serve-svc 8000:8000 -n rationai-jobs-ns

2. Set environment variable:
   export RATIONAI_QC_URL="http://localhost:8000"
   # or PowerShell:
   $Env:RATIONAI_QC_URL = "http://localhost:8000"

3. Run this example:
   python examples/qc_example.py
"""

import asyncio
import os

from rationai.clients.qc_client import QualityControl


async def example_single_slide():
    """Check quality of a single slide with retry logic."""
    print("=== Single Slide QC Check ===\n")
    print(f"Connecting to: {os.getenv('RATIONAI_QC_URL', 'cluster DNS (default)')}\n")

    async with QualityControl(
        request_timeout=1800,  # 30 minutes
        max_retries=5,  # Retry up to 5 times
        backoff_base=2,  # Exponential backoff: 2^1, 2^2, 2^3 seconds
    ) as qc:
        print(f"Using QC base URL: {qc.base_url}")
        result = await qc.check_slide(
            wsi_path="/mnt/data/Projects/Data/Public/prostate/PANDA/train_images/0a0f8e20b1222b69416301444b117678.tiff",
            output_path="/tmp",
            mask_level=4,
            sample_level=0,
            check_residual=True,
            check_folding=True,
            check_focus=True,
            wb_correction=False,  # Enable white balance correction if needed
        )

        print(f"Result: {result}")
        print(f"Success: {result.success}")
        print(f"Status: {result.status}")
        print(f"Response: {result.response}\n")


if __name__ == "__main__":
    print("RationAI QC Client Example\n")
    print("=" * 60)
    print("NOTE: For in-cluster use, no configuration needed.")
    print("      For local testing, see instructions at the top of this file.\n")

    # Run single slide example
    asyncio.run(example_single_slide())
