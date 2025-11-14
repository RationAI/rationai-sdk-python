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
from pathlib import Path

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
        result = await qc.check_single_slide(
            wsi_path="/mnt/data/FNBrno/prostate/AI-prostata/100-24-10-HE.czi",
            output_path="/tmp",
            mask_level=3,
            sample_level=1,
            check_residual=True,
            check_folding=True,
            check_focus=True,
            wb_correction=True,
        )

        print(f"Result: {result}")
        print(f"Success: {result.success}")
        print(f"Status: {result.status}")
        print(f"Response: {result.response}\n")


async def example_multiple_slides():
    """Check quality of multiple slides concurrently (max 4 at a time)."""
    print("=== Multiple Slides QC Check ===\n")
    print(f"Connecting to: {os.getenv('RATIONAI_QC_URL', 'cluster DNS (default)')}\n")

    # Example slide paths - adjust as needed
    slides = [
        "/mnt/data/FNBrno/prostate/AI-prostata/100-24-10-HE.czi",
        "/mnt/data/FNBrno/prostate/AI-prostata/101-24-11-HE.czi",
        "/mnt/data/FNBrno/prostate/AI-prostata/102-24-12-HE.czi",
        "/mnt/data/FNBrno/prostate/AI-prostata/103-24-13-HE.czi",
        "/mnt/data/FNBrno/prostate/AI-prostata/104-24-14-HE.czi",
    ]
    output_path = "/tmp/qc_output"

    async with QualityControl(
        request_timeout=1800,
        max_retries=5,
        backoff_base=2,
    ) as qc:
        print(f"Using QC base URL: {qc.base_url}")
        print(f"Processing {len(slides)} slides (max 4 concurrent)...\n")

        # Process all slides - semaphore automatically limits to 4 concurrent requests
        tasks = [
            qc.check_single_slide(
                wsi_path=slide,
                output_path=output_path,
                mask_level=3,
                sample_level=1,
                check_residual=True,
                check_folding=True,
                check_focus=True,
                wb_correction=True,
            )
            for slide in slides
        ]
        results = await asyncio.gather(*tasks)

        # Print results
        print("\n=== Results ===\n")
        for result in results:
            slide_name = Path(result.slide_path).name
            status_icon = "✓" if result.success else "✗"
            print(f"{status_icon} {slide_name}")
            print(f"  Status: {result.status}")
            print(f"  Response: {result.response}\n")

        # Summary
        successful = sum(1 for r in results if r.success)
        print(f"Summary: {successful}/{len(results)} slides processed successfully")


if __name__ == "__main__":
    print("RationAI QC Client Example\n")
    print("=" * 60)
    print("NOTE: For in-cluster use, no configuration needed.")
    print("      For local testing, see instructions at the top of this file.\n")

    # Run single slide example
    # asyncio.run(example_single_slide())

    # Run multiple slides example
    asyncio.run(example_multiple_slides())
