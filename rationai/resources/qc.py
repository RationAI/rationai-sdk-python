from collections.abc import Iterable
from pathlib import Path

from aiohttp import ClientTimeout, ServerTimeoutError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rationai._resource import AsyncAPIResource


class AsyncQualityControl(AsyncAPIResource):
    @retry(
        retry=retry_if_exception_type(ServerTimeoutError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def check_slide(
        self,
        wsi_path: str | Path,
        output_path: str | Path,
        timeout: ClientTimeout | None = None,
        *,
        mask_level: int = 4,
        sample_level: int = 0,
        check_residual: bool = True,
        check_folding: bool = True,
        check_focus: bool = True,
        wb_correction: bool = False,
    ) -> str:
        """Check quality of a single slide with automatic retry on failure.

        Args:
            wsi_path: Path to the whole slide image
            output_path: Directory to save output masks
            timeout: Optional timeout for the request
            mask_level: Pyramid level for mask generation
            sample_level: Pyramid level for sampling
            check_residual: Enable residual tissue detection
            check_folding: Enable folding artifact detection
            check_focus: Enable focus quality assessment
            wb_correction: Enable white balance correction

        Returns:
            QCResult with status and response
        """
        data = {
            "wsi_path": str(wsi_path),
            "output_path": str(output_path),
            "mask_level": mask_level,
            "sample_level": sample_level,
            "check_residual": check_residual,
            "check_folding": check_folding,
            "check_focus": check_focus,
            "wb_correction": wb_correction,
        }

        async with self._put(
            "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000/",
            json=data,
            timeout=timeout,
            raise_for_status=True,
        ) as response:
            return await response.text()

    async def generate_report(
        self,
        backgrounds: Iterable[str | Path],
        mask_dir: str | Path,
        save_location: str | Path,
        timeout: ClientTimeout | None = None,
        compute_metrics: bool = True,
    ) -> str:
        """Generate a QC report from processed slides.

        Args:
            backgrounds: List of paths to the background (slide) images
            mask_dir: Directory containing the generated masks
            save_location: Path where the report HTML will be saved
            compute_metrics: Whether to compute quality metrics

        Returns:
            QCResult with status and response
        """
        data = {
            "backgrounds": [str(bg) for bg in backgrounds],
            "mask_dir": str(mask_dir),
            "save_location": str(save_location),
            "compute_metrics": compute_metrics,
        }

        async with self._put(
            "report", json=data, timeout=timeout, raise_for_status=True
        ) as response:
            return await response.text()
