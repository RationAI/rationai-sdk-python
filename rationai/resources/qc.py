import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict
from os import PathLike

from httpx import USE_CLIENT_DEFAULT, HTTPStatusError
from httpx._client import UseClientDefault
from httpx._types import TimeoutTypes
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from rationai._resource import APIResource, AsyncAPIResource
from rationai.types import (
    EstimateStainingResult,
    SlideCheckConfig,
    SlideCheckResult,
    StainVector,
)


def _as_stain_vector(values: list[float]) -> StainVector:
    """Converts a list of floats to a StainVector."""
    a, b, c = values
    return (a, b, c)


def _is_500_error(exception: BaseException) -> bool:
    return (
        isinstance(exception, HTTPStatusError) and exception.response.status_code == 500
    )


class QualityControl(APIResource):
    @retry(
        retry=retry_if_exception(_is_500_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def check_slide(
        self,
        wsi_path: PathLike[str] | str,
        output_path: PathLike[str] | str,
        config: SlideCheckConfig | None = None,
        timeout: TimeoutTypes | UseClientDefault = 3600,
    ) -> str:
        """Check quality of a single slide with automatic retry on failure.

        Args:
            wsi_path: Path to the whole slide image.
            output_path: Directory to save output masks.
            config: Optional configuration for the slide quality check.
            timeout: Optional timeout for the request.

        Returns:
            An xOpat link containing the processed WSI, enabling a quick visual
            inspection of the generated masks. The masks are displayed in the same
            format as in the QC report.
        """
        response = self._put(
            "",
            json={
                "wsi_path": str(wsi_path),
                "output_path": str(output_path),
                **asdict(config or SlideCheckConfig()),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text

    @retry(
        retry=retry_if_exception(_is_500_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def estimate_staining(
        self,
        wsi_path: PathLike[str] | str,
        output_path: PathLike[str] | str | None = None,
        timeout: TimeoutTypes | UseClientDefault = 600,
    ) -> tuple[StainVector, StainVector]:
        """Estimate two dominant staining vectors for a single slide.

        Args:
            wsi_path: Path to the whole slide image.
            output_path: Optional directory where the reference region image will be saved.
            timeout: Optional timeout for the request.

        Returns:
            A tuple containing the estimated stains.
        """
        response = self._put(
            "staining",
            json={
                "wsi_path": str(wsi_path),
                "output_path": str(output_path) if output_path is not None else None,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        return (_as_stain_vector(data["stain1"]), _as_stain_vector(data["stain2"]))

    def generate_report(
        self,
        backgrounds: Iterable[PathLike[str] | str],
        mask_dir: PathLike[str] | str,
        save_location: PathLike[str] | str,
        compute_metrics: bool = True,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> None:
        """Generate a QC report from processed slides.

        Args:
            backgrounds: List of paths to the slide images.
            mask_dir: Directory containing the generated masks.
            save_location: Path where the report HTML will be saved.
            compute_metrics: Whether the generated report should contain aggregated
                metrics about the slide's coverage by the different classes of
                artifacts.
            timeout: Optional timeout for the request.
        """
        response = self._put(
            "report",
            json={
                "backgrounds": [str(bg) for bg in backgrounds],
                "mask_dir": str(mask_dir),
                "save_location": str(save_location),
                "compute_metrics": compute_metrics,
            },
            timeout=timeout,
        )
        response.raise_for_status()


class AsyncQualityControl(AsyncAPIResource):
    @retry(
        retry=retry_if_exception(_is_500_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def check_slide(
        self,
        wsi_path: PathLike[str] | str,
        output_path: PathLike[str] | str,
        config: SlideCheckConfig | None = None,
        timeout: TimeoutTypes | UseClientDefault = 3600,
    ) -> str:
        """Check quality of a single slide with automatic retry on failure.

        Args:
            wsi_path: Path to the whole slide image.
            output_path: Directory to save output masks.
            config: Optional configuration for the slide quality check.
            timeout: Optional timeout for the request.

        Returns:
            An xOpat link containing the processed WSI, enabling a quick visual
            inspection of the generated masks. The masks are displayed in the same
            format as in the QC report.
        """
        response = await self._put(
            "",
            json={
                "wsi_path": str(wsi_path),
                "output_path": str(output_path),
                **asdict(config or SlideCheckConfig()),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text

    async def check_slides(
        self,
        wsi_paths: Iterable[PathLike[str] | str],
        output_path: PathLike[str] | str,
        config: SlideCheckConfig | None = None,
        timeout: TimeoutTypes | UseClientDefault = 3600,
        max_concurrent: int = 4,
    ) -> AsyncIterator[SlideCheckResult]:
        """Check quality of multiple slides.

        Args:
            wsi_paths: List of paths to the whole slide images.
            output_path: Directory to save output masks.
            config: Configuration for the slide quality check.
            timeout: Optional timeout for the request.
            max_concurrent: Maximum number of concurrent slide checks.

        Yields:
            An asynchronous generator yielding SlideCheckResult for each slide.
        """

        async def safe_check(path: PathLike[str] | str) -> SlideCheckResult:
            try:
                url = await self.check_slide(path, output_path, config, timeout)
                return SlideCheckResult(path, xopat_url=url, success=True)

            except HTTPStatusError as e:
                code, reason = e.response.status_code, e.response.reason_phrase
                error_msg = f"HTTP Error {code} ({reason}): {e.response.text}"
                return SlideCheckResult(path, error=error_msg, success=False)

            except Exception as e:
                return SlideCheckResult(path, error=str(e), success=False)

        pending: set[asyncio.Task[SlideCheckResult]] = set()
        for path in wsi_paths:
            if len(pending) >= max_concurrent:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for d in done:
                    yield await d

            pending.add(asyncio.create_task(safe_check(path)))

        for task in asyncio.as_completed(pending):
            yield await task

    @retry(
        retry=retry_if_exception(_is_500_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def estimate_staining(
        self,
        wsi_path: PathLike[str] | str,
        output_path: PathLike[str] | str | None = None,
        timeout: TimeoutTypes | UseClientDefault = 600,
    ) -> tuple[StainVector, StainVector]:
        """Estimate two dominant staining vectors for a single slide.

        Args:
            wsi_path: Path to the whole slide image.
            output_path: Optional directory where the reference region image will be saved.
            timeout: Optional timeout for the request.

        Returns:
            A tuple containing the estimated stains.
        """
        response = await self._put(
            "staining",
            json={
                "wsi_path": str(wsi_path),
                "output_path": str(output_path) if output_path is not None else None,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        return (_as_stain_vector(data["stain1"]), _as_stain_vector(data["stain2"]))

    async def estimate_stainings(
        self,
        wsi_paths: Iterable[PathLike[str] | str],
        output_path: PathLike[str] | str | None = None,
        timeout: TimeoutTypes | UseClientDefault = 600,
        max_concurrent: int = 4,
    ) -> AsyncIterator[EstimateStainingResult]:
        """Estimate two dominant staining vectors for multiple slides.

        Args:
            wsi_paths: List of paths to the whole slide images.
            output_path: Optional directory where the images of reference regions will be saved.
            timeout: Optional timeout for the request.
            max_concurrent: Maximum number of concurrent staining estimations.

        Yields:
            An asynchronous generator yielding EstimateStainingResult for each slide.
        """

        async def safe_estimate(path: PathLike[str] | str) -> EstimateStainingResult:
            try:
                stain1, stain2 = await self.estimate_staining(
                    path, output_path, timeout
                )

                return EstimateStainingResult(
                    path, stain1=stain1, stain2=stain2, success=True
                )

            except HTTPStatusError as e:
                code, reason = e.response.status_code, e.response.reason_phrase
                error_msg = f"HTTP Error {code} ({reason}): {e.response.text}"
                return EstimateStainingResult(path, error=error_msg, success=False)

            except Exception as e:
                return EstimateStainingResult(path, error=str(e), success=False)

        pending: set[asyncio.Task[EstimateStainingResult]] = set()

        for path in wsi_paths:
            if len(pending) >= max_concurrent:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for d in done:
                    yield await d

            pending.add(asyncio.create_task(safe_estimate(path)))

        for task in asyncio.as_completed(pending):
            yield await task

    async def generate_report(
        self,
        backgrounds: Iterable[PathLike[str] | str],
        mask_dir: PathLike[str] | str,
        save_location: PathLike[str] | str,
        compute_metrics: bool = True,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> None:
        """Generate a QC report from processed slides.

        Args:
            backgrounds: List of paths to the slide images.
            mask_dir: Directory containing the generated masks.
            save_location: Path where the report HTML will be saved.
            compute_metrics: Whether the generated report should contain aggregated
                metrics about the slide's coverage by the different classes of
                artifacts.
            timeout: Optional timeout for the request.
        """
        data = {
            "backgrounds": [str(bg) for bg in backgrounds],
            "mask_dir": str(mask_dir),
            "save_location": str(save_location),
            "compute_metrics": compute_metrics,
        }

        response = await self._put("report", json=data, timeout=timeout)
        response.raise_for_status()
