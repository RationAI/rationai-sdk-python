"""Quality Control client for RationAI QC service.

This client handles slide quality checks including:
- Residual tissue detection
- Folding artifacts detection
- Focus quality assessment
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from aiohttp import ClientError, ClientSession, ClientTimeout, ServerTimeoutError


logger = logging.getLogger(__name__)


class QCResult:
    """Result from a single QC check."""

    def __init__(self, status: int, response: str, slide_path: str):
        self.status = status
        self.response = response
        self.slide_path = slide_path
        self.success = 200 <= status < 300
        self.timeout = status == -1

    def __repr__(self) -> str:
        return f"QCResult(slide={Path(self.slide_path).name}, status={self.status}, success={self.success})"


class QualityControl:
    """Minimal async client for RationAI Quality Control service.

    Thin wrapper for connection management and single PUT operation.

    Usage:
        async with QualityControl(base_url) as qc:
            result = await qc.check_slide(
                wsi_path="/path/on/server/slide.svs",
                output_path="/output",
                check_residual=True,
                check_folding=True,
                check_focus=True,
            )
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        request_timeout: int = 1800,  # 30 minutes
        max_retries: int = 5,
        backoff_base: int = 2,
    ):
        """Initialize QC client.

        Args:
            base_url: Base URL of the QC service. If None, will use
                environment variable RATIONAI_QC_URL if set; otherwise
                defaults to cluster-internal service DNS.
            request_timeout: Timeout for single slide processing (seconds).
                Default 1800s (30 min) to handle Ray Serve cold start + processing.
            max_retries: Maximum number of retry attempts for failed requests.
                Default 5 attempts.
            backoff_base: Base for exponential backoff in seconds.
                Default 2 seconds (2^1, 2^2, 2^3, etc.).
        """
        if base_url is None:
            base_url = os.getenv(
                "RATIONAI_QC_URL",
                "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
            )
        self._base_url = base_url.rstrip("/")
        self._session: ClientSession | None = None
        self._owns_session = True

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    @property
    def base_url(self) -> str:
        """Return the configured base URL (read-only)."""
        return self._base_url

    @property
    def session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager."
            )
        return self._session

    async def __aenter__(self) -> QualityControl:
        if self._session is None:
            self._session = ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def check_slide(
        self,
        wsi_path: str | Path,
        output_path: str | Path,
        *,
        mask_level: int = 4,
        sample_level: int = 0,
        check_residual: bool = True,
        check_folding: bool = True,
        check_focus: bool = True,
        wb_correction: bool = False,
    ) -> QCResult:
        """Check quality of a single slide with automatic retry on failure.

        Args:
            wsi_path: Path to the whole slide image
            output_path: Directory to save output masks
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

        timeout = ClientTimeout(total=self.request_timeout)

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.session.put(
                    self._base_url, json=data, timeout=timeout
                ) as response:
                    text = await response.text()

                    # Retry on 500 Internal Server Error
                    if response.status == 500 and attempt < self.max_retries:
                        logger.warning(
                            "Received status 500 for %s (attempt %d/%d), retrying...",
                            Path(wsi_path).name,
                            attempt,
                            self.max_retries,
                        )
                        await self._backoff(attempt)
                        continue

                    return QCResult(response.status, text, str(wsi_path))

            except (TimeoutError, ServerTimeoutError):
                logger.error(
                    "Request timed out after %d seconds for %s",
                    self.request_timeout,
                    Path(wsi_path).name,
                )
                return QCResult(-1, "Timeout", str(wsi_path))
            except ClientError as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Client error for %s (attempt %d/%d): %s, retrying...",
                        Path(wsi_path).name,
                        attempt,
                        self.max_retries,
                        e,
                    )
                    await self._backoff(attempt)
                    continue
                logger.error(
                    "Client connection error for %s: %s", Path(wsi_path).name, e
                )
                return QCResult(-2, f"Client error: {e}", str(wsi_path))

        # All retries exhausted
        logger.error("All retry attempts failed for %s", Path(wsi_path).name)
        return QCResult(-3, "All retry attempts failed", str(wsi_path))

    async def generate_report(
        self,
        backgrounds: list[str | Path],
        mask_dir: str | Path,
        save_location: str | Path,
        *,
        compute_metrics: bool = True,
    ) -> QCResult:
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

        url = f"{self._base_url}/report"
        timeout = ClientTimeout(total=self.request_timeout)

        try:
            async with self.session.put(url, json=data, timeout=timeout) as response:
                text = await response.text()
                return QCResult(response.status, text, str(save_location))
        except (TimeoutError, ServerTimeoutError):
            logger.error(
                "Report generation timed out after %d seconds", self.request_timeout
            )
            return QCResult(-1, "Timeout", str(save_location))
        except ClientError as e:
            logger.error("Client connection error during report generation: %s", e)
            return QCResult(-2, f"Client error: {e}", str(save_location))

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff delay.

        Args:
            attempt: Current attempt number (1-indexed)
        """
        delay = self.backoff_base**attempt
        await asyncio.sleep(delay)
