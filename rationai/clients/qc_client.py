"""Quality Control client for RationAI QC service.

This client handles slide quality checks including:
- Residual tissue detection
- Folding artifacts detection
- Focus quality assessment
"""

from __future__ import annotations

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
        request_timeout: int = 300,
    ):
        """Initialize QC client (minimal).

        Args:
            base_url: Base URL of the QC service. If None, will use
                environment variable RATIONAI_QC_URL if set; otherwise
                defaults to cluster-internal service DNS.
            request_timeout: Timeout for single slide processing (seconds).
                Default 300s (5 min) to handle Ray Serve cold start + processing.
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
    ) -> QCResult:
        """Check quality of a single slide (single PUT request).

        Args:
            wsi_path: Path to the whole slide image
            output_path: Directory to save output masks
            mask_level: Pyramid level for mask generation
            sample_level: Pyramid level for sampling
            check_residual: Enable residual tissue detection
            check_folding: Enable folding artifact detection
            check_focus: Enable focus quality assessment

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
        }

        timeout = ClientTimeout(total=self.request_timeout)

        try:
            async with self.session.put(
                self._base_url, json=data, timeout=timeout
            ) as response:
                text = await response.text()
                return QCResult(response.status, text, str(wsi_path))
        except (TimeoutError, ServerTimeoutError):
            logger.error(
                "Request timed out after %d seconds for %s",
                self.request_timeout,
                Path(wsi_path).name,
            )
            return QCResult(-1, "Timeout", str(wsi_path))
        except ClientError as e:
            logger.error("Client connection error for %s: %s", Path(wsi_path).name, e)
            return QCResult(-2, f"Client error: {e}", str(wsi_path))
