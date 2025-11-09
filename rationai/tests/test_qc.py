"""Tests for QualityControl client."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rationai.clients.qc_client import QCResult, QualityControl


@pytest.fixture
def mock_session():
    """Create a mock aiohttp ClientSession."""
    return AsyncMock()


@pytest.fixture
def qc_client(mock_session):
    """Create QC client with mocked session."""
    client = QualityControl(
        base_url="http://localhost:8000",
        request_timeout=10,
    )
    client._session = mock_session
    return client


class TestQCResult:
    def test_qc_result_success(self):
        """Test successful QC result."""
        result = QCResult(200, "OK", "/path/to/slide.svs")
        assert result.success
        assert not result.timeout
        assert result.status == 200

    def test_qc_result_timeout(self):
        """Test timeout QC result."""
        result = QCResult(-1, "Timeout", "/path/to/slide.svs")
        assert not result.success
        assert result.timeout

    def test_qc_result_error(self):
        """Test error QC result."""
        result = QCResult(500, "Internal Server Error", "/path/to/slide.svs")
        assert not result.success
        assert not result.timeout


class TestQualityControl:
    @pytest.mark.asyncio
    async def test_check_slide_success(self, qc_client, mock_session):
        """Test successful slide check."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="Success")

        mock_session.put = MagicMock()
        mock_session.put.return_value.__aenter__.return_value = mock_response

        result = await qc_client.check_slide(
            wsi_path="/path/to/slide.svs",
            output_path="/output",
        )

        assert result.success
        assert result.status == 200
        assert mock_session.put.called
