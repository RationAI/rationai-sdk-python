import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from aiohttp import ClientResponseError, ClientSession

from rationai.segmentation.batch import (
    _process_individually,
    _send_batch,
    _send_batch_with_retry,
    batch_process,
)


# Fixtures
@pytest.fixture
def sample_images():
    """Create sample image arrays for testing."""
    return [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(10)]


@pytest.fixture
def mock_session():
    """Create a mock aiohttp ClientSession."""
    return AsyncMock(spec=ClientSession)


@pytest.fixture
def sample_results():
    """Create sample Result objects."""
    return [{"id": i, "data": []} for i in range(10)]


# Tests for batch_process
class TestBatchProcess:
    @pytest.mark.asyncio
    async def test_batch_process_success(
        self, mock_session, sample_images, sample_results
    ):
        """Test successful batch processing."""
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_results[:8])
        mock_response.raise_for_status = MagicMock()

        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch(
            "rationai.segmentation.batch._send_batch", return_value=sample_results[:8]
        ):
            results = await batch_process(mock_session, sample_images[:8], batch_size=8)

        assert len(results) == 8
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_batch_process_empty_list(self, mock_session):
        """Test batch processing with empty image list."""
        results = await batch_process(mock_session, [])
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_process_mismatched_shapes(self, mock_session, sample_images):
        """Test that mismatched image shapes raise ValueError."""
        mismatched_images = sample_images.copy()
        mismatched_images.append(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))

        with pytest.raises(ValueError, match="same shape"):
            await batch_process(mock_session, mismatched_images)

    @pytest.mark.asyncio
    async def test_batch_process_multiple_batches(
        self, mock_session, sample_images, sample_results
    ):
        """Test processing multiple batches."""
        batch1_results = sample_results[:8]
        batch2_results = sample_results[:2]

        with patch("rationai.segmentation.batch._send_batch_with_retry") as mock_send:
            mock_send.side_effect = [batch1_results, batch2_results]

            results = await batch_process(mock_session, sample_images, batch_size=8)

        assert len(results) == 10
        assert mock_send.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_process_with_failures_no_fallback(
        self, mock_session, sample_images
    ):
        """Test batch processing with failures and no fallback."""
        error = Exception("Batch failed")

        with patch("rationai.segmentation.batch._send_batch_with_retry") as mock_send:
            mock_send.side_effect = error

            results = await batch_process(
                mock_session, sample_images, fallback_to_individual=False
            )

        assert None in results
        assert sum(1 for r in results if r is None) == len(sample_images)

    @pytest.mark.asyncio
    async def test_batch_process_with_fallback_recovery(
        self, mock_session, sample_images, sample_results
    ):
        """Test batch processing with fallback individual processing."""
        batch_error = Exception("Batch failed")

        with patch("rationai.segmentation.batch._send_batch_with_retry") as mock_send:
            mock_send.side_effect = [batch_error, sample_results[0], sample_results[1]]

            with patch(
                "rationai.segmentation.batch._process_individually"
            ) as mock_individual:
                mock_individual.return_value = [sample_results[0], sample_results[1]]

                results = await batch_process(
                    mock_session,
                    sample_images[:2],
                    batch_size=1,
                    fallback_to_individual=True,
                )

        assert len(results) == 2
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_batch_process_with_cancelled_error(
        self, mock_session, sample_images
    ):
        """Test that cancelled errors are treated as batch failures and result in None values."""
        with patch("rationai.segmentation.batch._send_batch_with_retry") as mock_send:
            mock_send.side_effect = asyncio.CancelledError()

            results = await batch_process(
                mock_session, sample_images, fallback_to_individual=False
            )

        # CancelledError should be caught and result in None placeholders
        assert None in results
        assert sum(1 for r in results if r is None) == len(sample_images)


# Tests for _send_batch_with_retry
class TestSendBatchWithRetry:
    @pytest.mark.asyncio
    async def test_retry_success_on_first_attempt(self, mock_session):
        """Test successful send on first attempt."""
        sample_batch = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)]
        expected_result = [{"id": 0}]

        with patch(
            "rationai.segmentation.batch._send_batch", return_value=expected_result
        ) as mock_send:
            result = await _send_batch_with_retry(
                mock_session,
                sample_batch,
                "lsp-detr",
                64,
                max_retries=3,
                timeout=300,
                retry_delay=1.0,
            )

        assert result == expected_result
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_timeout_then_success(self, mock_session):
        """Test retry on timeout then success."""
        sample_batch = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)]
        expected_result = [{"id": 0}]

        with patch("rationai.segmentation.batch._send_batch") as mock_send:
            mock_send.side_effect = [asyncio.TimeoutError(), expected_result]

            result = await _send_batch_with_retry(
                mock_session,
                sample_batch,
                "lsp-detr",
                64,
                max_retries=3,
                timeout=300,
                retry_delay=0.1,
            )

        assert result == expected_result
        assert mock_send.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_server_error_then_success(self, mock_session):
        """Test retry on server error (5xx) then success."""
        sample_batch = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)]
        expected_result = [{"id": 0}]

        server_error = ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=500,
            message="Server error",
        )

        with patch("rationai.segmentation.batch._send_batch") as mock_send:
            mock_send.side_effect = [server_error, expected_result]

            result = await _send_batch_with_retry(
                mock_session,
                sample_batch,
                "lsp-detr",
                64,
                max_retries=3,
                timeout=300,
                retry_delay=0.1,
            )

        assert result == expected_result
        assert mock_send.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self, mock_session):
        """Test that client errors (4xx) are not retried."""
        sample_batch = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)]

        client_error = ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=400,
            message="Bad request",
        )

        with patch("rationai.segmentation.batch._send_batch") as mock_send:
            mock_send.side_effect = client_error

            with pytest.raises(ClientResponseError, match="400"):
                await _send_batch_with_retry(
                    mock_session,
                    sample_batch,
                    "lsp-detr",
                    64,
                    max_retries=3,
                    timeout=300,
                    retry_delay=0.1,
                )

        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, mock_session):
        """Test all retries exhausted."""
        sample_batch = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)]

        with patch("rationai.segmentation.batch._send_batch") as mock_send:
            mock_send.side_effect = asyncio.TimeoutError()

            with pytest.raises(asyncio.TimeoutError):
                await _send_batch_with_retry(
                    mock_session,
                    sample_batch,
                    "lsp-detr",
                    64,
                    max_retries=2,
                    timeout=300,
                    retry_delay=0.01,
                )

        assert mock_send.call_count == 2


# Tests for _send_batch
class TestSendBatch:
    @pytest.mark.asyncio
    async def test_send_batch_success(self, mock_session):
        """Test successful batch send."""
        batch = [
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]
        expected_results = [{"id": 0}, {"id": 1}, {"id": 2}]

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=expected_results)
        mock_response.raise_for_status = MagicMock()

        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        results = await _send_batch(mock_session, batch, "lsp-detr", 64, 300)

        assert results == expected_results
        assert mock_session.post.called

    @pytest.mark.asyncio
    async def test_send_batch_invalid_response_count(self, mock_session):
        """Test handling of mismatched result count."""
        batch = [
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]
        wrong_result_count = [{"id": 0}]  # Only 1 result instead of 3

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=wrong_result_count)
        mock_response.raise_for_status = MagicMock()

        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(ValueError, match="Expected 3 results"):
            await _send_batch(mock_session, batch, "lsp-detr", 64, 300)

    @pytest.mark.asyncio
    async def test_send_batch_invalid_response_type(self, mock_session):
        """Test handling of non-list response."""
        batch = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)]

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"error": "something"})
        mock_response.raise_for_status = MagicMock()

        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(ValueError, match="non-list"):
            await _send_batch(mock_session, batch, "lsp-detr", 64, 300)

    @pytest.mark.asyncio
    async def test_send_batch_request_format(self, mock_session):
        """Test that request is formatted correctly."""
        batch = [np.ones((64, 64, 3), dtype=np.uint8) for _ in range(2)]
        expected_results = [{"id": 0}, {"id": 1}]

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=expected_results)
        mock_response.raise_for_status = MagicMock()

        mock_session.post = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        await _send_batch(mock_session, batch, "lsp-detr", 64, 300)

        # Verify the POST call
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "/lsp-detr/64"
        assert call_args[1]["headers"]["Content-Type"] == "application/octet-stream"
        assert call_args[1]["headers"]["X-Batch-Size"] == "2"


# Tests for _process_individually
class TestProcessIndividually:
    @pytest.mark.asyncio
    async def test_process_individually_success(self, mock_session):
        """Test successful individual processing."""
        images = [
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]
        expected_results = [[{"id": 0}], [{"id": 1}], [{"id": 2}]]

        with patch("rationai.segmentation.batch._send_batch_with_retry") as mock_send:
            mock_send.side_effect = expected_results

            results = await _process_individually(
                mock_session, images, "lsp-detr", 64, 3, 300, 1.0
            )

        assert len(results) == 3
        assert results == [{"id": 0}, {"id": 1}, {"id": 2}]

    @pytest.mark.asyncio
    async def test_process_individually_partial_failure(self, mock_session):
        """Test individual processing with some failures."""
        images = [
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)
        ]
        results_and_errors = [
            [{"id": 0}],
            Exception("Failed"),
            [{"id": 2}],
        ]

        with patch("rationai.segmentation.batch._send_batch_with_retry") as mock_send:
            mock_send.side_effect = results_and_errors

            results = await _process_individually(
                mock_session, images, "lsp-detr", 64, 3, 300, 1.0
            )

        assert len(results) == 3
        assert results[0] == {"id": 0}
        assert results[1] is None
        assert results[2] == {"id": 2}

    @pytest.mark.asyncio
    async def test_process_individually_all_failures(self, mock_session):
        """Test individual processing with all failures."""
        images = [
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(2)
        ]

        with patch("rationai.segmentation.batch._send_batch_with_retry") as mock_send:
            mock_send.side_effect = [Exception("Failed"), Exception("Failed")]

            results = await _process_individually(
                mock_session, images, "lsp-detr", 64, 3, 300, 1.0
            )

        assert len(results) == 2
        assert all(r is None for r in results)
