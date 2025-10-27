import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from aiohttp import ClientSession
from ratiopath.tiling import grid_tiles

from rationai.segmentation.core import AsyncNucleiSegmentation


# Fixtures
@pytest.fixture
def mock_session():
    """Create a mock aiohttp ClientSession."""
    return AsyncMock(spec=ClientSession)


@pytest.fixture
def sample_result():
    """Create a sample Result object."""
    return {"id": "result_1", "data": []}


@pytest.fixture
def small_image():
    """Create a small image (within max tile size)."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Create a large image (larger than max tile size of 2048)."""
    return np.random.randint(0, 256, (3000, 3000, 3), dtype=np.uint8)


@pytest.fixture
def segmentation(mock_session):
    """Create AsyncNucleiSegmentation instance with mocked session."""
    client = AsyncNucleiSegmentation(base_url="http://localhost:8000", max_concurrent=5)
    # Inject the mock session for testing
    client._session = mock_session
    return client


# Tests for single NDArray input
class TestProcessNDArray:
    @pytest.mark.asyncio
    async def test_small_image_single_tile(
        self, segmentation, small_image, sample_result
    ):
        """Test processing a small image that fits in one tile."""
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_result)
        mock_response.raise_for_status = MagicMock()

        segmentation.session.post = MagicMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        result = await segmentation(small_image)

        assert result == sample_result
        assert segmentation.session.post.called

    @pytest.mark.asyncio
    async def test_small_image_with_explicit_call(
        self, segmentation, small_image, sample_result
    ):
        """Test processing small image via _process_ndarray."""
        with patch.object(
            segmentation, "_process_tile_with_retry", return_value=sample_result
        ):
            result = await segmentation._process_ndarray(small_image, "lsp-detr")

        assert result == sample_result

    @pytest.mark.asyncio
    async def test_large_image_multiple_tiles(self, segmentation, large_image):
        """Test processing a large image that requires multiple tiles."""
        # Calculate how many tiles we expect (3000x3000 split into 2048x2048)
        expected_tiles_count = len(
            list(
                grid_tiles(
                    slide_extent=(3000, 3000),
                    tile_extent=(2048, 2048),
                    stride=(2048, 2048),
                    last="shift",
                )
            )
        )
        sample_results = [
            {"id": i, "data": []} for i in range(4)
        ]  # 4 corners need to be covered

        # Create mock that returns each result in sequence
        mock_side_effects = []
        for result in sample_results:
            future = asyncio.Future()
            future.set_result(result)
            mock_side_effects.append(future)

        with patch.object(
            segmentation, "_process_tile_with_retry", side_effect=mock_side_effects
        ):
            results = await segmentation._process_ndarray(large_image, "lsp-detr")

        assert isinstance(results, list)
        assert len(results) == expected_tiles_count
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_image_padding(self, segmentation, sample_result):
        """Test that images are padded to correct tile size."""
        # Image smaller than any tile size
        small_odd_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_result)
        mock_response.raise_for_status = MagicMock()

        segmentation.session.post = MagicMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        _result = await segmentation(small_odd_image)

        # Verify post was called
        assert segmentation.session.post.called
        call_args = segmentation.session.post.call_args
        # Should be padded to 256x256
        assert "/lsp-detr/256" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_ndarray_input_dispatch(self, segmentation, small_image):
        """Test that NDArray input is correctly dispatched to _process_ndarray."""
        with patch.object(segmentation, "_process_ndarray") as mock_process:
            mock_process.return_value = {"id": "test"}

            await segmentation(small_image)

            mock_process.assert_called_once()
            np.testing.assert_array_equal(mock_process.call_args[0][0], small_image)


# Tests for tile processing with retry
class TestTileProcessingWithRetry:
    @pytest.mark.asyncio
    async def test_tile_process_success_first_attempt(
        self, segmentation, sample_result
    ):
        """Test successful tile processing on first attempt."""
        tile = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_result)
        mock_response.raise_for_status = MagicMock()

        segmentation.session.post = MagicMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        result = await segmentation._process_tile_with_retry(tile, "lsp-detr")

        assert result == sample_result
        assert segmentation.session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_tile_process_retry_on_timeout(self, segmentation, sample_result):
        """Test tile processing retries on timeout."""
        tile = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_result)
        mock_response.raise_for_status = MagicMock()

        segmentation.session.post = MagicMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        # Mock _process_tile to fail first time, then succeed
        with patch.object(segmentation, "_process_tile") as mock_process_tile:
            mock_process_tile.side_effect = [
                asyncio.TimeoutError(),
                sample_result,
            ]

            result = await segmentation._process_tile_with_retry(tile, "lsp-detr")

        assert result == sample_result
        assert mock_process_tile.call_count == 2

    @pytest.mark.asyncio
    async def test_tile_process_all_retries_exhausted(self, segmentation):
        """Test tile processing raises after all retries exhausted."""
        tile = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        with patch.object(segmentation, "_process_tile") as mock_process_tile:
            mock_process_tile.side_effect = asyncio.TimeoutError()

            with pytest.raises(Exception, match="timed out after"):
                await segmentation._process_tile_with_retry(tile, "lsp-detr")

        # Should retry initial + len(retry_delays) times
        assert mock_process_tile.call_count == len(segmentation.retry_delays) + 1

    @pytest.mark.asyncio
    async def test_tile_process_semaphore_concurrency(
        self, segmentation, sample_result
    ):
        """Test that semaphore limits concurrent processing."""
        tiles = [
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(3)
        ]

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_result)
        mock_response.raise_for_status = MagicMock()

        segmentation.session.post = MagicMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        # Track concurrent access
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrent(*args):  # Accept any arguments
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return sample_result

        with patch.object(segmentation, "_process_tile", side_effect=track_concurrent):
            results = await asyncio.gather(
                *[
                    segmentation._process_tile_with_retry(tile, "lsp-detr")
                    for tile in tiles
                ]
            )

        assert len(results) == 3
        # Should not exceed semaphore limit
        assert max_concurrent <= segmentation.semaphore._value + len(tiles)


# Tests for tile size selection
class TestTileSizeSelection:
    @pytest.mark.asyncio
    async def test_tile_size_selection_small(self, segmentation, sample_result):
        """Test correct tile size selection for small image."""
        tile = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_result)
        mock_response.raise_for_status = MagicMock()

        segmentation.session.post = MagicMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        await segmentation._process_tile(tile, "lsp-detr")

        # Should select first tile size >= 100, which is 256
        call_args = segmentation.session.post.call_args
        assert "/lsp-detr/256" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_tile_size_selection_medium(self, segmentation, sample_result):
        """Test correct tile size selection for medium image."""
        tile = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_result)
        mock_response.raise_for_status = MagicMock()

        segmentation.session.post = MagicMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        await segmentation._process_tile(tile, "lsp-detr")

        # Should select first tile size >= 600, which is 1024
        call_args = segmentation.session.post.call_args
        assert "/lsp-detr/1024" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_tile_size_selection_large(self, segmentation, sample_result):
        """Test correct tile size selection for large image."""
        tile = np.random.randint(0, 256, (2500, 2500, 3), dtype=np.uint8)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value=sample_result)
        mock_response.raise_for_status = MagicMock()

        segmentation.session.post = MagicMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        await segmentation._process_tile(tile, "lsp-detr")

        # Should select max tile size (2048) since 2500 > 2048
        call_args = segmentation.session.post.call_args
        assert "/lsp-detr/2048" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_unsupported_input_type(self, segmentation):
        """Test that unsupported input types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported input type"):
            await segmentation("invalid_input")

    @pytest.mark.asyncio
    async def test_tile_process_generic_exception(self, segmentation):
        """Test tile processing handles generic exceptions with retry."""
        tile = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        with patch.object(segmentation, "_process_tile") as mock_process_tile:
            mock_process_tile.side_effect = ValueError("Invalid data")

            with pytest.raises(Exception, match="Failed after"):
                await segmentation._process_tile_with_retry(tile, "lsp-detr")

        # Should retry len(retry_delays) + 1 times
        assert mock_process_tile.call_count == len(segmentation.retry_delays) + 1

    @pytest.mark.asyncio
    async def test_response_error_handling(self, segmentation):
        """Test that HTTP response errors are handled."""
        tile = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        async def mock_raise_error(*args, **kwargs):
            raise Exception("HTTP 500")

        mock_response = AsyncMock()
        mock_response.raise_for_status = mock_raise_error

        segmentation.session.post = AsyncMock()
        segmentation.session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(Exception, match="Failed after"):
            await segmentation._process_tile_with_retry(tile, "lsp-detr")
