import asyncio
from collections.abc import AsyncIterator
from typing import Any, TypeVar, cast

import numpy as np
import pytest
from aiohttp import ClientSession
from numpy.typing import NDArray

from rationai.segmentation.core import AsyncNucleiSegmentation


T = TypeVar("T")


# Test fixtures
class DummySession:
    def __init__(self, delay: float = 0.01, error_on: list[int] | None = None):
        self.delay = delay
        self.error_on = error_on or []
        self.request_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def close(self):
        pass

    def post(self, url: str, *, data: Any = None, **kwargs):
        self.request_count += (
            1  # Increment first so 1-based counting matches test expectations
        )

        class DummyResponse:
            def __init__(self, delay: float, error_on: list[int], request_num: int):
                self.delay = delay
                self.error_on = error_on
                self.request_num = request_num
                self._closed = False

            async def __aenter__(self):
                if self.request_num in self.error_on:
                    raise Exception(f"Simulated error on request {self.request_num}")
                return self

            async def __aexit__(self, *args):
                self._closed = True

            def raise_for_status(self):
                pass

            async def json(self):
                await asyncio.sleep(self.delay)  # Simulate network delay
                return {
                    "polygons": [[[float(self.request_num), 0.0]]],
                    "embeddings": [[float(self.request_num), 2.0]],
                }

        return DummyResponse(self.delay, self.error_on, self.request_count)


@pytest.fixture
def mock_tiles() -> list[NDArray[np.uint8]]:
    """Create a sequence of test tiles."""
    return [np.full((256, 256, 3), i, dtype=np.uint8) for i in range(5)]


async def async_tile_generator(
    tiles: list[NDArray[np.uint8]], delay: float = 0.01
) -> AsyncIterator[NDArray[np.uint8]]:
    """Generate tiles asynchronously with a delay."""
    for tile in tiles:
        await asyncio.sleep(delay)
        yield tile


@pytest.mark.asyncio
async def test_unordered_streaming(mock_tiles):
    """Test unordered streaming processes tiles as they become available."""
    seg = AsyncNucleiSegmentation()
    seg._session = cast("ClientSession", DummySession(delay=0.02))
    tile_gen = async_tile_generator(mock_tiles, delay=0.01)
    streamer = await seg(tile_gen, stream_mode="unordered")

    results = []
    async for result in cast("AsyncIterator[dict[str, Any]]", streamer):
        results.append(result)

    # Should have a result for each tile
    assert len(results) == len(mock_tiles)
    # Order might be different from input due to async processing
    assert {r["embeddings"][0][0] for r in results} == set(
        range(1, len(mock_tiles) + 1)
    )


@pytest.mark.asyncio
async def test_ordered_streaming(mock_tiles):
    """Test ordered streaming maintains the order of input tiles."""
    seg = AsyncNucleiSegmentation()
    seg._session = cast("ClientSession", DummySession(delay=0.02))
    tile_gen = async_tile_generator(mock_tiles, delay=0.01)
    streamer = await seg(tile_gen, stream_mode="ordered")

    results = []
    async for result in cast("AsyncIterator[dict[str, Any]]", streamer):
        results.append(result)

    # Should have a result for each tile in order
    assert len(results) == len(mock_tiles)
    # Results should be in the same order as input tiles
    assert [r["embeddings"][0][0] for r in results] == list(
        range(1, len(mock_tiles) + 1)
    )


# @pytest.mark.asyncio
# async def test_stream_error_handling():
#     """Test error handling in streaming modes."""
#     error_tiles = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(3)]
#     session = DummySession(delay=0.01, error_on=[1])  # Error on first request
#     seg = AsyncNucleiSegmentation(cast(ClientSession, session))
#     tile_gen = async_tile_generator(error_tiles, delay=0.01)
#     streamer = await seg(tile_gen, stream_mode="unordered")

#     results = []
#     error_caught = False

#     try:
#         async for result in cast(AsyncIterator[Dict[str, Any]], streamer):
#             results.append(result)
#     except Exception as e:
#         print(f"Error caught: {e}")
#         assert "Simulated error" in str(e), f"Unexpected error type: {str(e)}"
#         error_caught = True
#     else:
#         assert False, "Should have raised an error"

#     assert error_caught, "Expected to catch a simulated error"
#     assert len(results) == 0, "Should not have processed any results"


@pytest.mark.asyncio
async def test_in_memory_small():
    """Test processing a single small image."""
    seg = AsyncNucleiSegmentation()
    seg._session = cast("ClientSession", DummySession())
    img = np.zeros((128, 128, 3), dtype=np.uint8)  # fits in max tile
    result = cast("dict[str, Any]", await seg(img, "lsp-detr"))
    assert "polygons" in result
    assert "embeddings" in result


@pytest.mark.asyncio
async def test_in_memory_large():
    """Test processing a large image that requires tiling."""
    seg = AsyncNucleiSegmentation()
    seg._session = cast("ClientSession", DummySession())
    img = np.zeros((3000, 3000, 3), dtype=np.uint8)  # larger than max tile
    results = cast("list[dict[str, Any]]", await seg(img, "lsp-detr"))
    assert isinstance(results, list)
    assert all("polygons" in r for r in results)
