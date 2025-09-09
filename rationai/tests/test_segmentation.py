import numpy as np
import pytest
from rationai.segmentation.segmentation import AsyncNucleiSegmentation

class DummySession:
    def post(self, url, data):
        class DummyResponse:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def raise_for_status(self): pass
            async def json(self):
                return {"polygons": [[[0.0, 0.0]]], "embeddings": [[1.0, 2.0]]}
        return DummyResponse()

@pytest.mark.asyncio
async def test_in_memory_small():
    seg = AsyncNucleiSegmentation(DummySession()) # type: ignore
    img = np.zeros((128, 128, 3), dtype=np.uint8)  # fits in max tile
    result = await seg(img, "lsp-detr")
    assert "polygons" in result
    assert "embeddings" in result

@pytest.mark.asyncio
async def test_in_memory_large():
    seg = AsyncNucleiSegmentation(DummySession()) # type: ignore
    img = np.zeros((3000, 3000, 3), dtype=np.uint8)  # larger than max tile
    results = await seg(img, "lsp-detr")
    assert isinstance(results, list)
    assert all("polygons" in r for r in results)
