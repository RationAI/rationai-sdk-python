from typing import List, TypedDict

import numpy as np
from numpy.typing import NDArray


class Result(TypedDict):
    polygons: List[List[List[float]]]
    embeddings: List[List[float]]


class Tile(TypedDict):
    data: NDArray[np.uint8]
    x: int
    y: int
