from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class Tile(TypedDict):
    data: NDArray[np.uint8]
    x: int
    y: int
