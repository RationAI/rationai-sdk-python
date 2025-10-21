from typing import List, TypedDict


class Result(TypedDict):
    polygons: List[List[List[float]]]
    embeddings: List[List[float]]
