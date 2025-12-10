from collections.abc import Callable
from typing import Any, cast

import httpx
from httpx import URL


def _wrap[T: Callable[..., Any]](method: T, base_url: str | URL) -> T:
    return cast("T", lambda url, **kwargs: method(URL(base_url).join(url), **kwargs))


class APIResource:
    def __init__(self, client: httpx.Client, base_url: URL | str = "") -> None:
        self._client = client
        self._get = _wrap(client.get, base_url)
        self._post = _wrap(client.post, base_url)
        self._patch = _wrap(client.patch, base_url)
        self._put = _wrap(client.put, base_url)
        self._delete = _wrap(client.delete, base_url)


class AsyncAPIResource:
    def __init__(self, client: httpx.AsyncClient, base_url: URL | str = "") -> None:
        self._client = client
        self._get = _wrap(client.get, base_url)
        self._post = _wrap(client.post, base_url)
        self._patch = _wrap(client.patch, base_url)
        self._put = _wrap(client.put, base_url)
        self._delete = _wrap(client.delete, base_url)
