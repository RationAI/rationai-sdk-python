from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from aiohttp import ClientSession
from requests import Response, Session


if TYPE_CHECKING:
    from rationai.client import SyncClient


class AsyncAPIResource:
    def __init__(self, client: ClientSession) -> None:
        self._client = client
        self._get = client.get
        self._post = client.post
        self._patch = client.patch
        self._put = client.put
        self._delete = client.delete


class SyncAPIResource:
    def __init__(self, client: "SyncClient") -> None:
        self._session: Session = client
        self._base_url: str = client.base_url

    def _request(self, method: str, url: str, **kwargs: Any) -> Response:
        full_url = url if url.startswith("http") else urljoin(f"{self._base_url}/", url)

        raise_for_status = kwargs.pop("raise_for_status", True)

        response = self._session.request(method, full_url, **kwargs)

        if raise_for_status:
            response.raise_for_status()

        return response

    def _get(self, url: str, **kwargs: Any) -> Response:
        return self._request("GET", url, **kwargs)

    def _post(self, url: str, **kwargs: Any) -> Response:
        return self._request("POST", url, **kwargs)

    def _patch(self, url: str, **kwargs: Any) -> Response:
        return self._request("PATCH", url, **kwargs)

    def _put(self, url: str, **kwargs: Any) -> Response:
        return self._request("PUT", url, **kwargs)

    def _delete(self, url: str, **kwargs: Any) -> Response:
        return self._request("DELETE", url, **kwargs)
