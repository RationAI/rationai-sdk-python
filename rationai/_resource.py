from aiohttp import ClientSession


class AsyncAPIResource:
    def __init__(self, client: ClientSession) -> None:
        self._client = client
        self._get = client.get
        self._post = client.post
        self._patch = client.patch
        self._put = client.put
        self._delete = client.delete
