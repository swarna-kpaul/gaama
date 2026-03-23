from __future__ import annotations

from pathlib import Path

from gaama.adapters.interfaces import BlobStoreAdapter


class LocalBlobStore(BlobStoreAdapter):
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def put_blob(self, key: str, data: bytes) -> None:
        path = self._root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get_blob(self, key: str) -> bytes:
        path = self._root / key
        return path.read_bytes()

