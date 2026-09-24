"""Client-side image normalization for SGLang ``image_data``."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Sequence

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def _data_uri(data: bytes) -> str:
    mime = next((m for sig, m in _SIGNATURES if data.startswith(sig)), None)
    if mime is None and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    # SGLang decodes everything after the comma; the type is informational.
    return f"data:{mime or 'application/octet-stream'};base64,{base64.b64encode(data).decode('ascii')}"


def encode_image(image: Any) -> str:
    """Return a URL or data URI that SGLang can load, whatever machine it runs on.

    Local files are read here rather than passed as paths, because SGLang
    would resolve a path on the server, not on the caller's machine.
    """
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if not data:
            raise ValueError("image bytes must not be empty")
        return _data_uri(data)
    if isinstance(image, str) and image.startswith(("http://", "https://", "data:")):
        return image
    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        if not os.path.isfile(path):
            raise ValueError(
                f"image {path!r} is not a local file, an http(s) URL, or a data: URI"
            )
        with open(path, "rb") as handle:
            return encode_image(handle.read())
    if callable(getattr(image, "save", None)):
        # A PIL image; PNG keeps it lossless.
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return _data_uri(buffer.getvalue())
    raise ValueError(
        "each image must be a file path, http(s) URL, data: URI, bytes, or PIL image; "
        f"got {type(image).__name__}"
    )


def encode_images(images: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(images, (str, bytes, os.PathLike)) or not isinstance(images, Sequence):
        raise ValueError("images must be a list, e.g. images=['receipt.png']")
    return tuple(encode_image(image) for image in images)
