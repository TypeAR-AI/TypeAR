"""Model-specific numeric and string-start token discovery, with persistent caching."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable


NUMERIC_CHARACTERS = frozenset("0123456789-.\"")
HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


class NumericTokenizerError(RuntimeError):
    pass


def _could_participate_in_number(text: str) -> bool:
    """Cheap tokenizer-independent filter before runtime state validation."""
    if text in {'"', "-", "."}:
        return True
    if not text or any(char not in NUMERIC_CHARACTERS for char in text):
        return False
    if '"' in text and (text.count('"') != 1 or not text.endswith('"')):
        return False
    body = text[:-1] if text.endswith('"') else text
    if body.startswith("-"):
        body = body[1:]
    if "-" in body or body.count(".") > 1 or not any(char.isdigit() for char in body):
        return False
    integer, separator, fraction = body.partition(".")
    if integer and not integer.isdigit():
        return False
    if separator and fraction and not fraction.isdigit():
        return False
    # A token ending the value must itself contribute a complete numeric
    # fragment; bare punctuation plus a quote can never become valid.
    if text.endswith('"') and (not fraction if separator else not integer):
        return False
    return True


def _continues_json_string(body: str) -> bool:
    """Whether body, written after an opening quote, can still end as '..."}'."""
    i = 0
    while i < len(body):
        char = body[i]
        if char == '"':
            return body[i + 1:] in ("", "}")
        if char == "\\":
            escape = body[i + 1:i + 2]
            if escape == "u":
                digits = body[i + 2:i + 6]
                if any(digit not in HEX_DIGITS for digit in digits):
                    return False
                i += 6
                continue
            if escape and escape not in '"\\/bfnrt':
                return False
            i += 2
            continue
        if ord(char) < 0x20:
            return False
        i += 1
    return True


def opens_json_string(text: str) -> bool:
    """Whether a token can start the string value right after '{"k":'."""
    for lead in (' "', '"'):
        if text.startswith(lead):
            return _continues_json_string(text[len(lead):])
    return False


def _default_cache_dir() -> Path:
    configured = os.environ.get("TYPELLM_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "typellm" / "numeric_tokens"


def _load_tokenizer(source: str) -> Any:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise NumericTokenizerError(
            "Open numeric decoding requires the 'tokenizers' package; "
            "install it with `pip install tokenizers`."
        ) from exc

    path = Path(source).expanduser()
    try:
        if path.is_dir():
            tokenizer_file = path / "tokenizer.json"
            if not tokenizer_file.is_file():
                raise NumericTokenizerError(
                    f"Tokenizer directory {source!r} has no tokenizer.json"
                )
            return Tokenizer.from_file(str(tokenizer_file))
        if path.is_file():
            return Tokenizer.from_file(str(path))
        return Tokenizer.from_pretrained(source)
    except NumericTokenizerError:
        raise
    except Exception as exc:
        raise NumericTokenizerError(
            f"Could not load tokenizer {source!r}. If SGLang uses a server-local "
            "path, pass its Hugging Face tokenizer ID to TypeLLMClient(tokenizer=...)."
        ) from exc


def _decoded_vocabulary(tokenizer: Any) -> list[tuple[int, str]]:
    try:
        token_ids: Iterable[int] = set(
            int(token_id)
            for token_id in tokenizer.get_vocab(with_added_tokens=True).values()
        )
    except Exception as exc:
        raise NumericTokenizerError("Tokenizer does not expose an enumerable vocabulary") from exc
    pieces = []
    for token_id in sorted(token_ids):
        try:
            pieces.append((token_id, tokenizer.decode([token_id], skip_special_tokens=False)))
        except Exception:
            continue
    return pieces


def build_numeric_token_table(tokenizer: Any) -> list[tuple[int, str]]:
    """Return every model token whose exact decoded text can occur in a number."""
    table = [(i, text) for i, text in _decoded_vocabulary(tokenizer) if _could_participate_in_number(text)]
    if not table:
        raise NumericTokenizerError("Tokenizer contains no usable numeric tokens")
    return table


def build_string_start_table(tokenizer: Any) -> list[tuple[int, str]]:
    """Return every model token that can start a JSON string value after '{"k":'."""
    return [(i, text) for i, text in _decoded_vocabulary(tokenizer) if opens_json_string(text)]


def load_token_tables(
    source: str, cache_dir: str | os.PathLike[str] | None = None
) -> dict[str, list[tuple[int, str]]]:
    """Load the numeric and string-start tables, rebuilding them only when the tokenizer changes."""
    tokenizer = _load_tokenizer(source)
    serialized = tokenizer.to_str()
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    directory = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    cache_path = directory / f"{digest}.json"

    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("version") == 3 and payload.get("tokenizer_sha256") == digest:
                return {key: [(int(item[0]), str(item[1])) for item in payload[key]]
                        for key in ("tokens", "string_starts")}
        except (OSError, ValueError, TypeError, KeyError):
            pass

    tables = {"tokens": build_numeric_token_table(tokenizer),
              "string_starts": build_string_start_table(tokenizer)}
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"version": 3, "source": source, "tokenizer_sha256": digest, **tables}
    # Unique per thread too: clients in one process may build the same table at once.
    temporary = cache_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, cache_path)
    return tables


def load_numeric_token_table(
    source: str, cache_dir: str | os.PathLike[str] | None = None
) -> list[tuple[int, str]]:
    """Load the tokenizer-derived numeric table."""
    return load_token_tables(source, cache_dir)["tokens"]
