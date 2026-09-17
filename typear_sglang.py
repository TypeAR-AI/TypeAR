"""Minimal native SGLang HTTP client and response-shape compatibility."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence


class SGLangError(RuntimeError):
    pass


class SGLangClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._label_tokens: dict[str, tuple[int, str]] = {}

    def _request(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        allow_text: bool = False,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SGLangError(
                f"SGLang {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SGLangError(
                f"Could not reach SGLang at {self.base_url}: {exc}"
            ) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            if allow_text:
                return raw
            raise SGLangError(
                f"SGLang {path} returned non-JSON data: {raw[:500]}"
            ) from exc

    def _tokenizer_model(self) -> str:
        if self.model:
            return self.model
        info = self._request("/get_model_info")
        for key in ("served_model_name", "model_path", "tokenizer_path"):
            value = info.get(key) if isinstance(info, Mapping) else None
            if isinstance(value, str) and value:
                self.model = value
                return value
        raise SGLangError(
            "Could not discover a tokenizer model from /get_model_info; "
            "pass model=... or set SGLANG_MODEL"
        )

    def single_token(self, label: str) -> tuple[int, str]:
        """Return (token_id, exact decoded text), rejecting multi-token labels."""
        if label in self._label_tokens:
            return self._label_tokens[label]
        model = self._tokenizer_model()
        tokenized = self._request("/v1/tokenize", {"model": model, "prompt": label})
        token_ids = tokenized.get("tokens") if isinstance(tokenized, Mapping) else None
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            raise ValueError(
                f"Candidate label {label!r} must encode to exactly one token; "
                f"SGLang returned token IDs {token_ids!r}"
            )
        token_id = int(token_ids[0])
        detokenized = self._request(
            "/v1/detokenize", {"model": model, "tokens": [token_id]}
        )
        token_text = detokenized.get("text") if isinstance(detokenized, Mapping) else None
        if token_text != label:
            raise ValueError(
                f"Candidate {label!r} tokenizes to ID {token_id}, but that token "
                f"decodes as {token_text!r}; exact append would be ambiguous"
            )
        result = (token_id, token_text)
        self._label_tokens[label] = result
        return result

    def score_candidates(
        self, prefix: str, candidate_ids: Sequence[int]
    ) -> tuple[dict[int, float], Mapping[str, Any], float]:
        payload = {
            "text": prefix,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "token_ids_logprob": list(candidate_ids),
            "return_text_in_logprobs": True,
        }
        start = time.perf_counter()
        response = self._request("/generate", payload)
        elapsed = time.perf_counter() - start
        # The unrestricted generated token is deliberately ignored. Decisions
        # use only the explicitly requested candidate-token log probabilities.
        scores = extract_candidate_logprobs(response, candidate_ids)
        meta = response.get("meta_info", {}) if isinstance(response, Mapping) else {}
        return scores, meta, elapsed

    def cache_prefix(self, prefix: str) -> Mapping[str, Any]:
        """Prefill a shared prefix without generating an output token."""
        response = self._request(
            "/generate",
            {
                "text": prefix,
                "sampling_params": {"max_new_tokens": 0, "temperature": 0},
            },
        )
        if isinstance(response, list):
            if len(response) != 1:
                raise SGLangError(
                    f"Expected one prefix-cache response, got {len(response)}"
                )
            response = response[0]
        if not isinstance(response, Mapping):
            raise SGLangError(
                "Unexpected prefix-cache response type: "
                f"{type(response).__name__}"
            )
        meta = response.get("meta_info", {})
        return meta if isinstance(meta, Mapping) else {}

    def score_candidates_batch(
        self,
        prefixes: Sequence[str],
        candidate_ids: Sequence[Sequence[int]],
    ) -> tuple[list[tuple[dict[int, float], Mapping[str, Any]]], float]:
        """Score one candidate set for each prompt in a native SGLang batch."""
        if not prefixes:
            return [], 0.0
        if len(prefixes) != len(candidate_ids):
            raise ValueError("prefixes and candidate_ids must have the same length")
        payload = {
            "text": list(prefixes),
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "token_ids_logprob": [list(ids) for ids in candidate_ids],
            "return_text_in_logprobs": True,
        }
        start = time.perf_counter()
        response = self._request("/generate", payload)
        elapsed = time.perf_counter() - start
        if isinstance(response, Mapping) and len(prefixes) == 1:
            responses = [response]
        elif isinstance(response, list):
            responses = response
        else:
            raise SGLangError(
                f"Expected a batched /generate response, got {type(response).__name__}"
            )
        if len(responses) != len(prefixes):
            raise SGLangError(
                f"Expected {len(prefixes)} batched responses, got {len(responses)}"
            )

        results: list[tuple[dict[int, float], Mapping[str, Any]]] = []
        for item, ids in zip(responses, candidate_ids):
            scores = extract_candidate_logprobs(item, ids)
            meta = item.get("meta_info", {}) if isinstance(item, Mapping) else {}
            results.append((scores, meta if isinstance(meta, Mapping) else {}))
        return results, elapsed

    def flush_cache(self) -> None:
        reply = self._request("/flush_cache", {}, allow_text=True)
        if isinstance(reply, str) and not reply.startswith("Cache flushed."):
            raise SGLangError(f"SGLang refused to flush its prefix cache: {reply}")


def _score_entry(entry: Any, candidate_ids: set[int]) -> tuple[int, float] | None:
    if isinstance(entry, Mapping):
        token_id = entry.get("token_id", entry.get("id"))
        logprob = entry.get("logprob", entry.get("log_prob"))
        if token_id is not None and logprob is not None:
            return int(token_id), float(logprob)
        if len(entry) == 1:
            key, value = next(iter(entry.items()))
            try:
                return int(key), float(value)
            except (TypeError, ValueError):
                return None
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        # SGLang 0.5.19: [logprob, token_id, optional_token_text].
        if isinstance(entry[1], int) and entry[1] in candidate_ids:
            return int(entry[1]), float(entry[0])
        if isinstance(entry[0], int) and entry[0] in candidate_ids:
            return int(entry[0]), float(entry[1])
    return None


def extract_candidate_logprobs(
    response: Any, candidate_ids: Sequence[int]
) -> dict[int, float]:
    """Extract next-token candidate scores across known SGLang response shapes."""
    if isinstance(response, list):
        if len(response) != 1:
            raise SGLangError(f"Expected one /generate response, got {len(response)}")
        response = response[0]
    if not isinstance(response, Mapping):
        raise SGLangError(f"Unexpected /generate response type: {type(response).__name__}")

    meta = response.get("meta_info", {})
    containers = []
    for owner in (meta, response):
        if isinstance(owner, Mapping):
            for key in ("output_token_ids_logprobs", "token_ids_logprobs"):
                if owner.get(key) is not None:
                    containers.append((key, owner[key]))

    wanted = {int(token_id) for token_id in candidate_ids}
    for _, container in containers:
        if isinstance(container, Mapping):
            entries = [{key: value} for key, value in container.items()]
        else:
            positions = container if isinstance(container, list) else [container]
            if positions and isinstance(positions[0], list):
                first = positions[0]
                entries = (
                    first
                    if first and isinstance(first[0], (list, tuple, Mapping))
                    else positions
                )
            else:
                entries = positions
        found: dict[int, float] = {}
        for entry in entries:
            parsed = _score_entry(entry, wanted)
            if parsed is not None and parsed[0] in wanted:
                found[parsed[0]] = parsed[1]
        if wanted.issubset(found):
            return {token_id: found[token_id] for token_id in candidate_ids}

    keys = list(meta.keys()) if isinstance(meta, Mapping) else []
    raise SGLangError(
        "Candidate logprobs were not found for every requested token ID. "
        f"wanted={sorted(wanted)}, meta_info keys={keys}, "
        f"candidate fields={containers!r}"
    )
