"""Minimal native SGLang HTTP client and response-shape compatibility."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Sequence

import httpx

from .numeric import load_token_tables
from .protocol import detect_protocol


class SGLangError(RuntimeError):
    def __init__(self, message: str = "", *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status  # The HTTP status SGLang answered with, if any.


# One character inside a JSON string: anything but a quote, backslash or control
# character, or an escape sequence.
_JSON_STRING_CHAR = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})'


def _after_open_quote(text: str) -> str:
    """Drop the ' "' or '"' the model wrote after '{"name":'."""
    if not text.startswith(('"', ' "')):
        raise ValueError(text)
    return text[text.index('"') + 1:]


def _closed_json_string(text: str) -> str:
    """Decode ' "characters"}' written after '{"name":'."""
    text = _after_open_quote(text)
    if not text.endswith('"}'):
        raise ValueError(text)
    return json.loads('"' + text[:-1])


def _partial_json_string(text: str) -> str:
    """Decode a string cut off before its closing quote, dropping a split escape."""
    text = _after_open_quote(text)
    for cut in range(min(len(text), 6) + 1):
        try:
            return json.loads('"' + text[:len(text) - cut] + '"')
        except ValueError:
            continue
    raise ValueError(text)



class SGLangClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000",
        model: str | None = None,
        timeout: float = 120.0,
        tokenizer: str | None = None,
        numeric_cache_dir: str | os.PathLike[str] | None = None,
        *,
        thinking: bool = False,
        thinking_budget: int | None = None,
        text_max_tokens: int = 512,
        answer_reserve_tokens: int = 64,
    ) -> None:
        if type(thinking) is not bool:
            raise ValueError("thinking must be a boolean")
        if thinking_budget is not None and (type(thinking_budget) is not int or thinking_budget <= 0):
            raise ValueError("thinking_budget must be a positive integer or None")
        if type(text_max_tokens) is not int or text_max_tokens <= 0:
            raise ValueError("text_max_tokens must be a positive integer")
        if type(answer_reserve_tokens) is not int or answer_reserve_tokens <= 0:
            raise ValueError("answer_reserve_tokens must be a positive integer")
        self.answer_reserve_tokens = max(answer_reserve_tokens, text_max_tokens)
        self._context_length_cache: int | None = None
        self.text_max_tokens = text_max_tokens
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.numeric_cache_dir = numeric_cache_dir
        self._label_tokens: dict[str, tuple[int, str]] = {}
        self._json_value_starts: dict[str, list[tuple[int, str]]] | None = None
        self._model_info_cache: Mapping[str, Any] | None = None
        self._numeric_tokens: list[tuple[int, str]] | None = None
        self._string_start_tokens: list[tuple[int, str]] | None = None
        self._chat_tokenizer: Any | None = None
        self._image_placeholder_cache: str | None = None
        self._end_of_message: tuple[int, str] | None = None
        # Serializes the expensive lazy loads when threads share one client.
        self._load_lock = threading.RLock()
        self._http_client: httpx.Client | None = None
        # Per-thread/task, so concurrent generate() calls never share images.
        self._active_images: ContextVar[tuple[str, ...]] = ContextVar(
            f"typellm_images_{id(self)}", default=()
        )

    def __getstate__(self) -> dict[str, Any]:
        # A ContextVar cannot be pickled; images belong to one call anyway.
        state = self.__dict__.copy()
        del state["_active_images"]
        del state["_load_lock"]
        state["_http_client"] = None  # Each copy opens its own connections.
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._active_images = ContextVar(f"typellm_images_{id(self)}", default=())
        self._load_lock = threading.RLock()

    def close(self) -> None:
        """Close the pooled connections to SGLang; a later request reopens them."""
        with self._load_lock:
            http_client, self._http_client = self._http_client, None
        if http_client is not None:
            http_client.close()

    def __enter__(self) -> "SGLangClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def warmup(self) -> None:
        """Load the model info, tokenizer and token tables before the first request."""
        self._tokenizer_model()
        self.end_of_message_token()
        self.json_value_starts()
        self.numeric_token_pieces()

    def _info(self, name: str) -> Any:
        # SGLang 0.5.6 renamed /get_<name> to /<name>; older servers and
        # sglang-router 0.3.2 only know the old name.
        try:
            return self._request(f"/{name}")
        except SGLangError as exc:
            if exc.status != 404:
                raise
            return self._request(f"/get_{name}")

    def _model_info(self) -> Mapping[str, Any]:
        with self._load_lock:
            if self._model_info_cache is None:
                response = self._info("model_info")
                if not isinstance(response, Mapping):
                    raise SGLangError("/model_info returned a non-object response")
                self._model_info_cache = response
            return self._model_info_cache

    def _http(self) -> httpx.Client:
        with self._load_lock:
            if self._http_client is None:
                # Keep-alive connections: numeric fields decode one request per token.
                self._http_client = httpx.Client(limits=httpx.Limits(
                    max_connections=None, max_keepalive_connections=64,
                ))
            return self._http_client

    def _request(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        allow_text: bool = False,
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        try:
            # Streamed so a failed body read still reports the status it follows.
            with self._http().stream(
                "GET" if payload is None else "POST",
                self.base_url + path,
                content=body,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            ) as response:
                if response.status_code >= 400:
                    try:
                        detail = response.read().decode("utf-8", errors="replace")
                    except httpx.HTTPError as read_error:
                        detail = f"Could not read error response: {read_error}"
                    raise SGLangError(
                        f"SGLang {path} returned HTTP {response.status_code}: {detail}",
                        status=response.status_code,
                    )
                raw = response.read().decode("utf-8")
        except httpx.HTTPError as exc:
            # Connection failures, timeouts, resets and truncated responses.
            raise SGLangError(
                f"Could not reach SGLang at {self.base_url}: {exc!r}"
            ) from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            if allow_text:
                return raw
            raise SGLangError(
                f"SGLang {path} returned non-JSON data: {raw[:500]}"
            ) from exc

    @contextmanager
    def images(self, images: Sequence[str]) -> Iterator[None]:
        """Attach encoded images to every /generate request made in this block."""
        images = tuple(images)
        if images:
            self.image_placeholder()  # Fail before any request if unsupported.
        token = self._active_images.set(images)
        try:
            yield
        finally:
            self._active_images.reset(token)

    def image_placeholder(self) -> str:
        """Return the text the chat template writes for one image."""
        if self._image_placeholder_cache is not None:
            return self._image_placeholder_cache
        # Derive the placeholder from the template, never hardcode a model's
        # vision tokens. The marker is confined to this local template probe.
        marker = "TYPELLM_IMAGE_BOUNDARY_3f1d7a"
        unsupported = (
            "The served chat template does not render image content; "
            "use a vision-language model to pass images"
        )
        text_only = self.render_chat(
            [{"role": "user", "content": marker}], add_generation_prompt=False
        )
        try:
            with_image = self.render_chat(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": marker}]}],
                add_generation_prompt=False,
            )
        except SGLangError as exc:
            raise SGLangError(unsupported) from exc
        if text_only.count(marker) != 1 or with_image.count(marker) != 1:
            raise SGLangError(unsupported)
        before, after = text_only.split(marker)
        image_before, image_after = with_image.split(marker)
        if not image_before.startswith(before) or image_after != after:
            raise SGLangError(unsupported)
        placeholder = image_before[len(before):].strip()
        if not placeholder:
            raise SGLangError(unsupported)
        self._image_placeholder_cache = placeholder
        return placeholder

    def _generate(self, payload: Mapping[str, Any]) -> Any:
        """POST /generate, adding the active images to each prompt."""
        images = self._active_images.get()
        if not images:
            return self._request("/generate", payload)
        text = payload["text"]
        prompts = [text] if isinstance(text, str) else list(text)
        placeholder = self.image_placeholder()
        for prompt in prompts:
            found = prompt.count(placeholder)
            if found != len(images):
                raise SGLangError(
                    f"Prompt contains {found} image placeholders for {len(images)} images; "
                    "the context text must not contain the model's image tokens"
                )
        if isinstance(text, str):
            image_data: Any = list(images)
        elif len(images) == 1:
            # SGLang gives a lone item to every prompt, so the image goes once
            # rather than once per prompt (720 copies for a 6-value "all").
            image_data = images[0]
        else:
            # A list is read as one entry per prompt.
            image_data = [list(images) for _ in prompts]
        return self._request("/generate", {**payload, "image_data": image_data})

    def _tokenizer_model(self) -> str:
        if self.model:
            return self.model
        info = self._model_info()
        for key in ("served_model_name", "model_path", "tokenizer_path"):
            value = info.get(key) if isinstance(info, Mapping) else None
            if isinstance(value, str) and value:
                self.model = value
                return value
        raise SGLangError(
            "Could not discover a tokenizer model from /model_info; "
            "pass model=... or set SGLANG_MODEL"
        )

    def _tokenizer_source(self) -> str:
        if self.tokenizer:
            return self.tokenizer
        info = self._model_info()
        for key in ("tokenizer_path", "model_path"):
            value = info.get(key)
            if isinstance(value, str) and value:
                return value
        if self.model:
            return self.model
        raise SGLangError(
            "Could not discover the tokenizer used by SGLang; pass tokenizer=..."
        )

    def _load_token_tables(self) -> None:
        with self._load_lock:
            if self._numeric_tokens is not None and self._string_start_tokens is not None:
                return
            tables = load_token_tables(self._tokenizer_source(), self.numeric_cache_dir)
            if self._numeric_tokens is None:
                self._numeric_tokens = tables["tokens"]
            if self._string_start_tokens is None:
                self._string_start_tokens = tables["string_starts"]

    def numeric_token_pieces(self) -> list[tuple[int, str]]:
        """Return the cached numeric-token table for the served model tokenizer."""
        if self._numeric_tokens is None:
            self._load_token_tables()
        return self._numeric_tokens

    def string_start_pieces(self) -> list[tuple[int, str]]:
        """Every token that can start a string value after '{"k":', such as ' "', '"' or '"This'."""
        if self._string_start_tokens is None:
            self._load_token_tables()
        return self._string_start_tokens

    def _get_chat_tokenizer(self) -> Any:
        with self._load_lock:
            return self._load_chat_tokenizer()

    def _load_chat_tokenizer(self) -> Any:
        if self._chat_tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise SGLangError(
                    "Chat-template rendering requires transformers; install it "
                    "with `pip install transformers`"
                ) from exc
            source = self._tokenizer_source()
            try:
                self._chat_tokenizer = AutoTokenizer.from_pretrained(
                    source, trust_remote_code=False,
                )
            except Exception as exc:
                raise SGLangError(
                    "Could not load the tokenizer chat template. Pass the model's "
                    "local tokenizer path or Hugging Face ID as tokenizer=..."
                ) from exc
        return self._chat_tokenizer

    def render_chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
        finish_thinking: bool = True,
    ) -> str:
        """Render history; optionally finish thinking before constrained decoding."""
        tokenizer = self._get_chat_tokenizer()
        if not getattr(tokenizer, "chat_template", None):
            raise SGLangError("The served tokenizer does not define a chat template")
        try:
            detect_protocol(tokenizer)
        except ValueError as exc:
            raise SGLangError(str(exc)) from exc
        try:
            rendered = tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.thinking and add_generation_prompt,
            )
        except Exception as exc:
            raise SGLangError(
                "Could not render the tokenizer chat template; ensure "
                "transformers and jinja2 are installed"
            ) from exc
        if not isinstance(rendered, str):
            raise SGLangError("Tokenizer chat template returned non-text output")
        # SGLang tokenizes the prompt with special tokens, so a tokenizer that
        # prepends BOS would double the one the template already wrote.
        bos = getattr(tokenizer, "bos_token", None)
        if bos and rendered.startswith(bos) and tokenizer.encode("x")[0] == tokenizer.bos_token_id:
            rendered = rendered[len(bos):]
        if add_generation_prompt and finish_thinking:
            return self._prepare_answer_prefix(rendered)
        return rendered

    def _prepare_answer_prefix(self, prefix: str) -> str:
        protocol = detect_protocol(self._get_chat_tokenizer())
        if self.thinking or protocol.has_open_thinking(prefix):
            return self._finish_thinking(prefix)
        return prefix

    def _continuation_parts(self, question: str | None = None) -> tuple[str, str]:
        # Derive turn delimiters from the actual tokenizer, never hardcode a
        # model's chat tokens. The marker is confined to this local template probe.
        marker = "TYPELLM_ASSISTANT_BOUNDARY_8b46c9"
        messages = [{"role": "user", "content": "Context"},
                    {"role": "assistant", "content": marker}]
        closed = self.render_chat(messages, add_generation_prompt=False)
        if closed.count(marker) != 1:
            raise SGLangError("Chat template cannot preserve assistant content for KV continuation")
        closing = closed.split(marker)[1]
        if question is None:
            return closing, ""
        extended = self.render_chat(
            messages + [{"role": "user", "content": question}],
            add_generation_prompt=True, finish_thinking=False,
        )
        if extended.count(marker) != 1:
            raise SGLangError("Chat template cannot preserve assistant content for KV continuation")
        tail = extended.split(marker)[1]
        if not tail.startswith(closing):
            raise SGLangError("Chat template does not support append-only KV continuation")
        return closing, tail[len(closing):]

    def complete_chat_prefix(self, prompt: str, answer: str) -> str:
        closing, _ = self._continuation_parts()
        return prompt + answer + closing

    def extend_chat_prefix(self, prefix: str, question: str, *, finish_thinking: bool = True) -> str:
        _, suffix = self._continuation_parts(question)
        prompt = prefix + suffix
        return self._prepare_answer_prefix(prompt) if finish_thinking else prompt

    def prepare_answer_prefixes(self, prefixes: Sequence[str]) -> list[str]:
        """Finish thinking for many generation prompts in one batched request."""
        protocol = detect_protocol(self._get_chat_tokenizer())
        pending = [i for i, p in enumerate(prefixes) if self.thinking or protocol.has_open_thinking(p)]
        finished = list(prefixes)
        if len(pending) == 1:
            finished[pending[0]] = self._finish_thinking(prefixes[pending[0]])
        elif pending:
            for i, value in zip(pending, self._finish_thinking_batch([prefixes[i] for i in pending])):
                finished[i] = value
        return finished

    def _context_length(self) -> int:
        """Read the served context window, including any server override."""
        if self._context_length_cache is None:
            info = self._info("server_info")
            candidates = []
            if isinstance(info, Mapping):
                candidates.append(info.get("context_length"))
                args = info.get("server_args", {})
                if isinstance(args, Mapping):
                    candidates.append(args.get("context_length"))
            limits = [n for n in candidates if type(n) is int and n > 0]
            if not limits:
                models = self._request("/v1/models")
                data = models.get("data", []) if isinstance(models, Mapping) else []
                for item in data:
                    if isinstance(item, Mapping) and (len(data) == 1 or item.get("id") == self.model):
                        limit = item.get("max_model_len")
                        if type(limit) is int and limit > 0:
                            limits.append(limit)
            if not limits:
                raise SGLangError("Could not discover the served context length to reserve final-answer space")
            self._context_length_cache = min(limits)
        return self._context_length_cache

    def _finish_thinking(self, prefix: str) -> str:
        params, image_tokens = self._thinking_params(prefix)
        response = self._generate({"text": prefix, "sampling_params": params})
        return self._complete_thinking(prefix, response, image_tokens)

    def _finish_thinking_batch(self, prefixes: Sequence[str]) -> list[str]:
        served: list[int | None] = [None] * len(prefixes)
        if self._active_images.get():
            response = self._generate({
                "text": list(prefixes),
                "sampling_params": {"max_new_tokens": 0, "temperature": 0},
            })
            if not isinstance(response, list) or len(response) != len(prefixes):
                raise SGLangError("Unexpected prompt-count batch response shape")
            served = [item.get("meta_info", {}).get("prompt_tokens") if isinstance(item, Mapping) else None
                      for item in response]
        planned = [self._thinking_params(prefix, count) for prefix, count in zip(prefixes, served)]
        response = self._generate({
            "text": list(prefixes),
            "sampling_params": [params for params, _ in planned],
        })
        if not isinstance(response, list) or len(response) != len(prefixes):
            raise SGLangError("Unexpected thinking batch response shape")
        return [self._complete_thinking(prefix, item, image_tokens)
                for prefix, item, (_, image_tokens) in zip(prefixes, response, planned)]

    def _thinking_params(self, prefix: str, served_tokens: int | None = None) -> tuple[dict[str, Any], int]:
        """Return sampling params for one thinking request and its image-token count."""
        tokenizer = self._get_chat_tokenizer()
        protocol = detect_protocol(tokenizer)
        if not protocol.has_open_thinking(prefix):
            raise SGLangError(
                f"thinking=True requires a native chat template ending in an open "
                f"{protocol.thinking_open} block; this template may not support thinking"
            )
        forced_end = protocol.forced_close()
        prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        image_tokens = 0
        if self._active_images.get():
            # Each image placeholder expands to many tokens on the server, so
            # count the prompt there; the prefill also warms the cache.
            if served_tokens is None:
                served_tokens = self.cache_prefix(prefix).get("prompt_tokens")
            if type(served_tokens) is not int:
                raise SGLangError("SGLang did not report prompt_tokens; cannot budget thinking with images")
            image_tokens = max(0, served_tokens - prefix_tokens)
            prefix_tokens += image_tokens
        closing_tokens = len(tokenizer.encode(forced_end, add_special_tokens=False))
        # This is available context, not an independent default thinking budget.
        available = self._context_length() - prefix_tokens - self.answer_reserve_tokens - closing_tokens - 16
        if available <= 0:
            raise SGLangError("Input leaves no room for thinking and the final constrained answer")
        limit = available if self.thinking_budget is None else min(available, self.thinking_budget)
        return {
            "max_new_tokens": limit,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "stop": [protocol.thinking_close], "no_stop_trim": True,
        }, image_tokens

    def _complete_thinking(self, prefix: str, response: Any, image_tokens: int) -> str:
        tokenizer = self._get_chat_tokenizer()
        protocol = detect_protocol(tokenizer)
        text = response.get("text") if isinstance(response, Mapping) else None
        if not isinstance(text, str):
            raise SGLangError("Thinking returned non-text output; no typed result returned")
        meta = response.get("meta_info", {})
        finish = meta.get("finish_reason", {}) if isinstance(meta, Mapping) else {}
        if isinstance(finish, Mapping) and finish.get("type") in {"abort", "error"}:
            raise SGLangError("Thinking was aborted; no typed result returned")
        if protocol.thinking_close not in text:
            stop_kind = finish.get("type") if isinstance(finish, Mapping) else None
            ended_turn = False
            if stop_kind == "stop":
                matched = finish.get("matched")
                # Only recover a recognized native EOS/turn end, not an
                # arbitrary stop or an unreported/truncated server response.
                endings = {protocol.turn_end, getattr(tokenizer, "eos_token", None)} - {None, ""}
                for ending in endings:
                    ids = tokenizer.encode(ending, add_special_tokens=False)
                    if matched == ending or (type(matched) is int and ids == [matched]):
                        ended_turn = True
                        # Some servers preserve the token, others filter it.
                        # Remove only its trailing occurrence, never user text.
                        trimmed = text.rstrip()
                        if trimmed.endswith(ending):
                            text = trimmed[:-len(ending)]
                        break
            if (stop_kind != "length" and not ended_turn) or not text.strip():
                raise SGLangError(f"Thinking ended without a closing {protocol.thinking_close} marker; no typed result returned")
            completed = protocol.answer_prefix(prefix, text + "\n\nI will now give the final answer.\n")
            # Guard tokenizer/count mismatches before issuing a final request.
            if len(tokenizer.encode(completed, add_special_tokens=False)) + image_tokens + self.answer_reserve_tokens > self._context_length():
                raise SGLangError("Thinking response exceeded the reserved context space")
            logging.getLogger("typellm").info(
                "Thinking %s; closing reasoning before constrained decoding",
                "ended at native turn terminator" if ended_turn else "length limit reached",
            )
            return completed
        reasoning = text.split(protocol.thinking_close, 1)[0]
        if not reasoning.strip():
            raise SGLangError("Thinking returned an empty block; no typed result returned")
        # Discard any unconstrained answer after the marker. The existing
        # runtime records only selected labels/numbers in subsequent history.
        return protocol.answer_prefix(prefix, reasoning)

    def end_of_message_token(self) -> tuple[int, str]:
        """Return the tokenizer's single native end-of-message token."""
        if self._end_of_message is None:
            self._end_of_message = self._find_end_of_message_token()
        return self._end_of_message

    def _find_end_of_message_token(self) -> tuple[int, str]:
        tokenizer = self._get_chat_tokenizer()
        turn_end = detect_protocol(tokenizer).turn_end
        if turn_end is not None:
            token_ids = tokenizer.encode(turn_end, add_special_tokens=False)
            if len(token_ids) != 1 or tokenizer.decode(token_ids, skip_special_tokens=False) != turn_end:
                raise SGLangError(f"Chat turn terminator {turn_end!r} must be one exact token")
            return int(token_ids[0]), turn_end
        token_id = getattr(tokenizer, "eos_token_id", None)
        token_text = getattr(tokenizer, "eos_token", None)
        if not isinstance(token_id, int) or not isinstance(token_text, str):
            raise SGLangError(
                "The served tokenizer must define one EOS/end-of-message token"
            )
        return token_id, token_text

    def _tokenize(self, text: str) -> list[int]:
        tokenized = self._request(
            "/v1/tokenize",
            {"model": self._tokenizer_model(), "prompt": text, "add_special_tokens": False},
        )
        ids = tokenized.get("tokens") if isinstance(tokenized, Mapping) else None
        if not isinstance(ids, list):
            raise SGLangError(f"/v1/tokenize returned no token list for {text!r}")
        return [int(i) for i in ids]

    def json_value_starts(self) -> dict[str, list[tuple[int, str]]]:
        """Tokens that start a JSON value right after '{"k":', read from the tokenizer.

        Returns {"null": [...], "positive": [...], "negative": [...], "string": [...]}
        as (token_id, text) pairs. Most tokenizers attach the space to the value:
        ' null', ' -', ' "'; a positive number starts with a lone ' '. The
        spaceless 'null', '"' and '""' are listed too when they are single
        tokens. Strings take every token that can start one, from the vocabulary.
        Kinds this tokenizer does not split that way are left empty.
        """
        if self._json_value_starts is not None:
            return self._json_value_starts
        key = '{"k":'
        key_ids = self._tokenize(key)
        starts: dict[str, list[tuple[int, str]]] = {"null": [], "positive": [], "negative": [], "string": []}
        for kind, sample, accept in (
            ("null", '{"k": null}', lambda piece: piece.strip() == "null"),
            ("positive", '{"k": 1}', lambda piece: piece != "" and piece.strip() == ""),
            ("negative", '{"k": -1}', lambda piece: piece.strip() == "-"),
        ):
            ids = self._tokenize(sample)
            if ids[:len(key_ids)] != key_ids or len(ids) <= len(key_ids):
                continue
            token = ids[len(key_ids)]
            piece = self._request("/v1/detokenize", {"model": self._tokenizer_model(), "tokens": [token]})
            piece = piece.get("text") if isinstance(piece, Mapping) else None
            if isinstance(piece, str) and accept(piece) and (token, piece) not in starts[kind]:
                starts[kind].append((token, piece))
        # A spaceless {"k":null} often merges with the colon ('":null'), so add
        # 'null' directly when it is a single token. Only next to ' null': without
        # it, {"k": null} starts with the space numbers share and null is hidden.
        if starts["null"]:
            try:
                token = self.single_token("null")
            except ValueError:
                token = None
            if token is not None and token not in starts["null"]:
                starts["null"].append(token)
        starts["string"] = list(self.string_start_pieces())
        self._json_value_starts = starts
        return starts

    def single_token(self, label: str) -> tuple[int, str]:
        """Return (token_id, exact decoded text), rejecting multi-token labels."""
        if label in self._label_tokens:
            return self._label_tokens[label]
        model = self._tokenizer_model()
        # A label continues the prompt, so it must be tokenized without BOS;
        # SGLang adds special tokens by default.
        tokenized = self._request(
            "/v1/tokenize",
            {"model": model, "prompt": label, "add_special_tokens": False},
        )
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
        result = (token_id, label)
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
        response = self._generate(payload)
        elapsed = time.perf_counter() - start
        # The unrestricted generated token is deliberately ignored. Decisions
        # use only the explicitly requested candidate-token log probabilities.
        scores = extract_candidate_logprobs(response, candidate_ids)
        meta = response.get("meta_info", {}) if isinstance(response, Mapping) else {}
        return scores, meta, elapsed

    def cache_prefix(self, prefix: str) -> Mapping[str, Any]:
        """Prefill a shared prefix without generating an output token."""
        response = self._generate(
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
        response = self._generate(payload)
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

    def generate_texts(
        self,
        prefixes: Sequence[str],
        max_lengths: Sequence[int | None],
        *,
        temperature: float = 0,
        seed: int = 0,
        after_key: bool = False,
    ) -> list[str]:
        """Generate JSON strings in a native batch, then validate every value.

        With after_key, every prompt ends with '{"name":', and the model writes the
        opening quote, the characters and the closing '"}'. Writing the quote
        itself lets it start with a merged token such as ' "$', which keeps the
        first character. A max length then truncates: generation is capped near
        that many tokens and the string is cut to that many characters, as a
        length-bounded grammar would.
        """
        if len(prefixes) != len(max_lengths):
            raise ValueError("prefixes and max_lengths must have the same length")
        if not prefixes:
            return []
        params = []
        for limit in max_lengths:
            budget = self.text_max_tokens
            if after_key:
                # End with the object's closing brace too: models close {"name": "text"}
                # with the single token '"}', which a bare '"' would rule out. A
                # length-bounded regex is several times slower, so the limit is
                # applied by truncation below instead.
                constraint = {"regex": ' ?"' + _JSON_STRING_CHAR + '*"\\}'}
                if limit is not None:
                    # Every token holds at least one character, besides the quotes.
                    budget = min(budget, limit + 3)
            else:
                schema: dict[str, Any] = {"type": "string"}
                if limit is not None:
                    schema["maxLength"] = limit
                constraint = {"json_schema": json.dumps(schema)}
            params.append({"max_new_tokens": budget,
                           "temperature": temperature, "sampling_seed": seed, **constraint})
        response = self._generate({"text": list(prefixes), "sampling_params": params})
        if isinstance(response, Mapping) and len(prefixes) == 1:
            response = [response]
        if not isinstance(response, list) or len(response) != len(prefixes):
            raise SGLangError("Unexpected text batch response shape")
        values = []
        for item, limit in zip(response, max_lengths):
            if not isinstance(item, Mapping):
                raise SGLangError("Invalid text response")
            meta = item.get("meta_info", {})
            finish = meta.get("finish_reason", {}) if isinstance(meta, Mapping) else {}
            kind = finish.get("type") if isinstance(finish, Mapping) else finish
            # With a max length, running out of tokens mid-string is a truncation.
            truncated = after_key and limit is not None and kind == "length"
            if kind != "stop" and not truncated:
                raise SGLangError(f"Text generation did not complete normally: {finish!r}")
            try:
                text = item["text"]
                if after_key:
                    value = _partial_json_string(text) if truncated else _closed_json_string(text)
                else:
                    value = json.loads(text)
            except (KeyError, TypeError, ValueError) as exc:
                raise SGLangError("Text generation returned an invalid JSON string") from exc
            if not isinstance(value, str):
                raise SGLangError("Text generation returned a non-string value")
            if after_key and limit is not None:
                value = value[:limit]
                if value and 0xD800 <= ord(value[-1]) <= 0xDBFF:
                    value = value[:-1]  # never leave half of a surrogate pair
            if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
                raise SGLangError("Text generation returned an unpaired Unicode surrogate")
            if limit is not None and len(value) > limit:
                raise SGLangError("Text generation exceeded maxLength")
            values.append(value)
        return values

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
