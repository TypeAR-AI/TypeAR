"""Minimal native SGLang HTTP client and response-shape compatibility."""

from __future__ import annotations

import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, Sequence

from .numeric import load_numeric_token_table
from .protocol import detect_protocol


class SGLangError(RuntimeError):
    pass


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
        self._model_info_cache: Mapping[str, Any] | None = None
        self._numeric_tokens: list[tuple[int, str]] | None = None
        self._chat_tokenizer: Any | None = None
        self._image_placeholder_cache: str | None = None
        # Per-thread/task, so concurrent generate() calls never share images.
        self._active_images: ContextVar[tuple[str, ...]] = ContextVar(
            f"typellm_images_{id(self)}", default=()
        )

    def _info(self, name: str) -> Any:
        # SGLang 0.5.6 renamed /get_<name> to /<name>; older servers and
        # sglang-router 0.3.2 only know the old name.
        try:
            return self._request(f"/{name}")
        except SGLangError as exc:
            if getattr(exc.__cause__, "code", None) != 404:
                raise
            return self._request(f"/get_{name}")

    def _model_info(self) -> Mapping[str, Any]:
        if self._model_info_cache is None:
            response = self._info("model_info")
            if not isinstance(response, Mapping):
                raise SGLangError("/model_info returned a non-object response")
            self._model_info_cache = response
        return self._model_info_cache

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
            try:
                with exc:
                    detail = exc.read().decode("utf-8", errors="replace")
            except (OSError, http.client.HTTPException) as read_error:
                detail = f"Could not read error response: {read_error}"
            raise SGLangError(
                f"SGLang {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            # URLError covers connection failures; read timeouts and resets
            # surface as bare OSError subclasses, and truncated responses as
            # http.client.IncompleteRead.
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
        image_data = list(images) if isinstance(text, str) else [list(images) for _ in prompts]
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

    def numeric_token_pieces(self) -> list[tuple[int, str]]:
        """Return the cached numeric-token table for the served model tokenizer."""
        if self._numeric_tokens is None:
            self._numeric_tokens = load_numeric_token_table(
                self._tokenizer_source(), self.numeric_cache_dir
            )
        return self._numeric_tokens

    def _get_chat_tokenizer(self) -> Any:
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

    def extend_chat_prefix(self, prefix: str, question: str) -> str:
        _, suffix = self._continuation_parts(question)
        prompt = prefix + suffix
        return self._prepare_answer_prefix(prompt)

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
            served = self.cache_prefix(prefix).get("prompt_tokens")
            if type(served) is not int:
                raise SGLangError("SGLang did not report prompt_tokens; cannot budget thinking with images")
            image_tokens = max(0, served - prefix_tokens)
            prefix_tokens += image_tokens
        closing_tokens = len(tokenizer.encode(forced_end, add_special_tokens=False))
        # This is available context, not an independent default thinking budget.
        available = self._context_length() - prefix_tokens - self.answer_reserve_tokens - closing_tokens - 16
        if available <= 0:
            raise SGLangError("Input leaves no room for thinking and the final constrained answer")
        limit = available if self.thinking_budget is None else min(available, self.thinking_budget)
        response = self._generate({
            "text": prefix,
            "sampling_params": {
                "max_new_tokens": limit,
                "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                "stop": [protocol.thinking_close], "no_stop_trim": True,
            },
        })
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
    ) -> list[str]:
        """Generate JSON strings in a native batch, then validate every value."""
        if len(prefixes) != len(max_lengths):
            raise ValueError("prefixes and max_lengths must have the same length")
        if not prefixes:
            return []
        params = []
        for limit in max_lengths:
            schema: dict[str, Any] = {"type": "string"}
            if limit is not None:
                schema["maxLength"] = limit
            params.append({"max_new_tokens": self.text_max_tokens,
                           "temperature": temperature, "sampling_seed": seed,
                           "json_schema": json.dumps(schema)})
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
            if kind != "stop":
                raise SGLangError(f"Text generation did not complete normally: {finish!r}")
            try:
                value = json.loads(item["text"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SGLangError("Text generation returned an invalid JSON string") from exc
            if not isinstance(value, str):
                raise SGLangError("Text generation returned a non-string value")
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
