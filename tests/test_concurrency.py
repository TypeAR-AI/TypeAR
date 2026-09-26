import pickle
import sys
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from typellm import SGLangClient, SGLangError, TypeLLMClient, Usage

from tests.test_batching import FakeServer

PICK = {"pick": {"type": "string", "enum": list("abcdef"), "permutations": 3}}


class SlowServer(FakeServer):
    """FakeServer that yields between requests so concurrent calls interleave."""

    def _request(self, path, payload=None, *, allow_text=False):
        time.sleep(0.001)
        return super()._request(path, payload, allow_text=allow_text)


class SharedClientTests(unittest.TestCase):
    def test_concurrent_calls_keep_their_own_prompts(self):
        client = TypeLLMClient(model="fake")
        client.sglang = SlowServer()
        finished = threading.Barrier(8)

        def run(n):
            client.generate(context=f"Receipt {n}", questions={
                "a": {"type": "integer"}, "b": {"type": "boolean"},
            })
            finished.wait()  # Every call has finished before any reads its prompts.
            return n, client.last_prompts

        with ThreadPoolExecutor(8) as pool:
            for n, prompts in pool.map(run, range(16)):
                self.assertEqual(len(prompts), 2)
                for prompt in prompts:
                    self.assertIn(f"Receipt {n}", prompt)

    def test_a_seeded_call_is_reproducible_and_leaves_the_shared_stream(self):
        client = TypeLLMClient(model="fake", seed=1)
        client.sglang = FakeServer()
        state = client.rng.getstate()
        client.generate(context="Roll", questions=PICK, seed=5)
        first = client.last_prompts
        client.generate(context="Roll", questions=PICK, seed=5)
        self.assertEqual(client.last_prompts, first)
        self.assertEqual(client.rng.getstate(), state)
        client.generate(context="Roll", questions=PICK)
        self.assertNotEqual(client.rng.getstate(), state)

    def test_clients_pickle_after_a_call(self):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions={"b": {"type": "boolean"}})
        copy = pickle.loads(pickle.dumps(client))
        self.assertEqual(copy.last_prompts, [])
        copy.generate(context="Receipt", questions={"b": {"type": "boolean"}})
        self.assertEqual(len(copy.last_prompts), 1)


class MeteredServer(SlowServer):
    """Reports 10 prompt, 4 cached and 1 completion token per prompt."""

    def __init__(self, fail_after=None):
        super().__init__()
        self.fail_after = fail_after

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and self.fail_after is not None and len(self.payloads) >= self.fail_after:
            raise SGLangError("SGLang went away")
        response = super()._request(path, payload, allow_text=allow_text)
        if path == "/generate":
            for item in response if isinstance(response, list) else [response]:
                item.setdefault("meta_info", {}).update(
                    prompt_tokens=10, cached_tokens=4, completion_tokens=1)
        return response


def prompts_sent(server):
    return sum(1 if isinstance(p["text"], str) else len(p["text"]) for p in server.payloads)


class UsageTests(unittest.TestCase):
    QUESTIONS = {"a": {"type": "integer"}, "b": {"type": "boolean"}, "c": {"type": "string"}}

    def test_usage_sums_every_generate_request_of_the_call(self):
        client = TypeLLMClient(model="fake")
        client.sglang = MeteredServer()
        self.assertIsNone(client.last_usage)
        client.generate(context="Receipt", questions=self.QUESTIONS)
        n = prompts_sent(client.sglang)
        self.assertEqual(client.last_usage, Usage(
            requests=len(client.sglang.payloads),
            prompt_tokens=10 * n, cached_tokens=4 * n, completion_tokens=n,
        ))
        client.sglang.payloads.clear()
        client.generate(context="Receipt", questions={"b": {"type": "boolean"}})
        self.assertEqual(client.last_usage.requests, len(client.sglang.payloads))

    def test_concurrent_calls_count_only_their_own_requests(self):
        client = TypeLLMClient(model="fake")
        client.sglang = MeteredServer()
        finished = threading.Barrier(4)

        def run(questions):
            client.generate(context="Receipt", questions=questions)
            finished.wait()
            return client.last_usage.requests

        one = {"b": {"type": "boolean"}}
        with ThreadPoolExecutor(4) as pool:
            counts = list(pool.map(run, [one, self.QUESTIONS, one, self.QUESTIONS]))
        self.assertEqual(counts[0], counts[2])
        self.assertEqual(counts[1], counts[3])
        self.assertGreater(counts[1], counts[0])
        self.assertEqual(sum(counts), len(client.sglang.payloads))

    def test_a_failed_call_still_reports_the_requests_it_made(self):
        client = TypeLLMClient(model="fake")
        client.sglang = MeteredServer(fail_after=2)
        with self.assertRaisesRegex(SGLangError, "went away"):
            client.generate(context="Receipt", questions=self.QUESTIONS)
        self.assertEqual(client.last_usage.requests, 2)
        with self.assertRaises(ValueError):
            client.generate(questions=self.QUESTIONS)
        self.assertIsNone(client.last_usage)


class LazyLoadTests(unittest.TestCase):
    def test_token_tables_load_once_across_threads(self):
        calls = []

        def load(source, cache_dir):
            calls.append(source)
            time.sleep(0.01)
            return {"tokens": [(1, "1")], "string_starts": [(2, '"')]}

        client = SGLangClient(model="fake", tokenizer="fake-tokenizer")
        with patch("typellm.sglang.load_token_tables", side_effect=load):
            with ThreadPoolExecutor(8) as pool:
                tables = list(pool.map(lambda _: client.numeric_token_pieces(), range(8)))
        self.assertEqual(calls, ["fake-tokenizer"])
        self.assertTrue(all(table == [(1, "1")] for table in tables))
        self.assertEqual(client.string_start_pieces(), [(2, '"')])

    def test_the_chat_tokenizer_loads_once_across_threads(self):
        calls = []
        barrier = threading.Barrier(4)

        def from_pretrained(source, **kwargs):
            calls.append(source)
            time.sleep(0.01)
            return object()

        client = SGLangClient(model="fake", tokenizer="fake-tokenizer")

        def load(_):
            barrier.wait()
            return client._get_chat_tokenizer()

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=from_pretrained)
        with patch.dict(sys.modules, {"transformers": transformers}):
            with ThreadPoolExecutor(4) as pool:
                loaded = set(map(id, pool.map(load, range(4))))
        self.assertEqual(calls, ["fake-tokenizer"])
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
