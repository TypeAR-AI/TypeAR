import json
import unittest

from typellm import SGLangClient, TypeLLMClient
from typellm.images import encode_image

from tests.test_images import PNG, VisionTokenizer, fake_detokenize, fake_tokenize

THINK_STOP = "</think>"


class ThinkingTokenizer(VisionTokenizer):
    def apply_chat_template(self, messages, *, add_generation_prompt, enable_thinking=False, **kwargs):
        rendered = super().apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        return rendered + ("<think>\n" if add_generation_prompt and enable_thinking else "")


class FakeServer(SGLangClient):
    """A real SGLangClient whose HTTP layer answers like SGLang.

    Numeric fields decode to 7: the server prefers the end token, then "7".
    """

    def __init__(self, thinking=False):
        super().__init__(model="fake", thinking=thinking)
        self._chat_tokenizer = ThinkingTokenizer()
        self._numeric_tokens = [(ord(c), c) for c in "-0123456789."]
        self._context_length_cache = 100_000
        self.payloads = []

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/v1/tokenize":
            return {"tokens": fake_tokenize(payload["prompt"])}
        if path == "/v1/detokenize":
            return {"text": fake_detokenize(payload["tokens"])}
        assert path == "/generate", path
        self.payloads.append(payload)
        texts = [payload["text"]] if isinstance(payload["text"], str) else payload["text"]
        params = payload["sampling_params"]
        params = params if isinstance(params, list) else [params] * len(texts)
        if "token_ids_logprob" in payload:
            rows = payload["token_ids_logprob"]
            rows = [rows] if isinstance(rows[0], int) else rows
            out = []
            for ids in rows:
                # Like the real model: a number starts with the lone space token.
                pick = ord(" ") if ord(" ") in ids else 1 if 1 in ids else ord("7") if ord("7") in ids else ids[0]
                out.append({"meta_info": {"output_token_ids_logprobs": [
                    [[0.0 if t == pick else -9.0, t, "?"] for t in ids]]}})
        elif params[0].get("stop") == [THINK_STOP]:
            out = [{"text": "Reasoned." + THINK_STOP, "meta_info": {}} for _ in texts]
        elif "regex" in params[0]:
            # The prompt ends with '{"name": "'; the string's characters and '"}' follow.
            out = [{"text": 'blue"}', "meta_info": {"finish_reason": {"type": "stop"}}} for _ in texts]
        elif "json_schema" in params[0]:
            out = []
            for p in params:
                schema = json.loads(p["json_schema"])
                # An object schema answers {"key": "blue"}; a plain string schema "blue".
                value = {next(iter(schema["properties"])): "blue"} if schema["type"] == "object" else "blue"
                out.append({"text": json.dumps(value), "meta_info": {"finish_reason": {"type": "stop"}}})
        else:
            out = [{"meta_info": {"prompt_tokens": 900}} for _ in texts]
        return out[0] if isinstance(payload["text"], str) else out

    def requests(self, kind):
        def matches(p):
            params = p["sampling_params"]
            first = params[0] if isinstance(params, list) else params
            return {"score": "token_ids_logprob" in p,
                    "think": first.get("stop") == [THINK_STOP],
                    "count": first.get("max_new_tokens") == 0}[kind]
        return [p for p in self.payloads if matches(p)]


def width(payload):
    return 1 if isinstance(payload["text"], str) else len(payload["text"])


class NumericLockstepTests(unittest.TestCase):
    def test_numbers_share_one_request_per_digit_step(self):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeServer()
        result = client.generate(context="Receipt", questions={
            "a": {"type": "integer"}, "b": {"type": "number"}, "c": {"type": "integer"},
        })
        self.assertEqual(result, {"a": 7, "b": 7, "c": 7})
        # Sign, "7", then the end token: three steps, each scoring all three fields.
        self.assertEqual([width(p) for p in client.sglang.requests("score")], [3, 3, 3])

    def test_a_single_number_keeps_single_requests(self):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeServer()
        client.generate(context="Receipt", questions={"a": {"type": "integer"}})
        self.assertEqual([width(p) for p in client.sglang.requests("score")], [1, 1, 1])


class BatchedThinkingTests(unittest.TestCase):
    QUESTIONS = {
        "flag": {"type": "boolean"},
        "count": {"type": "integer"},
        "name": {"type": "string"},
        "pick": {"type": "string", "enum": ["x", "y"], "permutations": "all"},
    }

    def test_one_thinking_request_covers_every_prompt_in_a_layer(self):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeServer(thinking=True)
        result = client.generate(context="Receipt", questions=self.QUESTIONS)
        self.assertEqual(result, {"flag": True, "count": 7, "name": "blue", "pick": "x"})
        [think] = client.sglang.requests("think")
        # Four fields plus the reversed ordering of "pick".
        self.assertEqual(width(think), 5)
        self.assertEqual(len(think["sampling_params"]), 5)
        for payload in client.sglang.requests("score"):
            texts = [payload["text"]] if isinstance(payload["text"], str) else payload["text"]
            self.assertTrue(all("Reasoned." + THINK_STOP in t for t in texts))

    def test_dag_thinks_once_per_layer(self):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeServer(thinking=True)
        client.generate(context="Receipt", questions={
            "a": {"type": "boolean"}, "b": {"type": "integer"},
            "c": {"type": "boolean", "depends_on": ["a", "b"]},
            "d": {"type": "string", "depends_on": ["a"]},
        })
        self.assertEqual([width(p) for p in client.sglang.requests("think")], [2, 2])

    def test_images_count_prompt_tokens_in_one_batch(self):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeServer(thinking=True)
        client.generate(context="Receipt", images=[PNG],
                        questions={"a": {"type": "boolean"}, "b": {"type": "boolean"}})
        counts = [p for p in client.sglang.requests("count") if width(p) > 1]
        self.assertEqual([width(p) for p in counts], [2])
        [think] = client.sglang.requests("think")
        self.assertEqual(think["image_data"], [[encode_image(PNG)]] * 2)


if __name__ == "__main__":
    unittest.main()
