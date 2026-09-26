import math
import unittest
from unittest.mock import patch

from typellm import TypeLLMClient

import tests.test_images as fake_vocab
from tests.test_batching import FakeServer


class SignServer(FakeServer):
    """Picks ' -' at the sign step when negative=True, then "7" and the end token."""

    def __init__(self, negative):
        super().__init__()
        self.negative = negative

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and "token_ids_logprob" in payload:
            rows = payload["token_ids_logprob"]
            rows = [rows] if isinstance(rows[0], int) else rows
            self.payloads.append(payload)
            out = []
            for ids in rows:
                sign = 900 if self.negative else ord(" ")
                pick = sign if sign in ids else 1 if 1 in ids else ord("7") if ord("7") in ids else ids[0]
                out.append({"meta_info": {"output_token_ids_logprobs": [
                    [[0.0 if t == pick else -9.0, t, "?"] for t in ids]]}})
            return out[0] if isinstance(payload["text"], str) else out
        return super()._request(path, payload, allow_text=allow_text)


class DirectDigitServer(FakeServer):
    """Writes the digit straight after the colon: {"t":7}."""

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and "token_ids_logprob" in payload:
            rows = payload["token_ids_logprob"]
            rows = [rows] if isinstance(rows[0], int) else rows
            self.payloads.append(payload)
            out = [{"meta_info": {"output_token_ids_logprobs": [
                [[0.0 if t == (1 if 1 in ids else ord("7")) else -9.0, t, "?"] for t in ids]]}} for ids in rows]
            return out[0] if isinstance(payload["text"], str) else out
        return super()._request(path, payload, allow_text=allow_text)


class SplitValueServer(FakeServer):
    """At the key: null 0.40, ' ' 0.35, ' -' 0.25 (value mass 0.60); then "7" and the end."""

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and "token_ids_logprob" in payload:
            ids = payload["token_ids_logprob"]
            self.payloads.append(payload)
            if 819 in ids:
                weights = {819: 0.40, ord(" "): 0.35, 900: 0.25}
                row = [[math.log(weights.get(t, 1e-9)), t, "?"] for t in ids]
            else:
                pick = 1 if 1 in ids else ord("7")
                row = [[0.0 if t == pick else -9.0, t, "?"] for t in ids]
            return {"meta_info": {"output_token_ids_logprobs": [row]}}
        return super()._request(path, payload, allow_text=allow_text)


class SpacelessNullServer(FakeServer):
    """Prefers the spaceless 'null' ({"t":null}) wherever it is offered."""

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and "token_ids_logprob" in payload:
            ids = payload["token_ids_logprob"]
            self.payloads.append(payload)
            pick = 2827 if 2827 in ids else 1 if 1 in ids else ord("7")
            return {"meta_info": {"output_token_ids_logprobs": [
                [[0.0 if t == pick else -9.0, t, "?"] for t in ids]]}}
        return super()._request(path, payload, allow_text=allow_text)


class KeyServer(FakeServer):
    """Scores the prefilled key with fixed probabilities; the rest is on unscored tokens."""

    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and "token_ids_logprob" in payload and 819 in payload["token_ids_logprob"]:
            return {"meta_info": {"output_token_ids_logprobs": [
                [[math.log(self.weights.get(t, 1e-9)), t, "?"] for t in payload["token_ids_logprob"]]]}}
        return super()._request(path, payload, allow_text=allow_text)


class ValueStartTests(unittest.TestCase):
    def test_starts_are_read_from_the_tokenizer(self):
        self.assertEqual(FakeServer().json_value_starts(), {
            "null": [(819, " null"), (2827, "null")], "positive": [(ord(" "), " ")],
            "negative": [(900, " -")], "string": [(328, ' "'), (901, ' ""'), (ord('"'), '"')],
        })

    def test_a_tokenizer_without_an_empty_string_token_still_works(self):
        del fake_vocab.PIECES[' ""']
        try:
            client = TypeLLMClient(model="fake")
            client.sglang = FakeServer()
            self.assertEqual(client.sglang.json_value_starts()["string"], [(328, ' "'), (ord('"'), '"')])
            result = client.generate(context="Receipt", questions={"note": {"type": ["string", "null"]}})
            self.assertIn(result["note"], (None, "blue"))
        finally:
            fake_vocab.PIECES[' ""'] = 901

    def test_negative_numbers_start_with_the_negative_token(self):
        client = TypeLLMClient(model="fake")
        client.sglang = SignServer(negative=True)
        result = client.generate(context="Log", questions={"t": {"type": "number"}})
        self.assertEqual(result, {"t": -7})
        self.assertIn('{"t": -7}', client.last_prompts[0])

    def test_a_positive_sign_rules_out_a_later_minus(self):
        client = TypeLLMClient(model="fake")
        client.sglang = SignServer(negative=False)
        self.assertEqual(client.generate(context="Log", questions={"t": {"type": "integer"}}), {"t": 7})
        digit_steps = [p["token_ids_logprob"] for p in client.sglang.requests("score")][1:]
        self.assertNotIn(ord("-"), digit_steps[0])


    def test_a_digit_right_after_the_colon_is_also_accepted(self):
        client = TypeLLMClient(model="fake")
        client.sglang = DirectDigitServer()
        result = client.generate(context="Log", questions={"t": {"type": ["integer", "null"]}})
        self.assertEqual(result, {"t": 7})
        [sign, *digits] = client.sglang.requests("score")
        self.assertTrue({ord(" "), 900, 819, ord("7"), ord("-")} <= set(sign["token_ids_logprob"]))
        # Decoding continued from the spaceless digit.
        self.assertTrue(digits[0]["text"].endswith('{"t":7'))


    def test_null_is_weighed_against_every_value_start_together(self):
        # null (0.40) beats each start alone, but not the 0.60 they share.
        client = TypeLLMClient(model="fake")
        client.sglang = SplitValueServer()
        self.assertEqual(client.generate(context="Log", questions={"t": {"type": ["integer", "null"]}}), {"t": 7})


    def test_a_spaceless_null_counts_and_is_written_with_the_space(self):
        client = TypeLLMClient(model="fake")
        client.sglang = SpacelessNullServer()
        result = client.generate(context="Log", questions={"t": {"type": ["number", "null"]}})
        self.assertEqual(result, {"t": None})
        self.assertIn('{"t": null}', client.last_prompts[0])
        self.assertNotIn('{"t":null', client.last_prompts[0])

    def test_a_spaceless_quote_counts_and_the_string_is_written_with_the_space(self):
        class SpacelessQuoteServer(SpacelessNullServer):
            def _request(self, path, payload=None, *, allow_text=False):
                if path == "/generate" and "token_ids_logprob" in payload:
                    ids = payload["token_ids_logprob"]
                    self.payloads.append(payload)
                    return {"meta_info": {"output_token_ids_logprobs": [
                        [[0.0 if t == ord('"') else -9.0, t, "?"] for t in ids]]}}
                return FakeServer._request(self, path, payload, allow_text=allow_text)

        client = TypeLLMClient(model="fake")
        client.sglang = SpacelessQuoteServer()
        result = client.generate(context="Log", questions={"s": {"type": ["string", "null"]}})
        self.assertEqual(result, {"s": "blue"})
        [text] = [p for p in client.sglang.payloads
                  if not isinstance(p["sampling_params"], dict) and "regex" in p["sampling_params"][0]]
        self.assertTrue(text["text"][0].endswith('{"s": "'))

    def test_a_spaceless_null_also_counts_for_strings(self):
        client = TypeLLMClient(model="fake")
        client.sglang = SpacelessNullServer()
        self.assertEqual(client.generate(context="Log", questions={"s": {"type": ["string", "null"]}}), {"s": None})


    def test_null_written_without_the_space_also_counts_as_null(self):
        # ' null' 0.30 + 'null' 0.25 = 0.55 is null, though ' null' alone loses to the rest.
        with patch.dict(fake_vocab.PIECES, {"null": 2827}):
            client = TypeLLMClient(model="fake")
            client.sglang = KeyServer({819: 0.30, 2827: 0.25})
            self.assertEqual(client.sglang.json_value_starts()["null"], [(819, " null"), (2827, "null")])
            result = client.generate(context="Receipt", questions={"note": {"type": ["string", "null"]}})
        self.assertEqual(result, {"note": None})


if __name__ == "__main__":
    unittest.main()
