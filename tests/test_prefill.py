import json
import unittest

from typellm import TypeLLMClient

from tests.test_batching import FakeServer


class CloseBraceServer(FakeServer):
    """Numbers decode to 7 and end at "}" rather than the end-of-message token."""

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and "token_ids_logprob" in payload:
            self.payloads.append(payload)
            rows = payload["token_ids_logprob"]
            rows = [rows] if isinstance(rows[0], int) else rows
            out = []
            for ids in rows:
                pick = (ord("}") if ord("}") in ids else ord(" ") if ord(" ") in ids
                        else ord("7") if ord("7") in ids else ids[0])
                out.append({"meta_info": {"output_token_ids_logprobs": [
                    [[0.0 if t == pick else -9.0, t, chr(t)] for t in ids]]}})
            return out[0] if isinstance(payload["text"], str) else out
        return super()._request(path, payload, allow_text=allow_text)


class PrefillTests(unittest.TestCase):
    QUESTIONS = {
        "total": {"type": "number"},
        "item": {"type": "string"},
        "paid": {"type": "boolean"},
    }

    def run_generate(self, questions, **kwargs):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeServer()
        result = client.generate(context="Receipt", questions=questions, **kwargs)
        return client, result

    def test_open_fields_continue_from_a_prefilled_key(self):
        client, result = self.run_generate(self.QUESTIONS)
        self.assertEqual(result, {"total": 7, "item": "blue", "paid": True})
        [text] = [p for p in client.sglang.payloads if not isinstance(p["sampling_params"], dict)
                  and "json_schema" in p["sampling_params"][0]]
        # Strings are generated as the whole one-key object; the grammar fixes the key.
        self.assertEqual(json.loads(text["sampling_params"][0]["json_schema"])["required"], ["item"])
        numbers = [p for p in client.sglang.requests("score") if isinstance(p["text"], str)]
        # The sign is chosen at the key, as the tokenizer splits {"total": 7}; digits follow the space.
        self.assertTrue(numbers[0]["text"].endswith('{"total":'))
        self.assertTrue(numbers[1]["text"].endswith('{"total": '))
        # Choices are scored at {"paid": " — the next token is the label.
        labels = [t for p in client.sglang.requests("score") for t in
                  ([p["text"]] if isinstance(p["text"], str) else p["text"]) if t.endswith('{"paid": "')]
        self.assertEqual(len(labels), 1)

    def test_history_holds_the_closed_object(self):
        client, _ = self.run_generate(self.QUESTIONS, execution="sequential")
        self.assertIn('{"total": 7}', client.last_prompt)
        self.assertIn('{"item": "blue"}', client.last_prompt)

    def test_numbers_can_end_at_the_closing_brace(self):
        client = TypeLLMClient(model="fake")
        client.sglang = CloseBraceServer()
        result = client.generate(context="Receipt", questions={"total": {"type": "integer"}}, execution="sequential")
        self.assertEqual(result, {"total": 7})
        # The decoded prompt ends with the brace the model chose, not the end-of-message token.
        self.assertTrue(client.last_prompt.endswith('{"total": 7}'))

    def test_dag_children_extend_the_closed_parent(self):
        client, _ = self.run_generate({
            "total": {"type": "number"},
            "big": {"type": "boolean", "depends_on": ["total"]},
        })
        child = client.last_prompts[1]
        self.assertIn('{"total": 7', child)
        self.assertTrue(child.startswith(client.last_prompts[0].rsplit('{"total": ', 1)[0]))


if __name__ == "__main__":
    unittest.main()
