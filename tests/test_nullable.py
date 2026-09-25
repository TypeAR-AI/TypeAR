import json
import unittest

from typellm import SchemaError, TypeLLMClient, compile_json_schema

from tests.test_batching import FakeServer, width
from tests.test_typellm import FakeSGLang

NULL_ID = 819  # " null", as tokenized after {"name":


def compile_field(field):
    return compile_json_schema({"type": "object", "properties": {"x": field}})[0]


class NullServer(FakeServer):
    """FakeServer whose model prefers null wherever null is offered."""

    def _request(self, path, payload=None, *, allow_text=False):
        if path == "/generate" and "token_ids_logprob" in payload:
            rows = payload["token_ids_logprob"]
            flat = rows if isinstance(rows[0], int) else sum(rows, [])
            if NULL_ID in flat:
                self.payloads.append(payload)
                rows = [rows] if isinstance(rows[0], int) else rows
                out = []
                for ids in rows:
                    pick = (NULL_ID if NULL_ID in ids else ord(" ") if ord(" ") in ids
                            else 1 if 1 in ids else ord("7") if ord("7") in ids else ids[0])
                    out.append({"meta_info": {"output_token_ids_logprobs": [
                        [[0.0 if t == pick else -9.0, t, "?"] for t in ids]]}})
                return out[0] if isinstance(payload["text"], str) else out
        return super()._request(path, payload, allow_text=allow_text)


class CompileTests(unittest.TestCase):
    def test_type_lists_with_null(self):
        self.assertTrue(compile_field({"type": ["string", "null"]}).nullable)
        self.assertTrue(compile_field({"type": ["null", "integer"]}).nullable)
        self.assertFalse(compile_field({"type": "number"}).nullable)
        self.assertEqual(compile_field({"type": ["boolean", "null"]}).choices, (True, False, None))

    def test_enums_allow_null_only_when_listed(self):
        self.assertEqual(compile_field({"type": ["string", "null"], "enum": ["a", None]}).choices, ("a", None))
        self.assertEqual(compile_field({"type": ["string", "null"], "enum": ["a", "b"]}).choices, ("a", "b"))
        with self.assertRaisesRegex(SchemaError, "do not match type"):
            compile_field({"type": "string", "enum": ["a", None]})

    def test_other_type_lists_are_rejected(self):
        for kinds in (["string", "integer"], ["null"], ["string", "null", "integer"], ["null", "null"]):
            with self.subTest(kinds=kinds), self.assertRaisesRegex(SchemaError, r'\[type, "null"\]'):
                compile_field({"type": kinds})


class RuntimeTests(unittest.TestCase):
    def test_nullable_number_can_answer_null(self):
        client = TypeLLMClient(model="fake")
        client.sglang = NullServer()
        result = client.generate(context="Receipt", questions={
            "tip": {"type": ["number", "null"]}, "count": {"type": "integer"},
        })
        self.assertEqual(result, {"tip": None, "count": 7})
        self.assertIn("Return null only if there is no value.", client.last_prompts[0])

    def test_nullable_string_decides_null_before_writing(self):
        client = TypeLLMClient(model="fake")
        client.sglang = NullServer()
        result = client.generate(context="Receipt", questions={
            "note": {"type": ["string", "null"], "maxLength": 20}, "name": {"type": "string"},
        })
        self.assertEqual(result, {"note": None, "name": "blue"})
        # Only the non-null string is generated, as the value of {"name": ...}.
        [text] = [p for p in client.sglang.payloads
                  if not isinstance(p["sampling_params"], dict) and "json_schema" in p["sampling_params"][0]]
        self.assertEqual(width(text), 1)
        schema = json.loads(text["sampling_params"][0]["json_schema"])
        self.assertEqual(schema["properties"], {"name": {"type": "string"}})

    def test_nullable_boolean_scores_null_as_a_choice(self):
        client = TypeLLMClient(model="fake")
        client.sglang = FakeSGLang(selected_ids=[ord("C")])
        result = client.generate(context="Receipt", questions={
            "paid": {"type": ["boolean", "null"], "return_probabilities": True},
        })
        self.assertIsNone(result["paid"]["value"])
        self.assertEqual(set(result["paid"]["probabilities"]), {True, False, None})

    def test_dependents_see_null(self):
        client = TypeLLMClient(model="fake")
        client.sglang = NullServer()
        client.generate(context="Receipt", questions={
            "tip": {"type": ["number", "null"]},
            "tipped": {"type": "boolean", "depends_on": ["tip"]},
        })
        self.assertIn('{"tip": null}', client.last_prompts[1])


if __name__ == "__main__":
    unittest.main()
