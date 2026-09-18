import unittest

from typear import (
    SGLangClient,
    SchemaError,
    TypeARClient,
    compile_json_schema,
)
from typear_sglang import SGLangError, extract_generated_text


class FakeSGLang:
    def __init__(self, selected_ids=None, generated_texts=None):
        self.selected_ids = iter(selected_ids or [32])
        self.generated_texts = iter(generated_texts or [])
        self.prompts = []
        self.cached_prefixes = []
        self.batch_prompts = []
        self.open_prompts = []
        self.open_batch_prompts = []

    def single_token(self, label):
        return ord(label), label

    def score_candidates(self, prefix, candidate_ids):
        self.prompts.append(prefix)
        selected = next(self.selected_ids)
        return (
            {token_id: (0.0 if token_id == selected else -10.0) for token_id in candidate_ids},
            {"cached_tokens": 0},
            0.0,
        )

    def cache_prefix(self, prefix):
        self.cached_prefixes.append(prefix)
        return {"cached_tokens": len(prefix)}

    def score_candidates_batch(self, prefixes, candidate_ids):
        self.batch_prompts.append(list(prefixes))
        scored = []
        for ids in candidate_ids:
            selected = next(self.selected_ids)
            scored.append(
                (
                    {
                        token_id: (0.0 if token_id == selected else -10.0)
                        for token_id in ids
                    },
                    {"cached_tokens": 1},
                )
            )
        return scored, 0.0

    def generate_text(self, prefix, **kwargs):
        self.open_prompts.append((prefix, kwargs))
        return next(self.generated_texts), {"cached_tokens": 2}, 0.0

    def generate_text_batch(self, prefixes, **kwargs):
        self.open_batch_prompts.append((list(prefixes), kwargs))
        return [
            (next(self.generated_texts), {"cached_tokens": 2})
            for _ in prefixes
        ], 0.0


class RecordingSGLangClient(SGLangClient):
    def __init__(self):
        super().__init__()
        self.requests = []

    def _request(self, path, payload=None, *, allow_text=False):
        self.requests.append((path, payload))
        return [
            {
                "meta_info": {
                    "cached_tokens": 64,
                    "output_token_ids_logprobs": [
                        [[-0.1, 65, "A"], [-2.0, 66, "B"]]
                    ],
                }
            },
            {
                "meta_info": {
                    "cached_tokens": 64,
                    "output_token_ids_logprobs": [
                        [[-3.0, 65, "A"], [-0.2, 66, "B"]]
                    ],
                }
            },
        ]


class OpenTextRecordingClient(SGLangClient):
    def __init__(self, response):
        super().__init__()
        self.response = response
        self.requests = []

    def _request(self, path, payload=None, *, allow_text=False):
        self.requests.append((path, payload))
        return self.response


class JsonSchemaCompilerTests(unittest.TestCase):
    def test_enum_with_x_question(self):
        [decision] = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "expense_type": {
                        "type": "string",
                        "enum": ["meal", "travel"],
                        "x-question": "Which expense?",
                    }
                },
            }
        )
        self.assertEqual(decision.question, "Which expense?")
        self.assertEqual(decision.choices, ("meal", "travel"))

    def test_description_then_generated_question_fallback(self):
        decisions = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "integer",
                        "enum": [1, 2],
                        "description": "Pick a priority.",
                    },
                    "size": {"type": "number", "enum": [0.1, 0.5, 1.0]},
                },
            }
        )
        self.assertEqual(decisions[0].question, "Pick a priority.")
        self.assertEqual(decisions[1].question, 'Choose the value for "size".')

    def test_boolean(self):
        [decision] = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "reimbursable": {
                        "type": "boolean",
                        "x-question": "Reimburse it?",
                    }
                },
            }
        )
        self.assertEqual(decision.choices, (True, False))
        self.assertEqual(decision.syntax, "Bool")

    def test_number_enum_preserves_float_values(self):
        [decision] = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "scale": {"type": "number", "enum": [0.1, 0.5, 1.0]}
                },
            }
        )
        self.assertEqual(decision.choices, (0.1, 0.5, 1.0))

    def test_score_expands_to_eleven_levels(self):
        [decision] = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "confidence": {
                        "type": "number",
                        "x-score": True,
                        "x-question": "How confident are you?",
                    }
                },
            }
        )
        self.assertEqual(
            decision.choices,
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.assertEqual(decision.syntax, "Score")

    def test_string_enum_with_other_adds_escape_branch(self):
        [decision] = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "expense_type": {
                        "type": "string",
                        "enum": ["meal", "travel", "equipment"],
                        "x-other": True,
                    }
                },
            }
        )
        self.assertEqual(
            decision.choices, ("meal", "travel", "equipment", "other")
        )
        self.assertTrue(decision.allow_other)

    def test_other_requires_boolean_string_enum_and_reserved_value(self):
        invalid_fields = [
            {"type": "string", "enum": ["x"], "x-other": "yes"},
            {"type": "number", "enum": [1], "x-other": True},
            {"type": "string", "x-other": True},
            {"type": "string", "enum": ["other"], "x-other": True},
        ]
        for field in invalid_fields:
            with self.subTest(field=field), self.assertRaises(
                (SchemaError, NotImplementedError)
            ):
                compile_json_schema(
                    {
                        "type": "object",
                        "properties": {"value": field},
                    }
                )

    def test_property_order_is_decision_order(self):
        decisions = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "second": {"type": "boolean"},
                    "first": {"type": "string", "enum": ["x"]},
                },
            }
        )
        self.assertEqual([decision.name for decision in decisions], ["second", "first"])

    def test_invalid_question_duplicate_enum_and_unsupported_type(self):
        with self.assertRaises(SchemaError):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string", "enum": ["a"], "x-question": 3}
                    },
                }
            )
        with self.assertRaises(SchemaError):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {"x": {"type": "number", "enum": [0.1, 0.1]}},
                }
            )
        with self.assertRaises(NotImplementedError):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {"x": {"type": "array"}},
                }
            )

    def test_required_names_must_exist(self):
        with self.assertRaises(SchemaError):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {"x": {"type": "boolean"}},
                    "required": ["missing"],
                }
            )


class JsonSchemaExecutionTests(unittest.TestCase):
    def test_manual_open_choice_requires_other_value(self):
        from typear import Choice

        with self.assertRaises(ValueError):
            Choice(question="Open?", choices={"A": "known"}, allow_other=True)

    def test_score_uses_a_through_k_and_returns_float(self):
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "x-score": True},
            },
        }
        client = TypeARClient()
        fake = FakeSGLang([ord("K")])
        client.sglang = fake

        [compiled] = client.compile_schema(schema)
        self.assertEqual(list(compiled.choices), list("ABCDEFGHIJK"))
        self.assertEqual(compiled.choices["A"], 0.0)
        self.assertEqual(compiled.choices["K"], 1.0)
        result = client.generate(context="context", schema=schema)
        self.assertEqual(result, {"score": 1.0})

    def test_semantic_result_and_prior_value_in_later_prompt(self):
        schema = {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "enum": [0.1, 0.5, 1.0],
                    "x-question": "Choose a scale.",
                },
                "enabled": {"type": "boolean"},
            },
            "required": ["scale", "enabled"],
        }
        client = TypeARClient()
        fake = FakeSGLang([ord("B"), ord("A")])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"scale": 0.5, "enabled": True})
        self.assertNotIn("A", result)
        self.assertIn('answer="B",\n  value=0.5', fake.prompts[1])

    def test_other_generates_bounded_string_and_conditions_next_decision(self):
        schema = {
            "type": "object",
            "properties": {
                "expense_type": {
                    "type": "string",
                    "enum": ["meal", "travel", "equipment"],
                    "x-other": True,
                    "x-question": "What type of expense is this?",
                },
                "reimbursable": {"type": "boolean"},
            },
        }
        client = TypeARClient(other_max_new_tokens=24)
        fake = FakeSGLang(
            [ord("D"), ord("A")], generated_texts=[' conference registration"ignored']
        )
        client.sglang = fake

        result = client.generate(
            context="context", schema=schema, return_probabilities=True
        )

        self.assertEqual(result["expense_type"]["value"], "conference registration")
        self.assertIn("other", result["expense_type"]["probabilities"])
        self.assertEqual(result["reimbursable"]["value"], True)
        self.assertEqual(fake.open_prompts[0][1]["stop"], '"')
        self.assertEqual(fake.open_prompts[0][1]["max_new_tokens"], 24)
        self.assertIn(
            'answer="D",\n  value="conference registration"', fake.prompts[1]
        )

    def test_batch_prefills_once_and_forks_independent_questions(self):
        schema = {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "enum": [0.1, 0.5, 1.0],
                    "x-question": "Choose a scale.",
                },
                "enabled": {
                    "type": "boolean",
                    "x-question": "Enable it?",
                },
            },
        }
        client = TypeARClient(execution="batch")
        fake = FakeSGLang([ord("B"), ord("A")])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"scale": 0.5, "enabled": True})
        self.assertEqual(fake.cached_prefixes, ["context\n\n"])
        self.assertEqual(len(fake.batch_prompts), 1)
        self.assertEqual(len(fake.batch_prompts[0]), 2)
        self.assertIn("Choose a scale.", fake.batch_prompts[0][0])
        self.assertIn("Enable it?", fake.batch_prompts[0][1])
        self.assertNotIn("value=0.5", fake.batch_prompts[0][1])
        self.assertIsNone(client.last_prompt)
        self.assertEqual(len(client.last_prompts), 2)

    def test_batch_other_values_use_bounded_generation(self):
        schema = {
            "type": "object",
            "properties": {
                "expense_type": {
                    "type": "string",
                    "enum": ["meal"],
                    "x-other": True,
                },
                "department": {
                    "type": "string",
                    "enum": ["sales"],
                    "x-other": True,
                },
            },
        }
        client = TypeARClient(execution="batch")
        fake = FakeSGLang(
            [ord("B"), ord("B")],
            generated_texts=["conference registration", "research"],
        )
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(
            result,
            {"expense_type": "conference registration", "department": "research"},
        )
        self.assertEqual(len(fake.open_batch_prompts), 1)
        self.assertEqual(len(fake.open_batch_prompts[0][0]), 2)
        self.assertIn('value="conference registration"', client.last_prompts[0])
        self.assertIn('value="research"', client.last_prompts[1])

    def test_native_batch_request_uses_per_prompt_candidate_ids(self):
        client = RecordingSGLangClient()

        scored, _ = client.score_candidates_batch(
            ["context + q1", "context + q2"],
            [[65, 66], [65, 66]],
        )

        path, payload = client.requests[0]
        self.assertEqual(path, "/generate")
        self.assertEqual(payload["text"], ["context + q1", "context + q2"])
        self.assertEqual(payload["token_ids_logprob"], [[65, 66], [65, 66]])
        self.assertEqual(scored[0][0], {65: -0.1, 66: -2.0})
        self.assertEqual(scored[1][0], {65: -3.0, 66: -0.2})

    def test_open_generation_uses_quote_stop_and_extracts_text(self):
        client = OpenTextRecordingClient(
            {"text": "conference registration", "meta_info": {"cached_tokens": 64}}
        )

        text, meta, _ = client.generate_text(
            'value="', max_new_tokens=24, temperature=0.2
        )

        path, payload = client.requests[0]
        self.assertEqual(path, "/generate")
        self.assertEqual(payload["sampling_params"]["stop"], ['"'])
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 24)
        self.assertEqual(text, "conference registration")
        self.assertEqual(meta["cached_tokens"], 64)

    def test_generated_text_response_shapes_and_errors(self):
        self.assertEqual(extract_generated_text({"output_text": "x"})[0], "x")
        self.assertEqual(extract_generated_text([{"generated_text": "y"}])[0], "y")
        with self.assertRaises(SGLangError):
            extract_generated_text({"meta_info": {}})


if __name__ == "__main__":
    unittest.main()
