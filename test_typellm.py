import unittest

from typellm import (
    SGLangClient,
    SchemaError,
    TypeLLMClient,
    compile_json_schema,
)
from typellm_numeric import build_numeric_token_table


class FakeSGLang:
    def __init__(self, selected_ids=None, numeric_pieces=None):
        self.selected_ids = iter(selected_ids or [32])
        self.prompts = []
        self.cached_prefixes = []
        self.batch_prompts = []
        self.candidate_sets = []
        self.numeric_pieces = numeric_pieces or [
            (ord(piece), piece) for piece in '-0123456789."'
        ]

    def render_chat(self, messages, *, add_generation_prompt):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def end_of_message_token(self):
        return 3, "<eom>"

    def single_token(self, label):
        return ord(label), label

    def numeric_token_pieces(self):
        return self.numeric_pieces

    def score_candidates(self, prefix, candidate_ids):
        self.prompts.append(prefix)
        self.candidate_sets.append(list(candidate_ids))
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


class FakeChatTokenizer:
    chat_template = "template"
    eos_token_id = 248046
    eos_token = "<|im_end|>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered-chat"


class JsonSchemaCompilerTests(unittest.TestCase):
    def test_instructions_take_priority_over_description(self):
        for spec in [{"type": "boolean"}, {"type": "integer"},
                     {"type": "number"}, {"type": "string", "enum": ["a", "b"]}]:
            field = {**spec, "instructions": "Select the value.", "description": "Fallback"}
            [decision] = compile_json_schema({"type": "object", "properties": {"value": field}})
            self.assertEqual(decision.question, "Select the value.")

    def test_question_aliases_are_rejected_even_with_instructions(self):
        for old_key in ["question", "x-question"]:
            for extra in [{}, {"instructions": "New wording"}]:
                field = {"type": "boolean", old_key: "Old wording", **extra}
                with self.subTest(old_key=old_key, extra=extra):
                    with self.assertRaisesRegex(SchemaError, "use instructions"):
                        compile_json_schema({"type": "object", "properties": {"paid": field}})
                    client = TypeLLMClient()
                    for kwargs in [
                        {"questions": {"paid": field}},
                        {"schema": {"properties": [{**field, "type": "bool", "name": "paid"}]}},
                    ]:
                        with self.assertRaisesRegex(SchemaError, "use instructions"):
                            client.generate(context="Paid", **kwargs)

    def test_invalid_instructions_are_rejected(self):
        for value in [None, 1, True, [], {}]:
            with self.subTest(value=value), self.assertRaisesRegex(SchemaError, "instructions"):
                compile_json_schema({"type": "object", "properties": {
                    "x": {"type": "boolean", "instructions": value}
                }})

    def test_legacy_list_accepts_instructions(self):
        client = TypeLLMClient()
        client.sglang = FakeSGLang()
        schema = {"properties": [{"name": "paid", "type": "bool", "instructions": "Is it paid?"}]}
        self.assertEqual(client.compile_schema(schema)[0].question, "Is it paid?")

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

    def test_table_comes_from_exact_decoded_model_tokens(self):
        class FakeTokenizer:
            def get_vocab(self, with_added_tokens=True):
                return {
                    "five": 5,
                    "fifty_four": 54,
                    "finished": 55,
                    "bad_suffix": 56,
                    "bad_quote": 57,
                    "repeated_minus": 58,
                    "repeated_dot": 59,
                }

            def decode(self, ids, skip_special_tokens=False):
                return {
                    5: "5",
                    54: "54",
                    55: '54"',
                    56: "12kg",
                    57: '"42',
                    58: "--",
                    59: '.."',
                }[ids[0]]

        self.assertEqual(
            build_numeric_token_table(FakeTokenizer()),
            [(5, "5"), (54, "54"), (55, '54"')],
        )

    def test_boolean(self):
        [decision] = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "reimbursable": {
                        "type": "boolean",
                        "instructions": "Reimburse it?",
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

    def test_open_integer_and_number_compile_with_bounds(self):
        decisions = compile_json_schema(
            {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "ratio": {"type": "number", "minimum": -1.0},
                },
            }
        )
        self.assertEqual(decisions[0].numeric_type, "integer")
        self.assertEqual(decisions[0].syntax, "Integer")
        self.assertEqual(decisions[0].choices, ())
        self.assertEqual((decisions[0].minimum, decisions[0].maximum), (0, 100))
        self.assertEqual(decisions[1].numeric_type, "number")

    def test_open_numeric_rejects_invalid_bounds(self):
        for field in (
            {"type": "integer", "minimum": "zero"},
            {"type": "number", "minimum": 2, "maximum": 1},
            {"type": "number", "maximum": float("inf")},
        ):
            with self.subTest(field=field), self.assertRaises(SchemaError):
                compile_json_schema(
                    {"type": "object", "properties": {"value": field}}
                )

    def test_x_score_is_rejected_in_favor_of_number_enum(self):
        with self.assertRaisesRegex(SchemaError, "x-score.*number enum"):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "confidence": {"type": "number", "x-score": True}
                    },
                }
            )

    def test_x_other_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "x-other.*not supported"):
            compile_json_schema(
                {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "string",
                            "enum": ["known"],
                            "x-other": True,
                        }
                    },
                }
            )

    def test_enum_is_limited_to_sixteen_values(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "integer", "enum": list(range(16))}
            },
        }
        [decision] = compile_json_schema(schema)
        self.assertEqual(len(decision.choices), 16)
        schema["properties"]["value"]["enum"].append(16)
        with self.assertRaisesRegex(SchemaError, "maximum is 16"):
            compile_json_schema(schema)

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
                        "x": {"type": "string", "enum": ["a"], "instructions": 3}
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


class QuestionsInterfaceTests(unittest.TestCase):
    def test_state_and_context_produce_identical_prompts(self):
        questions = {"paid": {"type": "boolean", "instructions": "Is it paid?"}}
        for execution in ["sequential", "batch"]:
            for text in ["Receipt", ""]:
                clients = [TypeLLMClient(execution=execution), TypeLLMClient(execution=execution)]
                for client in clients:
                    client.sglang = FakeSGLang([ord("A")])
                a = clients[0].generate(state=text, questions=questions)
                b = clients[1].generate(context=text, questions=questions)
                self.assertEqual(a, b)
                self.assertEqual(clients[0].last_prompts, clients[1].last_prompts)

    def test_state_rejects_conflicts_missing_and_invalid_types(self):
        from unittest.mock import Mock
        for kwargs in [{}, {"state": "x", "context": "x"},
                       {"state": "", "context": ""}, {"state": 1},
                       {"state": {}}, {"context": []}]:
            client = TypeLLMClient()
            client.sglang = Mock()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                client.generate(questions={"paid": {"type": "boolean"}}, **kwargs)
            self.assertEqual(client.sglang.mock_calls, [])

    def test_run_schema_supports_state_with_both_input_formats(self):
        from unittest.mock import patch
        from typellm import run_schema
        questions = {"paid": {"type": "boolean"}}
        for kwargs in [{"questions": questions}, {"schema": {"type": "object", "properties": questions}}]:
            with patch("typellm_runtime.SGLangClient", return_value=FakeSGLang([ord("A")])):
                self.assertEqual(run_schema(state="Paid", **kwargs), {"paid": True})
        with self.assertRaises(ValueError):
            run_schema("Paid", state="Paid", questions=questions)

    def test_questions_match_schema_in_both_modes(self):
        questions = {
            "expense": {"type": "string", "enum": ["meal", "travel"], "instructions": "Classify."},
            "paid": {"type": "boolean", "instructions": "Is it paid?"},
        }
        for execution in ["sequential", "batch"]:
            clients = [TypeLLMClient(execution=execution), TypeLLMClient(execution=execution)]
            for client in clients:
                client.sglang = FakeSGLang([ord("B"), ord("A")])
            new = clients[0].generate(context="Receipt", questions=questions)
            old = clients[1].generate(context="Receipt", schema={"type": "object", "properties": questions})
            self.assertEqual(new, {"expense": "travel", "paid": True})
            self.assertEqual(new, old)
            self.assertEqual(clients[0].last_prompts, clients[1].last_prompts)

    def test_questions_numeric_and_reserved_field_names(self):
        client = TypeLLMClient()
        client.sglang = FakeSGLang([ord("7"), 3, ord("A")])
        result = client.generate(context="Seven", questions={
            "type": {"type": "integer", "instructions": "Extract the number."},
            "properties": {"type": "boolean", "instructions": "Is it seven?"},
        })
        self.assertEqual(result, {"type": 7, "properties": True})

    def test_questions_invalid_inputs_fail_before_network(self):
        for kwargs in [{}, {"questions": {}, "schema": {}}, {"questions": {}},
                       {"questions": []}, {"questions": "bad"}, {"questions": {"x": None}}]:
            client = TypeLLMClient()
            with self.subTest(kwargs=kwargs), self.assertRaises(SchemaError):
                client.generate(context="Context", **kwargs)

    def test_run_schema_keeps_positional_schema_and_accepts_questions(self):
        from unittest.mock import patch
        from typellm import run_schema
        questions = {"paid": {"type": "boolean"}}
        with patch("typellm_runtime.SGLangClient", return_value=FakeSGLang([ord("A")])):
            new = run_schema("Context", questions=questions)
        with patch("typellm_runtime.SGLangClient", return_value=FakeSGLang([ord("A")])):
            old = run_schema("Context", {"type": "object", "properties": questions})
        self.assertEqual(new, old)
        self.assertEqual(new, {"paid": True})


class ThinkingTests(unittest.TestCase):
    def make_client(self, response=None):
        from unittest.mock import Mock
        client = SGLangClient(thinking=True, thinking_budget=128)
        tokenizer = FakeChatTokenizer()
        tokenizer.apply_chat_template = Mock(return_value="assistant\n<think>\n")
        client._chat_tokenizer = tokenizer
        client._request = Mock(return_value=response or {"text": "Work done. </think>illegal answer"})
        return client

    def test_thinking_stops_before_constrained_answer(self):
        client = self.make_client()
        prompt = client.render_chat([], add_generation_prompt=True)
        self.assertEqual(prompt, "assistant\n<think>\nWork done. </think>\n\n")
        self.assertNotIn("illegal answer", prompt)
        params = client._request.call_args.args[1]["sampling_params"]
        self.assertEqual(params["max_new_tokens"], 128)
        self.assertEqual(params["stop"], ["</think>"])
        self.assertTrue(params["no_stop_trim"])
        self.assertTrue(client._chat_tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_incomplete_or_empty_thinking_returns_no_answer(self):
        from typellm import SGLangError
        for response in [{"text": "not finished"}, {"text": "</think>"}, {"text": None}, []]:
            client = self.make_client()
            client._request.return_value = response
            with self.subTest(response=response), self.assertRaises(SGLangError):
                client.render_chat([], add_generation_prompt=True)

    def test_unsupported_template_does_not_generate(self):
        from typellm import SGLangError
        client = self.make_client()
        client._chat_tokenizer.apply_chat_template.return_value = "<think></think>"
        with self.assertRaisesRegex(SGLangError, "native chat template"):
            client.render_chat([], add_generation_prompt=True)
        client._request.assert_not_called()

    def test_history_render_never_runs_thinking(self):
        client = self.make_client()
        client.render_chat([], add_generation_prompt=False)
        client._request.assert_not_called()
        self.assertFalse(client._chat_tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])

    def test_public_configuration_and_validation(self):
        self.assertFalse(TypeLLMClient().sglang.thinking)
        client = TypeLLMClient(thinking=True, thinking_budget=256)
        self.assertTrue(client.sglang.thinking)
        self.assertEqual(client.sglang.thinking_budget, 256)
        for kwargs in [{"thinking": "false"}, {"thinking": 1}, {"thinking_budget": 0},
                       {"thinking_budget": True}, {"thinking_budget": 1.5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TypeLLMClient(**kwargs)


class JsonSchemaExecutionTests(unittest.TestCase):
    def test_chat_template_disables_thinking_and_uses_native_eom(self):
        client = SGLangClient()
        tokenizer = FakeChatTokenizer()
        client._chat_tokenizer = tokenizer

        rendered = client.render_chat(
            [{"role": "user", "content": "question"}],
            add_generation_prompt=True,
        )

        self.assertEqual(rendered, "rendered-chat")
        self.assertEqual(tokenizer.calls[0][1]["enable_thinking"], False)
        self.assertTrue(tokenizer.calls[0][1]["add_generation_prompt"])
        self.assertEqual(
            client.end_of_message_token(), (248046, "<|im_end|>")
        )

    def test_manual_choice_is_limited_to_sixteen_values(self):
        from typellm import Choice

        with self.assertRaisesRegex(ValueError, "maximum is 16"):
            Choice(
                question="Too many?",
                choices={str(index): index for index in range(17)},
            )

    def test_eleven_value_number_enum_uses_a_through_k(self):
        schema = {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [
                        0.0,
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.0,
                    ],
                },
            },
        }
        client = TypeLLMClient()
        fake = FakeSGLang([ord("K")])
        client.sglang = fake

        [compiled] = client.compile_schema(schema)
        self.assertEqual(list(compiled.choices), list("ABCDEFGHIJK"))
        self.assertEqual(compiled.choices["A"], 0.0)
        self.assertEqual(compiled.choices["K"], 1.0)
        self.assertIn(
            "Answer the question using only the best label.",
            compiled.opening_text(),
        )
        result = client.generate(context="context", schema=schema)
        self.assertEqual(result, {"score": 1.0})

    def test_semantic_result_and_prior_value_in_later_prompt(self):
        schema = {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "enum": [0.1, 0.5, 1.0],
                    "instructions": "Choose a scale.",
                },
                "enabled": {"type": "boolean"},
            },
            "required": ["scale", "enabled"],
        }
        client = TypeLLMClient()
        fake = FakeSGLang([ord("B"), ord("A")])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"scale": 0.5, "enabled": True})
        self.assertNotIn("A", result)
        self.assertIn("<assistant>B</assistant>", fake.prompts[1])

    def test_open_integer_is_constrained_per_character(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "instructions": "How many items?",
                },
                "enabled": {"type": "boolean"},
            },
        }
        client = TypeLLMClient()
        fake = FakeSGLang([ord("4"), ord("2"), 3, ord("A")])
        client.sglang = fake

        result = client.generate(
            context="context", schema=schema
        )

        self.assertEqual(result["count"], 42)
        self.assertEqual(result["enabled"], True)
        self.assertNotIn(3, fake.candidate_sets[0])
        self.assertIn(3, fake.candidate_sets[1])
        self.assertIn(
            "Return only a JSON number without a decimal point or exponent notation.",
            fake.prompts[0],
        )
        self.assertIn("<assistant>42</assistant>", fake.prompts[-1])

    def test_open_integer_can_finish_with_one_multi_character_model_token(self):
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "integer",
                    "instructions": "What is 127 multiplied by 43?",
                }
            },
        }
        client = TypeLLMClient()
        fake = FakeSGLang(
            [9001, 3],
            numeric_pieces=[
                (9001, "5461"),
                (9002, "127"),
                (9003, "12kg"),
            ],
        )
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"answer": 5461})
        self.assertEqual(fake.candidate_sets, [[9001, 9002], [9001, 9002, 3]])
        self.assertIn("<assistant>5461<eom>", client.last_prompt)

    def test_open_float_supports_sign_decimal_and_message_termination(self):
        schema = {
            "type": "object",
            "properties": {"temperature": {"type": "number"}},
        }
        client = TypeLLMClient()
        fake = FakeSGLang(
            [ord("-"), ord("0"), ord("."), ord("7"), ord("5"), 3]
        )
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"temperature": -0.75})
        self.assertIn(
            "Return only a JSON number without exponent notation.",
            client.last_prompt,
        )
        self.assertIn("<assistant>-0.75<eom>", client.last_prompt)

    def test_open_numeric_batch_preserves_schema_order(self):
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"},
            },
        }
        client = TypeLLMClient(execution="batch")
        fake = FakeSGLang([ord("7"), 3, ord("A")])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"count": 7, "enabled": True})
        self.assertIn("<assistant>7</assistant>", client.last_prompts[0])
        self.assertIn("<assistant>A</assistant>", client.last_prompts[1])

    def test_batch_prefills_once_and_forks_independent_questions(self):
        schema = {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "enum": [0.1, 0.5, 1.0],
                    "instructions": "Choose a scale.",
                },
                "enabled": {
                    "type": "boolean",
                    "instructions": "Enable it?",
                },
            },
        }
        client = TypeLLMClient(execution="batch")
        fake = FakeSGLang([ord("B"), ord("A")])
        client.sglang = fake

        result = client.generate(context="context", schema=schema)

        self.assertEqual(result, {"scale": 0.5, "enabled": True})
        self.assertEqual(fake.cached_prefixes, ["<user>context</user>"])
        self.assertEqual(len(fake.batch_prompts), 1)
        self.assertEqual(len(fake.batch_prompts[0]), 2)
        self.assertIn("Choose a scale.", fake.batch_prompts[0][0])
        self.assertIn("Enable it?", fake.batch_prompts[0][1])
        self.assertNotIn("value=0.5", fake.batch_prompts[0][1])
        self.assertIsNone(client.last_prompt)
        self.assertEqual(len(client.last_prompts), 2)

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

if __name__ == "__main__":
    unittest.main()
