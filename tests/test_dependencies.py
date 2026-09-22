import unittest
from unittest.mock import patch

from typellm import TypeLLMClient, SGLangClient, SchemaError, compile_json_schema
from tests.test_typellm import FakeSGLang


class DependencyFake(FakeSGLang):
    thinking = False
    _continuation_parts = SGLangClient._continuation_parts

    def _prepare_answer_prefix(self, prompt):
        return self._finish_thinking(prompt) if self.thinking else prompt
    complete_chat_prefix = SGLangClient.complete_chat_prefix
    extend_chat_prefix = SGLangClient.extend_chat_prefix

    def render_chat(self, messages, *, add_generation_prompt, finish_thinking=True):
        prompt = super().render_chat(messages, add_generation_prompt=add_generation_prompt)
        if self.thinking and add_generation_prompt and finish_thinking:
            return self._finish_thinking(prompt)
        return prompt

    def _finish_thinking(self, prompt):
        return prompt + '<think>retained reasoning</think>'


class DependencyTests(unittest.TestCase):
    def client(self, selected=None):
        client = TypeLLMClient('http://unused')
        client.sglang = DependencyFake(selected or [65] * 20)
        return client

    def test_diamond_forward_reference_and_branch_isolation(self):
        client = self.client()
        questions = {
            'final': {'type': 'boolean', 'depends_on': ['left', 'right']},
            'left': {'type': 'boolean', 'depends_on': ['root']},
            'unrelated': {'type': 'boolean'},
            'right': {'type': 'boolean', 'depends_on': ['root']},
            'root': {'type': 'boolean', 'return_probabilities': True},
        }
        result = client.generate(context='shared', questions=questions)
        self.assertEqual(list(result), list(questions))
        self.assertEqual([len(b) for b in client.sglang.batch_prompts], [2, 2, 1])
        middle = client.sglang.batch_prompts[1]
        for prompt in middle:
            self.assertIn('"root": true', prompt)
            self.assertNotIn('unrelated', prompt)
            self.assertNotIn('probabilities', prompt)
        final = client.sglang.batch_prompts[2][0]
        for name in ('root', 'left', 'right'):
            self.assertIn(f'"{name}": true', final)
        self.assertNotIn('unrelated', final)
        self.assertIsNone(client.last_prompt)
        self.assertIn('name="final"', client.last_prompts[0])
        self.assertTrue(result['root']['value'])

    def test_incremental_prefixes_and_unique_warmups(self):
        client = self.client()
        client.generate(context='long context', questions={
            'a': {'type': 'boolean', 'depends_on': []},
            'b': {'type': 'boolean', 'depends_on': ['a']},
            'c': {'type': 'boolean', 'depends_on': ['a']},
            'd': {'type': 'boolean', 'depends_on': ['b', 'c']},
        })
        a, b, c, d = client.last_prompts
        self.assertTrue(b.startswith(a))
        self.assertTrue(c.startswith(a))
        self.assertTrue(d.startswith(b))
        self.assertNotIn('name="c"', b)
        self.assertNotIn('name="b"', c)
        self.assertEqual(client.sglang.cached_prefixes, [
            '<user>long context</user>', a, b,
        ])
        # Completed prefixes retain the exact prompt sent for scoring.
        for batch, completed in zip(client.sglang.batch_prompts, [[a], [b, c], [d]]):
            for prompt, final in zip(batch, completed):
                self.assertTrue(final.startswith(prompt))

    def test_thinking_prefix_retained_across_dependency(self):
        client = self.client()
        client.sglang.thinking = True
        client.generate(context='', questions={
            'a': {'type': 'boolean'},
            'b': {'type': 'boolean', 'depends_on': ['a']},
        })
        a, b = client.last_prompts
        self.assertTrue(b.startswith(a))
        self.assertEqual(a.count('<think>retained reasoning</think>'), 1)
        self.assertEqual(b.count('<think>retained reasoning</think>'), 2)

    def test_twenty_four_candidates_in_dependency_graph(self):
        client = self.client([ord('X'), ord('A')])
        result = client.generate(context='', questions={
            'pick': {'type': 'integer', 'enum': list(range(24)),
                     'return_probabilities': True},
            'check': {'type': 'boolean', 'depends_on': ['pick']},
        })
        self.assertEqual(result['pick']['value'], 23)
        self.assertEqual(len(result['pick']['probabilities']), 24)
        self.assertEqual(list(client.label_token_map)[:24], list('ABCDEFGHIJKLMNOPQRSTUVWX'))
        self.assertIn('"pick": 23', client.last_prompts[1])
        self.assertTrue(client.last_prompts[1].startswith(client.last_prompts[0]))

    def test_always_thinking_template_continuation_runs_once(self):
        client = SGLangClient(thinking=False)

        class Tokenizer:
            chat_template = 'test'
            def apply_chat_template(self, messages, **kwargs):
                history = ''.join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
                return history + ('<assistant><think>' if kwargs['add_generation_prompt'] else '')

        client._chat_tokenizer = Tokenizer()
        with patch.object(client, '_finish_thinking', side_effect=lambda p: p + 'reasoning</think>') as finish:
            prompt = client.render_chat([{'role': 'user', 'content': 'root'}], add_generation_prompt=True)
            parent = client.complete_chat_prefix(prompt, 'A')
            child = client.extend_chat_prefix(parent, 'child')
            self.assertTrue(child.startswith(parent))
            self.assertEqual(finish.call_count, 2)
            self.assertTrue(finish.call_args.args[0].startswith(parent))

    def test_cli_defaults_to_auto_and_accepts_dag(self):
        from typellm.cli import main
        for args, mode in [(['typellm'], 'auto'), (['typellm', '--execution', 'dag'], 'dag')]:
            with patch('sys.argv', args), patch('typellm.cli.TypeLLMClient') as factory, patch('builtins.print'), patch('typellm.cli.logging.basicConfig'):
                factory.return_value.generate.return_value = {}
                main()
                self.assertEqual(factory.call_args.kwargs['execution'], mode)

    def test_invalid_graph_before_tokenizer_or_inference(self):
        cases = [
            {'a': {'type': 'boolean', 'depends_on': value}}
            for value in (None, 'b', [1], [''], ['a'], ['missing'], ['b', 'b'])
        ]
        cases.append({'a': {'type': 'boolean', 'depends_on': ['b']},
                      'b': {'type': 'boolean', 'depends_on': ['a']}})
        for questions in cases:
            with self.subTest(questions=questions), self.assertRaises(SchemaError):
                compile_json_schema({'type': 'object', 'properties': questions})

    def test_explicit_modes_and_empty_dependencies(self):
        for mode in ('sequential', 'batch'):
            client = self.client()
            with self.assertRaises(SchemaError):
                client.generate(context='', questions={'a': {'type': 'boolean', 'depends_on': []}}, execution=mode)
            self.assertFalse(client.sglang.batch_prompts)
            self.assertFalse(client.sglang.prompts)
        client = self.client()
        client.generate(context='', questions={'a': {'type': 'boolean', 'depends_on': []}, 'b': {'type': 'boolean'}})
        self.assertEqual(len(client.sglang.batch_prompts[0]), 2)

    def test_numeric_dependency_uses_semantic_value(self):
        client = self.client([ord('7'), 3, 65])
        result = client.generate(context='', questions={
            'number': {'type': 'integer', 'depends_on': []},
            'check': {'type': 'boolean', 'depends_on': ['number']},
        })
        self.assertEqual(result, {'number': 7, 'check': True})
        self.assertTrue(client.last_prompts[1].startswith(client.last_prompts[0]))
        self.assertIn('"number": 7', client.sglang.batch_prompts[0][0])

    def test_text_dependency_and_schema_interface(self):
        client = self.client()
        client.sglang.generate_texts = lambda prompts, limits, **kwargs: ['hello "世界"'] * len(prompts)
        result = client.generate(context='', schema={'type': 'object', 'properties': {
            'text': {'type': 'string'},
            'check': {'type': 'boolean', 'depends_on': ['text']},
        }})
        self.assertEqual(result['text'], 'hello "世界"')
        self.assertTrue(client.last_prompts[1].startswith(client.last_prompts[0]))
        self.assertIn('"text": "hello \\"世界\\""', client.sglang.batch_prompts[0][0])

    def test_auto_batches_roots_and_explicit_sequential_is_preserved(self):
        for mode, batch in [(None, True), ('auto', True), ('dag', True), ('sequential', False)]:
            client = self.client()
            client.generate(context='', questions={'a': {'type': 'boolean'}, 'b': {'type': 'boolean'}}, execution=mode)
            self.assertEqual(bool(client.sglang.batch_prompts), batch)
            self.assertEqual(bool(client.sglang.prompts), not batch)


if __name__ == '__main__':
    unittest.main()
