import math
import random
import unittest
from unittest.mock import patch

from typellm import TypeLLMClient, SchemaError, compile_json_schema
from typellm.runtime import Choice, _choice_orderings
from tests.test_dependencies import DependencyFake


class PermutationTests(unittest.TestCase):
    def run_case(self, execution, values, **options):
        client = TypeLLMClient(seed=42, execution=execution, **options)
        client.sglang = DependencyFake([65] * 100)
        fields = {
            'roll': {'type': 'string' if isinstance(values[0], str) else 'integer',
                     'enum': values, 'permutations': 'all', 'return_probabilities': True},
            'flag': {'type': 'boolean'},
        }
        if execution == 'dag':
            fields['flag']['depends_on'] = ['roll']
        return client, client.generate(context='test', questions=fields)

    def test_position_bias_cancels_and_non_enum_runs_once(self):
        for execution in ('batch', 'sequential', 'dag'):
            for values in (['one', 'two', 'three'], [1, 2, 3]):
                with self.subTest(execution=execution, values=values):
                    client, result = self.run_case(execution, values)
                    for probability in result['roll']['probabilities'].values():
                        self.assertAlmostEqual(probability, 1 / 3)
                    self.assertTrue(result['flag'])
                    count = sum(map(len, client.sglang.batch_prompts)) + len(client.sglang.prompts)
                    self.assertEqual(count, 7)
                    if execution == 'dag':
                        self.assertIn('"roll":', client.sglang.batch_prompts[-1][0])
                        self.assertNotIn('probabilities', client.sglang.batch_prompts[-1][0])

    def test_sampling_uses_averaged_distribution(self):
        with patch('typellm.runtime._sample', return_value='B') as sample:
            _client, result = self.run_case('batch', ['one', 'two', 'three'], mode='sample', temperature=.5)
        self.assertEqual(result['roll']['value'], 'two')
        self.assertEqual(sample.call_count, 2)  # one final choice plus the boolean
        for p in sample.call_args_list[0].args[0].values():
            self.assertAlmostEqual(p, 1 / 3)

    def test_sampling_unique_reproducible_without_factorial_allocation(self):
        d = Choice('q', dict(zip('ABCDEFGHIJKLMNOPQRSTUVWX', range(24))), permutations=8)
        first = _choice_orderings(d, random.Random(42))
        second = _choice_orderings(d, random.Random(42))
        self.assertEqual(first, second)
        self.assertEqual(len({order for _, order in first}), 8)
        self.assertNotEqual(first, _choice_orderings(d, random.Random(43)))

    def test_validation_before_tokenizer_or_inference(self):
        invalid = [
            {'type': 'boolean', 'permutations': 1},
            {'type': 'string', 'permutations': 2},
            {'type': 'number', 'permutations': 2},
        ] + [{'type': 'string', 'enum': ['a', 'b'], 'permutations': v}
             for v in (0, -1, True, False, 2.0, None, '8', {}, [])]
        invalid.append({'type': 'integer', 'enum': list(range(7)), 'permutations': 'all'})
        client = TypeLLMClient()
        with patch.object(client.sglang, 'single_token') as tokens:
            for field in invalid:
                with self.subTest(field=field), self.assertRaises(SchemaError):
                    client.generate(context='x', questions={'x': field})
            tokens.assert_not_called()

    def test_plain_output_boolean_enum_and_clamped_budget(self):
        client = TypeLLMClient()
        client.sglang = DependencyFake([65] * 3)
        result = client.generate(context='x', questions={
            'x': {'type': 'boolean', 'enum': [False, True], 'permutations': 100},
            'y': {'type': 'string', 'enum': ['a'], 'permutations': 'all'},
        })
        self.assertIsInstance(result['x'], bool)
        self.assertEqual(result['y'], 'a')
        self.assertEqual(len(client.sglang.batch_prompts[0]), 3)

    def test_average_probabilities_not_logits(self):
        client = TypeLLMClient()
        fake = DependencyFake()
        client.sglang = fake
        with patch.object(fake, 'score_candidates_batch', return_value=([
            ({65: math.log(.9), 66: math.log(.1)}, {}),
            ({65: math.log(.7), 66: math.log(.3)}, {}),
        ], 0.0)):
            result = client.generate(context='x', questions={'x': {
                'type': 'string', 'enum': ['a', 'b'], 'permutations': 'all',
                'return_probabilities': True}})
        self.assertAlmostEqual(result['x']['probabilities']['a'], .6)
        self.assertAlmostEqual(result['x']['probabilities']['b'], .4)
        self.assertEqual(result['x']['value'], 'a')

    def test_one_matches_default(self):
        outputs = []
        for setting in ({}, {'permutations': 1}):
            client = TypeLLMClient()
            client.sglang = DependencyFake([66])
            outputs.append(client.generate(context='x', questions={'x': {
                'type': 'string', 'enum': ['a', 'b'], 'return_probabilities': True, **setting}}))
        self.assertEqual(*outputs)


if __name__ == '__main__':
    unittest.main()
