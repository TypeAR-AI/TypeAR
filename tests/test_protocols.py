import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from typellm import SGLangClient, SGLangError
from typellm.protocol import detect_protocol
from tests.test_typellm import FakeChatTokenizer


class ProtocolTokenizer(FakeChatTokenizer):
    def __init__(self, family):
        super().__init__()
        self.family = family
        self.chat_template = {
            'gemma': '<|turn>model <|channel>thought',
            'minicpm5': '<|im_start|>assistant <|im_end|>',
            'ling': '<role>ASSISTANT</role><|role_end|>',
            'ring': '<role>ASSISTANT</role><think>',
        }[family]
        self.eos_token, self.eos_token_id = '</s>', 1
        self.special = {'<turn|>': 2, '<|im_end|>': 3, '<|role_end|>': 4}

    def encode(self, text, *, add_special_tokens=False):
        return [self.special[text]] if text in self.special else super().encode(text)

    def decode(self, ids, **kwargs):
        return {v: k for k, v in self.special.items()}.get(ids[0], '')

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        close = {'minicpm5': '<|im_end|>\n',
                 'ling': '<|role_end|>', 'ring': ''}[self.family]
        text = ''.join(f"<{m['role']}>{m['content']}{close}" for m in messages)
        if kwargs['add_generation_prompt']:
            text += '<assistant>'
            if self.family == 'ring' or (self.family == 'minicpm5' and kwargs['enable_thinking']):
                text += '<think>\n'
        return text


class ProtocolTests(unittest.TestCase):
    def client(self, family, thinking=False, response=None):
        client = SGLangClient(thinking=thinking, thinking_budget=64)
        client._chat_tokenizer = ProtocolTokenizer(family)
        client._context_length_cache = 8192
        end = '</think>'
        client._request = Mock(return_value=response or {'text': 'REASONING' + end + 'discard'})
        return client

    def test_tokenizer_loading_disables_custom_model_code(self):
        tokenizer = ProtocolTokenizer('minicpm5')
        auto = Mock()
        auto.from_pretrained.return_value = tokenizer
        client = SGLangClient(tokenizer='/tokenizer')
        with patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoTokenizer=auto)}):
            self.assertIs(client._get_chat_tokenizer(), tokenizer)
            self.assertIs(client._get_chat_tokenizer(), tokenizer)
        auto.from_pretrained.assert_called_once_with('/tokenizer', trust_remote_code=False)

    def test_template_turn_end_takes_precedence_over_eos(self):
        for family, expected in [('minicpm5', '<|im_end|>'), ('ling', '<|role_end|>'), ('ring', '</s>')]:
            client = self.client(family)
            self.assertEqual(client.end_of_message_token()[1], expected)

    def test_multi_token_terminator_is_rejected(self):
        client = self.client('minicpm5')
        client._chat_tokenizer.encode = lambda *args, **kwargs: [1, 2]
        with self.assertRaisesRegex(SGLangError, 'one exact token'):
            client.end_of_message_token()

    def test_gemma_is_rejected_before_inference(self):
        for thinking in (False, True):
            client = self.client('gemma', thinking)
            with self.assertRaisesRegex(SGLangError, 'Gemma 4.*not supported'):
                client.render_chat([], add_generation_prompt=True)
            client._request.assert_not_called()

    def test_native_turn_stop_recovers_preserved_and_filtered_terminators(self):
        for family, ending, token in [('minicpm5', '<|im_end|>', 3)]:
            for matched in (ending, token):
                for suffix in ('', ending + '\n'):
                    with self.subTest(family=family, matched=matched, suffix=suffix):
                        client = self.client(family, True, {
                            'text': 'Keep reasoning' + suffix,
                            'meta_info': {'finish_reason': {'type': 'stop', 'matched': matched}},
                        })
                        prefix = detect_protocol(client._chat_tokenizer).thinking_open + '\n'
                        with self.assertLogs('typellm', level='INFO') as logs:
                            prompt = client._finish_thinking(prefix)
                        self.assertTrue(prompt.startswith(prefix + 'Keep reasoning'))
                        self.assertNotIn(ending, prompt)
                        self.assertTrue(prompt.rstrip().endswith(detect_protocol(client._chat_tokenizer).thinking_close))
                        self.assertIn('native turn terminator', logs.output[0])

    def test_unknown_empty_or_aborted_thinking_is_not_recovered(self):
        for text, finish in [
            ('Partial', {'type': 'stop', 'matched': 'unknown'}),
            ('Partial', {'type': 'stop'}),
            ('Partial', {}),
            ('<|im_end|>', {'type': 'stop', 'matched': 3}),
            ('Partial<|im_end|>', {'type': 'abort', 'matched': 3}),
            ('Partial<|im_end|>', {'type': 'error', 'matched': 3}),
        ]:
            with self.subTest(text=text, finish=finish):
                client = self.client('minicpm5', True, {'text': text, 'meta_info': {'finish_reason': finish}})
                with self.assertRaises(SGLangError):
                    client._finish_thinking('<think>\n')

    def test_minicpm5_switch_and_ring_always_thinking(self):
        for family in ('minicpm5', 'ring'):
            for thinking in (False, True):
                client = self.client(family, thinking)
                prompt = client.render_chat([], add_generation_prompt=True)
                expected = thinking or family == 'ring'
                self.assertEqual(client._request.called, expected)
                parent = client.complete_chat_prefix(prompt, 'A')
                self.assertEqual('REASONING' in parent, expected)
                self.assertTrue(client.extend_chat_prefix(parent, 'next').startswith(parent))

    def test_non_thinking_templates_reject_requested_thinking(self):
        for family in ('ling',):
            client = self.client(family, True)
            with self.assertRaisesRegex(SGLangError, 'may not support thinking'):
                client.render_chat([], add_generation_prompt=True)
            client._request.assert_not_called()



if __name__ == '__main__':
    unittest.main()
