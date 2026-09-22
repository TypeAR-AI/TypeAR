"""CPU-only checks against pinned official tokenizers; no model inference."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typellm import SGLangClient
from typellm.numeric import build_numeric_token_table
from typellm.protocol import detect_protocol

MODELS = ['openbmb/MiniCPM5-1B',
          'inclusionAI/Ling-mini-2.0', 'inclusionAI/Ring-mini-2.0']


def probe(model):
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(model).sha
    snapshot = snapshot_download(model, revision=revision, allow_patterns=[
        'config.json', 'tokenizer.json', 'tokenizer_config.json', 'tokenizer.model',
        'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja',
        'chat_templates/*.jinja',
    ])
    loader = SGLangClient(tokenizer=snapshot)
    tok = loader._get_chat_tokenizer()
    protocol = detect_protocol(tok)
    labels = []
    for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
        ids = tok.encode(label, add_special_tokens=False)
        if len(ids) == 1 and tok.decode(ids, skip_special_tokens=False) == label:
            labels.append((label, ids[0]))
    backend = tok.backend_tokenizer
    numeric = build_numeric_token_table(backend)
    row = {'model': model, 'revision': revision, 'protocol': protocol.name,
           'labels': labels, 'enough_labels_for_24': len({v for _, v in labels}) >= 24,
           'numeric_token_count': len(numeric), 'modes': []}
    row['template_sha256'] = hashlib.sha256(tok.get_chat_template().encode()).hexdigest()
    for thinking in (False, True):
        calls = []
        client = SGLangClient(thinking=thinking, thinking_budget=64)
        client._chat_tokenizer = tok
        client._context_length_cache = 32768
        # Only the server response is simulated; tokenizer and native templates
        # are real. This is not an accuracy or GPU-cache test.
        def request(path, payload=None, **kwargs):
            calls.append(payload)
            return {'text': 'PROBE_REASONING' + protocol.thinking_close}
        client._request = request
        mode = {'thinking': thinking}
        try:
            prompt = client.render_chat([{'role': 'user', 'content': 'Context'},
                                        {'role': 'user', 'content': 'Return A.'}],
                                       add_generation_prompt=True)
            parent = client.complete_chat_prefix(prompt, 'A')
            child = client.extend_chat_prefix(parent, 'Return B.')
            child_history = client.complete_chat_prefix(child, 'B')
            parent_ids = tok.encode(parent, add_special_tokens=False)
            child_ids = tok.encode(child, add_special_tokens=False)
            common = 0
            for a, b in zip(parent_ids, child_ids):
                if a != b:
                    break
                common += 1
            turn_id, turn_text = client.end_of_message_token()
            mode.update(ok=True, turn_end=turn_text, turn_end_id=turn_id,
                        reasoning_requests=len(calls),
                        parent_prefix_preserved=child.startswith(parent),
                        parent_tokens=len(parent_ids), shared_tokens=common,
                        history_contains_reasoning='PROBE_REASONING' in child_history,
                        prompt_tail=prompt[-200:], child_tail=child[-200:])
        except Exception as exc:
            mode.update(ok=False, error=f'{type(exc).__name__}: {exc}')
        row['modes'].append(mode)
    return row


def checks_passed(row):
    if 'error' in row or not row['enough_labels_for_24'] or not row['numeric_token_count']:
        return False
    for mode in row['modes']:
        unsupported = mode['thinking'] and row['model'] in (
            'inclusionAI/Ling-mini-2.0',
        )
        if unsupported:
            if mode['ok'] or 'may not support thinking' not in mode.get('error', ''):
                return False
            continue
        if not mode['ok'] or not mode['parent_prefix_preserved'] or mode['shared_tokens'] != mode['parent_tokens']:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=MODELS)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        'validation': 'CPU only; reasoning HTTP responses simulated; no weights loaded',
        'versions': {name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers', 'jinja2')},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve().parents[2] / 'typellm').glob('*.py')},
    }
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    passed = True
    with args.output.open('w') as output:
        for model in args.models:
            try:
                row = probe(model)
            except Exception as exc:
                row = {'model': model, 'error': f'{type(exc).__name__}: {exc}'}
            passed = checks_passed(row) and passed
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
