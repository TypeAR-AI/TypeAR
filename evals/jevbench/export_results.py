"""Export allowlisted JevBench answers from private/local run artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

PIN = 'f8ce71361165846101d02ebc83ad44e47ae44fc3'
RUNS = {
    'no_thinking': 'jevbench_qwen38_no_thinking_20260923',
    'thinking': 'jevbench_qwen38_thinking_final_20260923',
    'interrupted_thinking2048': 'jevbench_qwen38_thinking2048_20260923',
}


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n' for value in values))


def export(args):
    tasks = {}
    for tier in ('original', 'easy', 'hard'):
        for line, task in enumerate(rows(args.benchmark / f'datasets/public/{tier}.jsonl'), 1):
            tasks[task['id']] = (tier, task, line)
    assert len(tasks) == 231
    all_results = {}
    for name, directory in RUNS.items():
        src = args.runs / directory
        manifest = read(src / 'manifest.json')
        assert manifest['benchmark_revision'] == PIN
        output = []
        for record in rows(src / 'results.jsonl'):
            tier, task, line = tasks[record['task_id']]
            raw_file = src / 'raw' / (hashlib.sha256(task['id'].encode()).hexdigest() + '.json')
            blob = raw_file.read_bytes()
            assert hashlib.sha256(blob).hexdigest() == record['raw_sha256']
            raw = json.loads(blob)['response']
            answer = raw['answers']['decision']
            answer = {key: value for key, value in answer.items()
                      if key in ('type', 'noul', 'choice', 'score', 'probabilities', 'confidence')}
            calls = []
            retries = 0
            for event in raw.get('transport', []):
                if event.get('network_error'):
                    retries += 1
                    continue
                if event['path'] != '/generate':
                    continue
                sample = event['request']['sampling_params']
                phase = ('thinking' if '</think>' in sample.get('stop', []) else
                         'prefill' if sample['max_new_tokens'] == 0 else 'answer')
                responses = event['response']
                for response in responses if isinstance(responses, list) else [responses]:
                    meta = response['meta_info']
                    calls.append({'phase': phase, 'completion_tokens': meta['completion_tokens'],
                                  'finish_type': meta.get('finish_reason', {}).get('type')})
            output.append({
                'task_id': task['id'], 'tier': tier, 'family': task['family'],
                'question_type': task['question']['type'], 'labels': task['labels'],
                'expected': task['expected'], 'predicted': record['predicted'],
                'correct': record['correct'], 'answer': answer,
                'probabilities': record['probs_as_returned'], 'strict_valid': record['strict_valid'],
                'renormalized': record['renormalized'], 'latency_s': record['latency_s'],
                'thinking_tokens': sum(c['completion_tokens'] for c in calls if c['phase'] == 'thinking'),
                'answer_tokens': sum(c['completion_tokens'] for c in calls if c['phase'] == 'answer'),
                'generation_calls': calls, 'transport_network_retries': retries,
                'source_raw_sha256': record['raw_sha256'],
                'task_source': f'https://github.com/fstandhartinger/jevbench/blob/{PIN}/datasets/public/{tier}.jsonl#L{line}',
            })
        all_results[name] = output
        write_rows(args.output / name / 'answers.jsonl', output)
        public_manifest = {key: manifest[key] for key in (
            'benchmark_revision', 'dataset_hash', 'model', 'thinking', 'thinking_budget',
            'thinking_sampling', 'permutations', 'mode', 'temperature', 'concurrency',
            'candidate_order', 'probability_source', 'dependencies', 'source_sha256') if key in manifest}
        public_manifest.update(n_exported=len(output), complete=len(output) == 231,
                               included_in_headline=name != 'interrupted_thinking2048',
                               source_raw_artifacts_public=False,
                               inference_source_hashes={k: v for k, v in manifest['source_sha256'].items()
                                                        if k.startswith(('TypeLLM/typellm/', 'jev-compat/jev_compat/'))})
        public_manifest.pop('source_sha256')
        write(args.output / name / 'manifest.json', public_manifest)
        if (src / 'summary.json').exists():
            write(args.output / name / 'summary.json', read(src / 'summary.json'))
        if (src / 'audit.json').exists():
            write(args.output / name / 'local_audit.json', read(src / 'audit.json'))

    before = {row['task_id']: row for row in all_results['no_thinking']}
    comparisons = []
    for row in all_results['thinking']:
        old = before[row['task_id']]
        transition = ('both_correct' if old['correct'] and row['correct'] else
                      'both_wrong' if not old['correct'] and not row['correct'] else
                      'fixed_by_thinking' if row['correct'] else 'regressed_with_thinking')
        comparisons.append({'task_id': row['task_id'], 'tier': row['tier'],
                            'question_type': row['question_type'], 'expected': row['expected'],
                            'no_thinking_answer': old['predicted'], 'thinking_answer': row['predicted'],
                            'no_thinking_correct': old['correct'], 'thinking_correct': row['correct'],
                            'transition': transition, 'task_source': row['task_source']})
    write_rows(args.output / 'comparison.jsonl', comparisons)
    failed = []
    for stage, directory in enumerate(('jevbench_qwen38_thinking_20260923',
                                       'jevbench_qwen38_thinking_recovered_20260923'), 1):
        for record in rows(args.runs / directory / 'results.jsonl'):
            if not record['ok']:
                failed.append({'stage': stage, 'task_id': record['task_id'],
                               'failure_class': 'IAP_SSH_broken_pipe', 'latency_s': record['latency_s'],
                               'model_answer_received': False, 'retried_in_final_run': True,
                               'source_raw_sha256': record['raw_sha256']})
    write_rows(args.output / 'infrastructure_failures.jsonl', failed)
    lines = ['# Per-task answer comparison', '',
             '| Task | Tier | Gold | No thinking | Thinking | Outcome |', '|---|---|---|---|---|---|']
    for r in comparisons:
        lines.append(f"| [{r['task_id']}]({r['task_source']}) | {r['tier']} | `{r['expected']}` | `{r['no_thinking_answer']}` | `{r['thinking_answer']}` | {r['transition']} |")
    (args.output / 'ANSWERS.md').write_text('\n'.join(lines) + '\n')
    mismatches = [r for r in comparisons if r['transition'] != 'both_correct']
    write_rows(args.output / 'mismatches.jsonl', mismatches)
    print(f'Exported {sum(map(len, all_results.values()))} answers; {len(failed)} infrastructure failures; {len(comparisons)} paired tasks.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    export(parser.parse_args())
