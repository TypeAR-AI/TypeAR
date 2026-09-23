"""Verify published answers, paired transitions, aggregates and file hashes.

Optional --benchmark additionally checks labels and gold answers against the
pinned upstream checkout. No model calls or third-party packages are needed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
PIN = 'f8ce71361165846101d02ebc83ad44e47ae44fc3'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(benchmark=None):
    checksum_file = ROOT / 'checksums.json'
    for name, digest in read(checksum_file).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    gold = {}
    if benchmark:
        assert subprocess.check_output(['git', '-C', str(benchmark), 'rev-parse', 'HEAD'], text=True).strip() == PIN
        gold = {t['id']: t for tier in ('original', 'easy', 'hard')
                for t in rows(benchmark / f'datasets/public/{tier}.jsonl')}
    exported = {}
    for name, count, correct in [('no_thinking', 231, 195), ('thinking', 231, 228),
                                 ('interrupted_thinking2048', 5, None)]:
        records = rows(ROOT / name / 'answers.jsonl')
        assert len(records) == len({r['task_id'] for r in records}) == count
        if correct is not None:
            assert sum(r['correct'] for r in records) == correct
        brier = []
        for record in records:
            probabilities = record['probabilities']
            assert set(probabilities) == set(record['labels'])
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values())
            assert abs(sum(probabilities.values()) - 1) <= .001
            predicted = min(probabilities, key=lambda key: (-probabilities[key], key))
            assert predicted == record['predicted']
            assert (predicted == str(record['expected'])) == record['correct']
            answer = record['answer']
            if record['question_type'] == 'noul':
                assert answer['noul'] == probabilities['yes']
            else:
                assert answer['probabilities'] == probabilities
                if record['question_type'] == 'choice':
                    assert answer['choice'] in probabilities
                else:
                    assert math.isclose(answer['score'], sum(float(k)*p for k,p in probabilities.items()), abs_tol=1e-12)
            assert sum(c['completion_tokens'] for c in record['generation_calls'] if c['phase']=='thinking') == record['thinking_tokens']
            assert record['answer_tokens'] == 1
            if gold:
                task = gold[record['task_id']]
                assert record['expected'] == task['expected'] and record['labels'] == task['labels']
                assert record['question_type'] == task['question']['type']
            brier.append(sum((p-int(k == str(record['expected'])))**2 for k,p in probabilities.items()))
        if correct is not None:
            summary = read(ROOT / name / 'summary.json')
            assert summary['n_correct'] == correct and summary['n_attempted'] == count
            assert math.isclose(summary['accuracy'], correct/count)
            assert math.isclose(summary['brier_mean'], sum(brier)/count, abs_tol=1e-12)
            for tier, size in [('original',72), ('easy',48), ('hard',111)]:
                subset = [r for r in records if r['tier']==tier]
                assert len(subset) == size
                assert sum(r['correct'] for r in subset) == summary['tiers'][tier]['n_correct']
        exported[name] = {r['task_id']:r for r in records}
    comparisons = rows(ROOT / 'comparison.jsonl')
    assert len(comparisons) == len({r['task_id'] for r in comparisons}) == 231
    transitions = Counter()
    for r in comparisons:
        a,b = (exported[run][r['task_id']] for run in ('no_thinking','thinking'))
        assert r['no_thinking_answer'] == a['predicted'] and r['thinking_answer'] == b['predicted']
        expected = ('both_correct' if a['correct'] and b['correct'] else
                    'both_wrong' if not a['correct'] and not b['correct'] else
                    'fixed_by_thinking' if b['correct'] else 'regressed_with_thinking')
        assert r['transition'] == expected
        transitions[expected] += 1
    assert transitions == {'both_correct':194, 'both_wrong':2, 'fixed_by_thinking':34, 'regressed_with_thinking':1}
    assert rows(ROOT / 'mismatches.jsonl') == [r for r in comparisons if r['transition'] != 'both_correct']
    assert len(rows(ROOT / 'infrastructure_failures.jsonl')) == 6
    assert sum(r['thinking_tokens'] for r in exported['thinking'].values()) == 212255
    assert sum(r['transport_network_retries'] for r in exported['thinking'].values()) == 1
    print('Verified 462 complete-run answers, 5 excluded partial answers, 231 paired tasks, 6 failed attempts and all checksums.')
    if benchmark:
        print('All exported gold answers and label sets match the pinned upstream public tasks.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path)
    verify(parser.parse_args().benchmark)
