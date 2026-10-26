"""Prepare hash-bound matched prompts and run the frozen 4B intervention."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from gleipnir.adversarial.contract import join_prompt, split_prompt

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
DATA = ROOT / 'data/adversarial_peer_deception_prompt'
OUT = ROOT / 'results/adversarial_peer_deception_prompt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def replace_head(prompt, head):
    trajectory = split_prompt(prompt).trajectory
    result = join_prompt(head, trajectory)
    assert split_prompt(result).trajectory == trajectory
    return result


def prepare():
    config = json.loads((EXP / 'config.json').read_text())
    for path, expected in config['sha256'].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f'Input drift: {path}')
    heads = {'peer': (EXP / 'peer.txt').read_text(),
             'teacher': (ROOT / 'experiments/tool_trajectory_monitoring/prompts/teacher.txt').read_text()}
    sources = {'main': ROOT / 'data/adversarial_lab_liars/main_student.jsonl',
               'calibration': ROOT / 'data/adversarial_is_it_4b/calibration_student.jsonl'}
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {'inputs': config['sha256'], 'files': {}}
    for arm, head in heads.items():
        for name, source in sources.items():
            if arm == 'teacher' and name == 'calibration':
                continue
            rows = []
            for row in read(source):
                if row['metadata'].get('over_limit'):
                    continue
                rows.append({**row, 'prompt': replace_head(row['prompt'], head)})
            path = DATA / f'{arm}_{name}.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            manifest['files'][path.name] = {'sha256': sha(path), 'rows': len(rows)}
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['files'], indent=2))


def score():
    from gleipnir.adversarial.scorer import Scorer, TooLong
    config = json.loads((EXP / 'config.json').read_text())
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, spec in manifest['files'].items():
        assert sha(DATA / name) == spec['sha256'], name
    assert manifest['inputs'] == config['sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    canary_path = ROOT / 'data/adversarial_lab_liars/canaries.jsonl'
    assert sha(canary_path) == config['sha256'][str(canary_path.relative_to(ROOT))]
    started = time.monotonic()
    scorer = Scorer(config['monitor'])
    identity = {**scorer.identity(), 'config_sha256': sha(EXP / 'config.json'),
                'manifest_sha256': sha(DATA / 'manifest.json'),
                'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True).stdout.strip() or (ROOT / '.peer_prompt_synced_commit').read_text().strip(),
                'runner_sha256': sha(EXP / 'run.py')}
    identity_path = OUT / 'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Resume identity drift'
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')

    def canaries(phase):
        receipts = []
        for row in read(canary_path):
            got = scorer.score(row['prompts']['student'])
            expected = row['expected']['4b']
            if isinstance(expected, dict):
                expected = expected['score']
            assert math.isfinite(got['score']) and abs(got['score'] - expected) <= 0.05
            receipts.append({'id': row['id'], 'expected': expected, **got})
        if phase == 'before':
            base = scorer.score(read(canary_path)[0]['prompts']['student'], adapter=False)
            assert abs(base['score'] - receipts[0]['score']) > 1e-6, 'No adapter effect'
            receipts.append({'base_control': base})
        (OUT / f'canaries_{phase}.json').write_text(json.dumps(receipts, indent=2)+'\n')
        print(f'Canaries {phase}: passed', flush=True)

    canaries('before')
    for arm, name in [('peer', 'main'), ('peer', 'calibration'), ('teacher', 'main')]:
        source = DATA / f'{arm}_{name}.jsonl'
        output = OUT / f'{arm}_{name}.jsonl'
        done = {r['id'] for r in read(output)}
        rows = read(source)
        total_tokens = 0
        arm_start = time.monotonic()
        with output.open('a') as handle:
            for n, row in enumerate(rows, 1):
                if row['id'] in done:
                    continue
                if time.monotonic() - started > config['max_seconds']:
                    canaries('after')
                    print('STOP: wall-clock cap', flush=True)
                    return
                record = {'id': row['id'], 'metadata': row['metadata'], 'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()}
                try:
                    values = scorer.score(row['prompt'])
                    assert all(math.isfinite(values[k]) for k in ['score', 'logit_0', 'logit_1'])
                    record.update(values, status='ok')
                    total_tokens += values['tokens']
                except TooLong as error:
                    record.update(status='too_long', error=str(error))
                handle.write(json.dumps(record) + '\n')
                handle.flush()
                if n % 25 == 0 or n == len(rows):
                    print(f'{arm}/{name} {n}/{len(rows)}; {total_tokens / max(time.monotonic()-arm_start, .001):.0f} prompt tokens/s', flush=True)
    canaries('after')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'score'])
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else score()
