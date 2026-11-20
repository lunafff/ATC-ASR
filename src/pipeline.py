import argparse
import gc
import json
import platform
import subprocess
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
from common import ROOT, WORK, dump, digest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    parser.add_argument('--stage', choices=['all', 'prepare', 'train', 'final', 'package'], default='all')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    WORK.mkdir(parents=True, exist_ok=True)
    if args.stage == 'package':
        package(); return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; install NVIDIA driver and CUDA-compatible PyTorch first')
    lock = WORK / 'provenance/config.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) != cfg:
        raise RuntimeError('Frozen config changed: move runtime to a separate archive before a new suite')
    dump(lock, cfg)
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'src').glob('*.py'))}
    sources = WORK / 'provenance/sources.json'
    if sources.exists() and json.loads(sources.read_text(encoding='utf-8')) != source_hashes:
        if any(p.is_file() for p in (WORK / 'runs').rglob('*')):
            raise RuntimeError('Source changed after training began: preserve runtime and start a new suite directory')
        dump(WORK / 'provenance/sources_before_download_fix.json',
             json.loads(sources.read_text(encoding='utf-8')))
        print('Updating source lock before training; preserving local downloads.', flush=True)
    dump(sources, source_hashes)
    def git(*cmd):
        return subprocess.run(['git', *cmd], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
         'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda, 'git_commit': git('rev-parse', 'HEAD'),
         'git_status': git('status', '--short'), 'timestamp': datetime.now(timezone.utc).isoformat(), 'stage': args.stage}
    dump(WORK / f'provenance/invocations/{invocation}.json', environment)
    if not (WORK / 'provenance/environment.json').exists():
        dump(WORK / 'provenance/environment.json', environment)
    frozen = subprocess.check_output([__import__('sys').executable, '-m', 'pip', 'freeze'], text=True)
    (WORK / f'provenance/invocations/{invocation}_pip.txt').write_text(frozen, encoding='utf-8')
    if not (WORK / 'provenance/pip_freeze.txt').exists():
        (WORK / 'provenance/pip_freeze.txt').write_text(frozen, encoding='utf-8')
    if args.stage in ('all', 'prepare'):
        if not (WORK / 'manifests/summary.json').exists():
            from prepare import prepare
            prepare(cfg)
    if args.stage in ('all', 'train'):
        from engine import train
        runs = [('smoke', True, cfg['adv_weights'][0], True), ('supervised', False, 0, False),
                ('kd', True, 0, False)]
        runs += [(f'kd_adv_{w}', True, w, False) for w in cfg['adv_weights']]
        for name, kd, adv, smoke in runs:
            if (WORK / 'runs' / name / 'complete.json').exists():
                continue
            dump(WORK / 'status.json', {'current': name, 'state': 'running'})
            train(cfg, name, kd, adv, smoke)
            gc.collect(); torch.cuda.empty_cache()
    if args.stage in ('all', 'final'):
        from engine import model, processor, evaluate
        from common import rows
        run_names = ['supervised', 'kd'] + [f'kd_adv_{w}' for w in cfg['adv_weights']]
        for name in run_names:
            if not (WORK / 'runs' / name / 'complete.json').exists():
                raise RuntimeError(f'{name} is incomplete; final test is gated on completing the suite')
        out = WORK / 'evaluation'; out.mkdir(exist_ok=True)
        candidates = [('teacher', WORK / 'models/teacher'), ('student_initial', WORK / 'models/student')]
        candidates += [(name, WORK / 'runs' / name / 'best') for name in run_names]
        summary = {}
        for name, path in candidates:
            metrics_path = out / f'{name}.json'
            if metrics_path.exists() and (out / f'{name}.jsonl').exists():
                saved = json.loads(metrics_path.read_text())
                if 'weight_bytes' in saved:
                    summary[name] = saved
                    continue
            m, p = model(path), processor(path, cfg)
            summary[name] = evaluate(m, p, rows('test'), cfg, out / name)
            summary[name]['weight_bytes'] = sum(f.stat().st_size for f in path.glob('*.safetensors'))
            dump(out / f'{name}.json', summary[name])
            del m, p; gc.collect(); torch.cuda.empty_cache()
        dump(out / 'summary.json', summary)
        import csv
        with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['run', 'wer', 'rtf', 'parameters', 'weight_bytes', 'n'])
            writer.writeheader()
            for name, value in summary.items():
                writer.writerow({'run': name, **{k: value[k] for k in writer.fieldnames[1:]}})
        dump(WORK / 'status.json', {'state': 'complete'})
        package()

def package():
    target = WORK / 'results_return.tar.gz'
    with tarfile.open(target, 'w:gz') as tar:
        for folder in ('provenance', 'manifests', 'evaluation', 'logs', 'runs'):
            for p in (WORK / folder).rglob('*'):
                if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.txt', '.log') and 'best' not in p.parts:
                    tar.add(p, arcname=str(p.relative_to(WORK)))
        for p in (WORK / 'status.json', WORK / 'failure.json'):
            if p.exists(): tar.add(p, arcname=p.name)
    print(f'Return bundle: {target}', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        dump(WORK / 'failure.json', {'traceback': traceback.format_exc()})
        raise
