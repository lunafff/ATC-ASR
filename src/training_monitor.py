"""Dependency-free progress messages, resume log cleanup and SVG curves."""
import json
import math
import shutil
import time
from datetime import datetime
from html import escape
from pathlib import Path


def monitor_update_allowed(root, previous, current):
    path = Path(root) / 'configs/monitor_compatibility.json'
    if not path.exists():
        return False
    known = json.loads(path.read_text(encoding='utf-8'))
    return current == known['current'] and previous in known['previous']


def log(message):
    print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)


def progress(label, done, total, started, details=''):
    elapsed = time.perf_counter() - started
    speed = done / max(elapsed, 1e-6)
    eta = (total - done) / max(speed, 1e-6)
    bar = '#' * int(24 * done / max(total, 1))
    log(f'{label} [{bar:<24}] {done}/{total} ({100*done/max(total,1):.1f}%) '
        f'{speed:.2f} samples/s ETA {eta/60:.1f} min {details}')


def records(path):
    result = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # An interrupted final write is not a training observation.
            result[(item['epoch'], item['sample'])] = item
    return [result[k] for k in sorted(result)]


def rewind_log(out, completed_epochs):
    path = Path(out) / 'train.jsonl'
    if not path.exists():
        return
    rows = records(path)
    kept = [r for r in rows if r['epoch'] <= completed_epochs]
    archive = Path(out) / 'interrupted_logs' / (datetime.now().strftime('%Y%m%dT%H%M%S%f') + '.jsonl')
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, archive)
    temporary = path.with_suffix('.rewind.tmp')
    temporary.write_text(''.join(json.dumps(r) + '\n' for r in kept), encoding='utf-8')
    temporary.replace(path)
    log(f'Resume log: kept {len(kept)} samples from {completed_epochs} completed epochs; previous log: {archive}')


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def draw_curves(out):
    out = Path(out)
    rows = records(out / 'train.jsonl')
    if not rows:
        return
    dev = []
    for path in out.glob('dev_epoch_*.json'):
        epoch = int(path.stem.split('_')[-1])
        item = json.loads(path.read_text(encoding='utf-8'))
        dev.append((epoch, item))
    dev.sort()
    panels = []
    for label, key in [('Total loss', 'loss'), ('CE loss', 'ce'), ('KD loss', 'kd')]:
        panels.append((label + ' (100-sample moving mean)', [(i+1, r[key]) for i, r in enumerate(rows)], True))
    panels += [('Adversarial G / discriminator D (G blue, D orange)',
                [(i+1, r['adv']) for i, r in enumerate(rows)], True),
               ('Dev WER (%)', [(e, r['wer']*100) for e, r in dev], False),
               ('Dev RTF', [(e, r['rtf']) for e, r in dev], False)]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="890" viewBox="0 0 1100 890">',
           '<rect width="1100" height="890" fill="white"/>',
           f'<text x="30" y="28" font-size="19">{escape(out.parent.parent.name + " / " + out.name)} — {escape(datetime.now().isoformat(timespec="seconds"))}</text>']

    def smooth(points):
        total = 0.0
        values = []
        for i, (x, value) in enumerate(points):
            total += value
            if i >= 100:
                total -= points[i-100][1]
            values.append((x, total / min(i+1, 100)))
        return values

    for index, (title, points, moving) in enumerate(panels):
        left = 65 + (index % 2)*540; top = 80 + (index // 2)*275
        width, height = 440, 185
        series = [smooth(points) if moving else points]
        if index == 3:
            series.append(smooth([(i+1, r['d']) for i, r in enumerate(rows)]))
        all_points = [(x, y) for s in series for x, y in s if math.isfinite(y)]
        svg.append(f'<text x="{left}" y="{top-18}" font-size="13">{escape(title)}</text>')
        svg.append(f'<path d="M{left} {top}v{height}h{width}" fill="none" stroke="#888"/>')
        if not all_points:
            continue
        xmax = max(x for x,y in all_points); ymax = max(max(y for x,y in all_points)*1.05, 1e-6)
        for tick in range(5):
            y = top + height - tick*height/4
            svg.append(f'<text x="{left-5}" y="{y+4}" text-anchor="end" font-size="10">{ymax*tick/4:.3g}</text>')
        for tick in range(5):
            x = left + tick*width/4
            svg.append(f'<text x="{x}" y="{top+height+15}" text-anchor="middle" font-size="10">{xmax*tick/4:.0f}</text>')
        svg.append(f'<text x="{left+width/2}" y="{top+height+34}" text-anchor="middle" font-size="11">{"Logged training sample" if moving else "Epoch"}</text>')
        for s, color in zip(series, ['#1261a0', '#df8011']):
            stride = max(1, len(s)//1200)
            sampled = s[::stride]
            if s and sampled[-1] != s[-1]:
                sampled.append(s[-1])
            coords = ' '.join(f'{left+x/max(xmax,1)*width:.1f},{top+height-y/ymax*height:.1f}'
                              for x,y in sampled if math.isfinite(y))
            svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.6"/>')
            if len(sampled) == 1:
                x,y = sampled[0]
                svg.append(f'<circle cx="{left+x/max(xmax,1)*width}" cy="{top+height-y/ymax*height}" r="3" fill="{color}"/>')
    svg.append('</svg>')
    atomic_text(out / 'training_curves.svg', '\n'.join(svg))
    atomic_text(out / 'training_curves.html', '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Training curves</title><img style="max-width:100%" src="training_curves.svg">')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, required=True)
    args = parser.parse_args()
    for folder in (args.work_dir / 'runs').glob('*'):
        if folder.is_dir():
            draw_curves(folder)
    log(f'Curves updated: {args.work_dir / "runs"}')
