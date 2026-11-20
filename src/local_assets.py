"""Reuse exact pinned model/data files before making any Hub request."""
import hashlib
import json
import os
import shutil
from pathlib import Path


def matches(path, entry):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    hasher = hashlib.sha256() if entry.get('sha256') else hashlib.sha1()
    if not entry.get('sha256'):
        hasher.update(f"blob {entry['size']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest() == (entry.get('sha256') or entry['blob'])


def ensure_assets(cfg, key, root, work, download):
    root, work = Path(root), Path(work)
    inventories = json.loads((root / 'configs/assets.json').read_text(encoding='utf-8'))
    revision = cfg[f'{key}_revision']
    inventory = next((v for v in inventories.values()
                      if v['repo'] == cfg[key] and v['revision'] == revision), None)
    if inventory is None:
        raise ValueError(f'No pinned file inventory for {key}: {cfg[key]} @ {revision}')
    relative = Path('dataset' if key == 'dataset' else f'models/{key}')
    destination = work / relative
    suites = [root / 'runtime', root / 'runtime/base', root / 'runtime/small']
    for archive in sorted(root.glob('runtime_old*')):
        if archive.is_dir():
            suites.extend([archive, archive / 'base', archive / 'small'])
    # Reuse only individual files matching the pinned upstream size and hash.
    missing = []
    for entry in inventory['files']:
        target = destination / entry['path']
        if matches(target, entry):
            continue
        found = False
        for suite in suites:
            source = suite / relative / entry['path']
            if source.resolve() == target.resolve() or not matches(source, entry):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.reuse.tmp')
            if temporary.exists():
                temporary.unlink()
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
            temporary.replace(target)
            print(f'Reusing local {key}: {source}', flush=True)
            found = True
            break
        if not found:
            missing.append(entry['path'])
    if missing:
        print(f'Downloading {key}: {len(missing)} missing/invalid files', flush=True)
        download(cfg[key], repo_type='dataset' if key == 'dataset' else 'model',
                 revision=revision, local_dir=destination, allow_patterns=missing)
        bad = [entry['path'] for entry in inventory['files']
               if not matches(destination / entry['path'], entry)]
        if bad:
            raise RuntimeError(f'Incomplete or incorrect {key} files after download: {bad}')
    else:
        print(f'Local {key} complete, pinned hashes verified; skipping Hub download.', flush=True)
    return revision
