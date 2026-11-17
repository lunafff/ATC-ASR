import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, snapshot_download
from common import WORK, dump, normalize, digest

def prepare(cfg):
    revisions = {}
    api = HfApi()
    for key in ('teacher', 'student', 'dataset'):
        repo_type = 'dataset' if key == 'dataset' else 'model'
        revision = cfg.get(f'{key}_revision') or api.repo_info(cfg[key], repo_type=repo_type).sha
        revisions[key] = revision
        snapshot_download(cfg[key], repo_type=repo_type, revision=revision,
                          local_dir=WORK / ('dataset' if key == 'dataset' else f'models/{key}'),
                          allow_patterns=['*.json', '*.txt', '*.safetensors', '*.parquet', 'README.md'])
    dump(WORK / 'provenance/revisions.json', revisions)
    sets = {'train': [], 'dev': [], 'test': []}
    seen = {}
    rejects = []
    # Process held-out split first: duplicate audio in training is removed.
    files = sorted((WORK / 'dataset').rglob('*.parquet'))
    held = [p for p in files if p.name.startswith('validation')]
    training = [p for p in files if p.name.startswith('train')]
    if not held or not training:
        raise RuntimeError('Expected official train and validation parquet files')
    for p in held + training:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                text = row.get('text', '')
                payload = row['audio']['bytes']
                audio, sr = sf.read(io.BytesIO(payload), dtype='float32', always_2d=True)
                audio = audio.mean(axis=1)
                key = hashlib.sha256(audio.tobytes() + str(sr).encode()).hexdigest()
                if key in seen:
                    if normalize(text) != seen[key]:
                        raise ValueError(f'Conflicting transcripts for duplicate audio {key}')
                    continue
                seen[key] = normalize(text)
                duration = len(audio) / sr
                if not normalize(text) or not 0 < duration <= 30:
                    rejects.append({'id': key, 'duration': duration, 'reason': 'empty text or duration outside (0,30]'})
                    continue
                split = 'test' if p in held else ('dev' if int(key[:8], 16) / 2**32 < cfg['dev_fraction'] else 'train')
                path = WORK / 'audio' / f'{key}.wav'
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio, sr, subtype='FLOAT')
                sets[split].append({'id': key, 'audio': str(path.relative_to(WORK)), 'text': text,
                                    'duration': duration, 'source': p.name})
    for split, records in sets.items():
        if not records:
            raise ValueError(f'Empty split {split}')
        path = WORK / 'manifests' / f'{split}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    dump(WORK / 'manifests/summary.json', {'counts': {k: len(v) for k, v in sets.items()},
          'sha256': {k: digest(WORK / 'manifests' / f'{k}.jsonl') for k in sets}, 'excluded': rejects,
          'limitation': 'Utterance-level split; speaker/session independence and teacher pretraining overlap unknown'})
