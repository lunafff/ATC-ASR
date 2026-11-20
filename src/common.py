import hashlib
import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get('ATC_WORK_DIR', str(ROOT / 'runtime'))).resolve()

def evaluation_path(output, suffix):
    output = Path(output)
    return output.parent / (output.name + suffix)

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def rows(split):
    return [json.loads(x) for x in (WORK / 'manifests' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]

def normalize(text):
    return ' '.join(re.sub(r"[^a-z0-9\s]", ' ', text.lower().replace("'", '')).split())

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def alignment(student_vocab, teacher_vocab):
    # A permutation into teacher coordinates; never assume numeric IDs coincide.
    ordered = sorted(student_vocab, key=student_vocab.get)
    # The published teacher tokenizer renamed Whisper's no-speech marker.
    aliases = {'<|nocaptions|>': '<|nospeech|>'}
    target = [token if token in teacher_vocab else aliases.get(token, token) for token in ordered]
    missing = set(target) - teacher_vocab.keys()
    if missing:
        raise ValueError(f'Teacher lacks student tokens: {missing}')
    mapped = [teacher_vocab[token] for token in target]
    if len(set(mapped)) != len(mapped):
        raise ValueError('Token mapping is not one-to-one')
    return mapped
