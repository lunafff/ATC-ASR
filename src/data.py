import math
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from common import WORK

def load_audio(row):
    x, sr = sf.read(WORK / row['audio'], dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        x = resample_poly(x, 16000 // divisor, sr // divisor).astype(np.float32)
    if len(x) > 480000:
        raise ValueError('Audio exceeds 30 seconds')
    return x

def features(processor, audio, device):
    return processor.feature_extractor(audio, sampling_rate=16000, return_tensors='pt').input_features.to(device)

def labels_for(processor, text, limit, device):
    ids = processor.tokenizer(text).input_ids
    if ids[0] == processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>'):
        ids = ids[1:]
    if len(ids) > limit:
        raise ValueError(f'Target length {len(ids)} exceeds limit {limit}; do not truncate EOS')
    if ids[-1] != processor.tokenizer.eos_token_id:
        raise ValueError('Missing EOS')
    return torch.tensor([ids], device=device)
