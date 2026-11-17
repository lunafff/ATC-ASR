import json
import time
import torch
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import process_words
from common import WORK, dump, rows, normalize, seed_all, alignment
from data import load_audio, features, labels_for

def processor(path, cfg):
    p = WhisperProcessor.from_pretrained(path, language=cfg['language'], task='transcribe')
    p.tokenizer.set_prefix_tokens(language=cfg['language'], task='transcribe', predict_timestamps=False)
    return p

def model(path):
    m = WhisperForConditionalGeneration.from_pretrained(path, attn_implementation='eager').cuda()
    m.generation_config.forced_decoder_ids = None
    m.config.forced_decoder_ids = None
    return m

def evaluate(m, p, records, cfg, output):
    m.eval()
    predictions = []
    errors = words = 0
    total_time = duration = 0.0
    # Unmeasured warmup prevents first-kernel initialization dominating latency.
    if records:
        with torch.inference_mode():
            m.generate(features(p, load_audio(records[0]), 'cuda'), language=cfg['language'],
                       task='transcribe', num_beams=1, do_sample=False, max_new_tokens=cfg['max_tokens'])
    for r in records:
        x = load_audio(r)
        torch.cuda.synchronize()
        start = time.perf_counter()
        f = features(p, x, 'cuda')
        with torch.inference_mode():
            ids = m.generate(f, language=cfg['language'], task='transcribe', num_beams=1,
                             do_sample=False, max_new_tokens=cfg['max_tokens'])
            # Pinned Transformers 4.46.3 short-form output retains the prefix
            # but strips trailing pad/EOS. Full length thus flags capped output.
            hit_cap = ids.shape[1] >= cfg['max_tokens'] + len(p.tokenizer.prefix_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        hyp = p.batch_decode(ids, skip_special_tokens=True)[0]
        ref, cleaned = normalize(r['text']), normalize(hyp)
        result = process_words(ref, cleaned)
        errors += result.substitutions + result.deletions + result.insertions
        words += len(ref.split())
        total_time += elapsed
        duration += r['duration']
        predictions.append({'id': r['id'], 'reference': r['text'], 'prediction': hyp,
                            'seconds': elapsed, 'hit_length_cap': hit_cap,
                            'adjacent_repeat_count': sum(a == b for a,b in zip(cleaned.split(), cleaned.split()[1:]))})
    metrics = {'wer': errors / words, 'word_errors': errors, 'reference_words': words,
               'rtf': total_time / duration, 'measured_seconds': total_time,
               'parameters': sum(v.numel() for v in m.parameters()), 'n': len(records),
               'latency_scope': 'feature extraction + greedy generation, batch=1, float32, no VAD; excludes file I/O',
               'entity_error': None, 'length_cap_count': sum(r['hit_length_cap'] for r in predictions)}
    dump(output.with_suffix('.json'), metrics)
    output.with_suffix('.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in predictions), encoding='utf-8')
    return metrics

def train(cfg, name, kd, adv, smoke=False):
    seed_all(cfg['seed'])
    out = WORK / 'runs' / name
    out.mkdir(parents=True, exist_ok=True)
    sp = processor(WORK / 'models/student', cfg)
    s = model(WORK / 'models/student')
    s.config.use_cache = False
    s.gradient_checkpointing_enable()
    t = tp = mapping = None
    if kd:
        tp = processor(WORK / 'models/teacher', cfg)
        t = model(WORK / 'models/teacher').eval().requires_grad_(False)
        mapping = torch.tensor(alignment(sp.tokenizer.get_vocab(), tp.tokenizer.get_vocab()), device='cuda')
        if len(mapping) != s.config.vocab_size:
            raise ValueError('Student tokenizer and output vocabulary sizes differ')
    # D distinguishes teacher token distributions (real) from student (fake).
    # Fixed shared random projection avoids a V x hidden trainable discriminator.
    projection = torch.randn(s.config.vocab_size, 64, device='cuda') / 8
    d = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).cuda()
    opt = torch.optim.AdamW(s.parameters(), lr=cfg['lr'])
    dop = torch.optim.AdamW(d.parameters(), lr=cfg['adv_lr'])
    start_epoch, best, step = 0, float('inf'), 0
    latest = out / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, weights_only=False)
        if state['config'] != cfg:
            raise ValueError('Config changed; choose a new suite directory')
        s.load_state_dict(state['student']); opt.load_state_dict(state['optimizer'])
        d.load_state_dict(state['discriminator']); dop.load_state_dict(state['d_optimizer'])
        projection = state['projection'].cuda()
        start_epoch, best, step = state['epoch'], state['best'], state['step']
        torch.set_rng_state(state['rng_cpu']); torch.cuda.set_rng_state_all(state['rng_cuda'])
    data = rows('train')
    epochs = 1 if smoke else cfg['epochs']
    for epoch in range(start_epoch, epochs):
        import random
        order = list(range(len(data)))
        random.Random(cfg['seed'] + epoch).shuffle(order)
        if smoke:
            order = order[:cfg['smoke_steps']]
        s.train(); opt.zero_grad(set_to_none=True)
        for i, idx in enumerate(order):
            r = data[idx]
            audio = load_audio(r)
            labels = labels_for(sp, r['text'], cfg['max_tokens'], 'cuda')
            result = s(input_features=features(sp, audio, 'cuda'), labels=labels)
            ce = result.loss
            kl = ce.new_zeros(()); generator = ce.new_zeros(()); dl = ce.new_zeros(())
            if kd:
                with torch.no_grad():
                    tlabels = mapping[labels]
                    target = t(input_features=features(tp, audio, 'cuda'), labels=tlabels).logits
                    target = target.index_select(-1, mapping)
                    teacher_prob = F.softmax(target.float() / cfg['temperature'], dim=-1)
                student_log = F.log_softmax(result.logits.float() / cfg['temperature'], dim=-1)
                kl = F.kl_div(student_log, teacher_prob, reduction='none').sum(-1).mean() * cfg['temperature']**2
                if adv:
                    real = teacher_prob @ projection
                    fake = student_log.exp() @ projection
                    d.requires_grad_(True); dop.zero_grad(set_to_none=True)
                    dl = F.binary_cross_entropy_with_logits(d(real.detach()), torch.ones_like(d(real.detach())))
                    dl = dl + F.binary_cross_entropy_with_logits(d(fake.detach()), torch.zeros_like(d(fake.detach())))
                    dl.backward(); dop.step(); d.requires_grad_(False)
                    generator = F.binary_cross_entropy_with_logits(d(fake), torch.ones_like(d(fake)))
            loss = (1-cfg['kd_weight'])*ce + cfg['kd_weight']*kl if kd else ce
            loss = loss + adv * generator
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            group_size = min(cfg['grad_accum'], len(order) - (i // cfg['grad_accum']) * cfg['grad_accum'])
            (loss / group_size).backward()
            if (i+1) % cfg['grad_accum'] == 0 or i+1 == len(order):
                norm = torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0, error_if_nonfinite=True)
                if smoke and norm.item() == 0:
                    raise RuntimeError('Smoke: no student gradient')
                if t is not None and any(p.grad is not None for p in t.parameters()):
                    raise RuntimeError('Frozen teacher unexpectedly received gradients')
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
            record = {'epoch': epoch+1, 'sample': i+1, 'step': step, 'ce': ce.item(), 'kd': kl.item(),
                      'adv': generator.item(), 'd': dl.item(), 'loss': loss.item()}
            with (out / 'train.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record)+'\n')
            if i % 20 == 0:
                print(name, record, flush=True)
        metrics = evaluate(s, sp, rows('dev')[:2] if smoke else rows('dev'), cfg, out / f'dev_epoch_{epoch+1}')
        if metrics['wer'] < best:
            best = metrics['wer']
            s.save_pretrained(out / 'best'); sp.save_pretrained(out / 'best')
        state = {'student': s.state_dict(), 'optimizer': opt.state_dict(), 'discriminator': d.state_dict(),
                 'd_optimizer': dop.state_dict(), 'projection': projection, 'epoch': epoch+1, 'best': best,
                 'step': step, 'config': cfg, 'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()}
        torch.save(state, out / 'latest.tmp'); (out / 'latest.tmp').replace(latest)
    dump(out / 'complete.json', {'best_dev_wer': best, 'epochs': epochs, 'smoke': smoke})
