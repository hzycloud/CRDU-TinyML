"""Source-disjoint Speech Commands data and the selection-before-test gate.

This cache is deliberately independent of the historical processed speech data.
Silence crops are dependent within a role: each role has one whole noise source.
"""
import hashlib
import json
import random
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .audio import mfcc_features, pad_or_truncate
from .speech_commands import TARGET_WORDS, WORD_TO_LABEL, SILENCE_LABEL, UNKNOWN_LABEL


ROLES = ('train', 'validation', 'quantization', 'temperature', 'selection', 'test')
CLASS_NAMES = list(TARGET_WORDS) + ['silence', 'unknown']
DEFAULT_CAPS = dict(zip(ROLES, (2000, 200, 180, 180, 180, 400)))
SPLIT_SEED = 20261027
BUDGETS = (1., .95, .9, .85, .8)
PROTOCOL = 'round2_speech_source_disjoint_v1'


def sha256_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    # Windows readers/virus scanners can briefly hold the destination open.
    # Keep the old complete JSON visible while retrying the atomic replacement.
    for attempt in range(9):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(.025 * (attempt + 1))


def wav_metadata(path):
    with wave.open(str(path), 'rb') as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f'{path}: expected mono PCM16')
        rate, frames = stream.getframerate(), stream.getnframes()
    if rate != 16000:
        raise ValueError(f'{path}: actual sample rate {rate}; expected 16000')
    return {'sample_rate': rate, 'source_samples': frames, 'sha256': sha256_file(path)}


def speaker_id(relative):
    name = Path(relative).stem
    if '_nohash_' not in name:
        raise ValueError(f'missing Speech Commands speaker ID: {relative}')
    return name.split('_nohash_', 1)[0]


def _overlap_summary(rows):
    starts = sorted(r['crop_start_samples'] for r in rows)
    left, pairs = 0, 0
    for right, start in enumerate(starts):
        while left < right and starts[left] + 16000 <= start:
            left += 1
        pairs += right - left
    return {'crop_count': len(starts), 'unique_crop_starts': len(set(starts)),
            'overlapping_crop_pairs': pairs, 'clip_samples': 16000,
            'within_role_overlap_allowed': True,
            'disclosure': 'One background recording per role; crops can overlap and are not independent sources.'}


def audit_manifest(manifest):
    """Reject cross-role speakers, source paths, recording IDs or raw content."""
    seen = {key: {} for key in ('speaker', 'background', 'path', 'hash')}
    counts = {}
    for role in ROLES:
        rows = manifest['roles'][role]
        local = {key: set() for key in seen}
        for row in rows:
            if Path(row['path']).is_absolute() or '..' in Path(row['path']).parts:
                raise ValueError('manifest paths must be dataset-relative')
            local['path'].add(row['path'])
            local['hash'].add(row['sha256'])
            if row['kind'] == 'word':
                local['speaker'].add(row['speaker_id'])
            else:
                local['background'].add(row['background_id'])
        if len(local['background']) != 1:
            raise ValueError(f'{role}: requires exactly one background source')
        for key, values in local.items():
            for value in values:
                if value in seen[key]:
                    raise ValueError(f'cross-role {key} overlap: {role}, {seen[key][value]}, {value}')
                seen[key][value] = role
        counts[role] = {key: len(values) for key, values in local.items()}
    return {'passed': True, 'cross_role_speaker_overlap': 0,
            'cross_role_background_overlap': 0, 'cross_role_path_overlap': 0,
            'cross_role_source_hash_overlap': 0, 'unique_sources_by_role': counts}


def build_manifest(root, caps=None, split_seed=SPLIT_SEED):
    root = Path(root)
    caps = dict(DEFAULT_CAPS if caps is None else caps)
    if set(caps) != set(ROLES) or any(int(n) < 1 for n in caps.values()):
        raise ValueError('positive class caps are required for all six roles')
    official = {}
    for role, filename in (('validation', 'validation_list.txt'), ('test', 'testing_list.txt')):
        official[role] = set((root / filename).read_text(encoding='utf-8').replace('\\', '/').splitlines()) - {''}
    if official['validation'] & official['test']:
        raise ValueError('official validation/test file overlap')
    paths = sorted(p.relative_to(root).as_posix() for directory in root.iterdir()
                   if directory.is_dir() and directory.name != '_background_noise_'
                   for p in directory.glob('*.wav'))
    if not (official['validation'] | official['test']) <= set(paths):
        raise ValueError('missing official held-out word files')
    held_speakers = {role: {speaker_id(p) for p in files} for role, files in official.items()}
    if held_speakers['validation'] & held_speakers['test']:
        raise ValueError('official held-out speaker overlap')
    remaining = [p for p in paths if p not in official['validation'] | official['test']]
    train_speakers = sorted({speaker_id(p) for p in remaining})
    if set(train_speakers) & (held_speakers['validation'] | held_speakers['test']):
        raise ValueError('official training and held-out speaker overlap')
    random.Random(split_seed).shuffle(train_speakers)
    small = len(train_speakers) // 10
    if small < 1:
        raise ValueError('too few official train speakers for four groups')
    group_counts = (len(train_speakers) - 3 * small, small, small, small)
    groups, offset = {}, 0
    for role, count in zip(('train', 'quantization', 'temperature', 'selection'), group_counts):
        groups[role] = sorted(train_speakers[offset:offset + count])
        offset += count
    assignments = {speaker: role for role, speakers in groups.items() for speaker in speakers}
    available = {role: [[] for _ in CLASS_NAMES] for role in ROLES}
    for relative in paths:
        role = ('validation' if relative in official['validation'] else
                'test' if relative in official['test'] else assignments[speaker_id(relative)])
        word = relative.split('/', 1)[0]
        label = WORD_TO_LABEL.get(word, UNKNOWN_LABEL)
        available[role][label].append({'path': relative, 'label': label,
            'class_name': CLASS_NAMES[label], 'word': word, 'kind': 'word',
            'speaker_id': speaker_id(relative), 'background_id': None,
            'official_split': role if role in ('validation', 'test') else 'train'})
    backgrounds = sorted(p.relative_to(root).as_posix() for p in (root / '_background_noise_').glob('*.wav'))
    if len(backgrounds) != 6:
        raise ValueError(f'exactly six whole background WAVs required; found {len(backgrounds)}')
    random.Random(f'{split_seed}:backgrounds').shuffle(backgrounds)
    roles, summaries, metadata = {}, {}, {}
    for role, background in zip(ROLES, backgrounds):
        rng = random.Random(f'{split_seed}:{role}:class_caps')
        rows, counts, under = [], {}, {}
        for label, class_name in enumerate(CLASS_NAMES):
            if label == SILENCE_LABEL:
                continue
            pool = available[role][label]
            chosen = rng.sample(pool, min(caps[role], len(pool)))
            rows.extend(chosen)
            counts[class_name] = len(chosen)
            if len(chosen) < caps[role]:
                under[class_name] = {'available': len(pool), 'selected': len(chosen), 'requested': caps[role]}
        if background not in metadata:
            metadata[background] = wav_metadata(root / background)
        noise = metadata[background]
        if noise['source_samples'] < 16000:
            raise ValueError('background recordings must be at least one second')
        for _ in range(caps[role]):
            rows.append({'path': background, 'label': SILENCE_LABEL, 'class_name': 'silence',
                'kind': 'silence', 'speaker_id': None, 'background_id': background,
                'official_split': 'background',
                'crop_start_samples': rng.randrange(noise['source_samples'] - 16000 + 1)})
        for row in rows:
            if row['path'] not in metadata:
                metadata[row['path']] = wav_metadata(root / row['path'])
            row.update(metadata[row['path']])
        rows.sort(key=lambda r: (r['label'], r['path'], r.get('crop_start_samples', 0)))
        if {r['label'] for r in rows} != set(range(12)):
            raise ValueError(f'{role}: at least one of twelve classes is absent')
        roles[role] = rows
        counts['silence'] = caps[role]
        summaries[role] = {'records': len(rows), 'class_counts': counts,
            'requested_per_class': caps[role], 'underrepresented': under,
            'available_word_class_counts': {CLASS_NAMES[i]: len(pool) for i, pool in enumerate(available[role]) if i != SILENCE_LABEL},
            'speaker_count_selected': len({r['speaker_id'] for r in rows if r['kind'] == 'word'}),
            'background_source': background, 'background_source_count': 1,
            'silence_overlap': _overlap_summary([r for r in rows if r['kind'] == 'silence'])}
    manifest = {'protocol': PROTOCOL, 'dataset': 'Speech Commands v0.02',
        'dataset_root': 'datasets/speech_commands/extracted', 'paths_relative_to': 'dataset_root',
        'split_seed': split_seed, 'training_seeds': [42, 43, 44, 45, 46],
        'class_names': CLASS_NAMES, 'caps_per_class': caps,
        'official_lists_sha256': {f: sha256_file(root / f) for f in ('validation_list.txt', 'testing_list.txt')},
        'all_word_file_count': len(paths), 'speaker_groups': {**groups, **{r: sorted(s) for r, s in held_speakers.items()}},
        'roles': roles, 'role_summary': summaries,
        'source_file_sha256': {p: m['sha256'] for p, m in sorted(metadata.items())},
        'feature_specification': {'extractor': 'cig_mpq.audio.mfcc_features',
            'sample_rate': 16000, 'actual_header_rate_checked': True,
            'clip_samples': 16000, 'window_samples': 480, 'stride_samples': 320,
            'fft_size': 512, 'mel_filters': 40, 'coefficients': 10,
            'normalization_role': 'train', 'old_processed_data_reused': False},
        'limitations': ['One background recording per role; within-role silence crops may overlap.',
            'Silence examples are not independent recording sources; estimates do not establish generalization across background types.',
            'Official validation/test word membership is preserved, with deterministic class caps and no replacement.']}
    manifest['audit'] = audit_manifest(manifest)
    return manifest


def row_features(root, row):
    path = Path(root) / row['path']
    # Inspect the real header rather than trusting the manifest or MFCC default.
    with wave.open(str(path), 'rb') as stream:
        rate = stream.getframerate()
        if rate != 16000:
            raise ValueError(f'{path}: actual sample rate {rate}; expected 16000')
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f'{path}: expected mono PCM16')
        if row['kind'] == 'silence':
            stream.setpos(row['crop_start_samples'])
            waveform = np.frombuffer(stream.readframes(16000), dtype='<i2').astype(np.float32) / 32768.
        else:
            waveform = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32) / 32768.
    return mfcc_features(pad_or_truncate(waveform, 16000), sample_rate=rate)


def normalize_features(arrays):
    train = arrays['train']
    mean = train.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.maximum(train.std(axis=0, keepdims=True, dtype=np.float64), 1e-5).astype(np.float32)
    return {role: ((features - mean) / std).astype(np.float32) for role, features in arrays.items()}, mean, std


def prepare_cache(root, cache, manifest, workers=4, log=print):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / 'split_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('existing round2 feature cache manifest differs; refuse reuse')
    else:
        write_json(manifest_path, manifest)
    digest = sha256_file(manifest_path)
    receipt_path = cache / 'cache_receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['manifest_sha256'] != digest:
            raise ValueError('cached features have a different manifest')
        for filename, expected in receipt['files_sha256'].items():
            if sha256_file(cache / filename) != expected:
                raise ValueError(f'feature cache hash mismatch: {filename}')
        log({'phase': 'features', 'status': 'verified_cache_reused'})
        return digest
    arrays = {}
    for role in ROLES:
        rows = manifest['roles'][role]
        log({'phase': 'features', 'role': role, 'status': 'extracting', 'records': len(rows)})
        with ThreadPoolExecutor(max_workers=workers) as executor:
            arrays[role] = np.stack(list(executor.map(lambda row: row_features(root, row), rows))).astype(np.float32)
        if not np.isfinite(arrays[role]).all():
            raise ValueError(f'nonfinite features in {role}')
    normalized, mean, std = normalize_features(arrays)
    for role in ROLES:
        rows = manifest['roles'][role]
        np.savez_compressed(cache / f'{role}.npz', features=normalized[role],
            labels=np.array([r['label'] for r in rows], dtype=np.int64),
            paths=np.array([r['path'] for r in rows]),
            speaker_ids=np.array([r['speaker_id'] or '' for r in rows]),
            background_ids=np.array([r['background_id'] or '' for r in rows]),
            source_sha256=np.array([r['sha256'] for r in rows]), manifest_sha256=np.array(digest))
    np.savez(cache / 'normalization.npz', mean=mean, std=std, fit_role=np.array('train'))
    files = [f'{role}.npz' for role in ROLES] + ['normalization.npz']
    write_json(receipt_path, {'protocol': PROTOCOL, 'manifest_sha256': digest,
        'files_sha256': {name: sha256_file(cache / name) for name in files}})
    log({'phase': 'features', 'status': 'complete', 'manifest_sha256': digest})
    return digest


def freeze_selection(run, rows):
    """All selection/temperature arrays must exist before decisions are frozen."""
    from scripts.review_reanalysis import freeze_decisions
    run = Path(run)
    if any((run / f'mask{i}_logits.npz').exists() for i in range(128)) or (run / 'uniform_w4_logits.npz').exists():
        raise RuntimeError('test predictions already exist before frozen decisions')
    if {r['mask'] for r in rows} != set(range(128)):
        raise ValueError('freeze requires the complete seven-layer, 128-candidate space')
    for row in rows:
        with np.load(run / f"mask{row['mask']}_selection_logits.npz", allow_pickle=False) as archive:
            if set(archive.files) != {'selection_logits', 'selection_labels', 'temperature_logits', 'temperature_labels',
                                     'selection_group_ids', 'temperature_group_ids'}:
                raise ValueError('selection phase must contain exactly selection and temperature arrays')
    write_json(run / 'selection_candidates.json', rows)
    write_json(run / 'frozen_decisions.json', freeze_decisions(rows, budgets=BUDGETS))
    names = ['selection_candidates.json', 'frozen_decisions.json'] + [f'mask{i}_selection_logits.npz' for i in range(128)]
    names += [name for name in ('uniform_w4_selection.json', 'uniform_w4_selection_logits.npz') if (run / name).exists()]
    write_json(run / 'freeze_receipt.json', {'protocol': PROTOCOL, 'candidate_count': 128,
        'budgets': BUDGETS, 'test_outputs_present_at_freeze': False,
        'files_sha256': {name: sha256_file(run / name) for name in names}})
    return require_test_gate(run)


def require_test_gate(run):
    from scripts.review_reanalysis import freeze_decisions
    run = Path(run)
    receipt_path = run / 'freeze_receipt.json'
    if not receipt_path.exists():
        raise RuntimeError('test evaluation requires saved selection outputs and frozen decisions')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['candidate_count'] != 128 or receipt['budgets'] != list(BUDGETS):
        raise RuntimeError('invalid frozen candidate space or budgets')
    for name, expected in receipt['files_sha256'].items():
        if sha256_file(run / name) != expected:
            raise RuntimeError(f'frozen selection artifact changed: {name}')
    rows = json.loads((run / 'selection_candidates.json').read_text(encoding='utf-8'))
    frozen = json.loads((run / 'frozen_decisions.json').read_text(encoding='utf-8'))
    if len(rows) != 128 or frozen != freeze_decisions(rows, budgets=BUDGETS):
        raise RuntimeError('frozen decisions do not match selection-only records')
    return sha256_file(run / 'frozen_decisions.json')
