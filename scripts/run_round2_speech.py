"""Run the five-seed, source-disjoint Speech Commands second-review replication.

.venv/Scripts/python.exe -u -B scripts/run_round2_speech.py
All new data, models, manifests, predictions and progress live in results/round2/speech.
"""
import argparse
import copy
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from torch import nn

from cig_mpq.metrics import classification_metrics
from cig_mpq.models import AudioDSCNN
from cig_mpq.quantization import (calibrate_activation_ranges,
    install_activation_quantization_hooks, quantize_mixed_precision)
from cig_mpq.reproducibility import seed_everything
from cig_mpq.round2_speech import (ROLES, CLASS_NAMES, DEFAULT_CAPS, SPLIT_SEED,
    PROTOCOL, build_manifest, prepare_cache, sha256_file, write_json,
    freeze_selection, require_test_gate)
from cig_mpq.temperature import fit_temperature
from cig_mpq.training import evaluate_logits
from scripts.run_pareto_pilot import bits_from_mask, weight_bytes


class TensorBatches:
    """Cached device tensors, indexed once per batch rather than once per sample."""
    def __init__(self, features, labels, batch_size, shuffle=False):
        self.features, self.labels = features, labels
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        order = torch.randperm(len(self.labels), device=self.labels.device) if self.shuffle else None
        for start in range(0, len(self.labels), self.batch_size):
            index = slice(start, start + self.batch_size) if order is None else order[start:start + self.batch_size]
            yield self.features[index], self.labels[index]

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='reuse completed training and verified frozen evaluations')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 50 <= args.epochs <= 80 or not 20 <= args.min_epochs <= args.epochs or args.patience != 10:
        parser.error('protocol requires max epochs 50..80, min epochs >=20, patience 10')
    if args.batch_size != 256 or any(seed not in range(42, 47) for seed in args.seeds):
        parser.error('protocol fixes batch size 256 and training seeds 42..46')
    return args


def group_ids(rows):
    return np.array([f"speaker:{r['speaker_id']}" if r['kind'] == 'word'
                     else f"background:{r['background_id']}" for r in rows])


def load_role(cache, role, device, batch_size):
    with np.load(cache / f'{role}.npz', allow_pickle=False) as archive:
        features = torch.from_numpy(archive['features']).float().unsqueeze(1).to(device)
        labels = torch.from_numpy(archive['labels']).long().to(device)
    return TensorBatches(features, labels, batch_size, shuffle=role == 'train')


def training_configuration(args):
    return {'epochs': args.epochs, 'min_epochs': args.min_epochs,
        'patience': args.patience, 'batch_size': args.batch_size,
        'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': 1e-4,
        'scheduler': 'CosineAnnealingLR', 'loss': 'cross_entropy_no_label_smoothing',
        'checkpoint_criterion': 'validation_nll', 'minimum_nll_improvement': .0001}


def train_seed(seed, args, run, loaders, device, manifest_digest, provenance, log):
    configuration = training_configuration(args)
    summary_path = run / 'training_summary.json'
    model = AudioDSCNN(num_classes=12, in_channels=1).to(device)
    if args.resume and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary['manifest_sha256'] != manifest_digest or summary['training_configuration'] != configuration:
            raise ValueError('resume training manifest/configuration differs')
        checkpoint = torch.load(run / 'model.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        log({'seed': seed, 'phase': 'training', 'status': 'completed_checkpoint_reused'})
        return model.eval(), summary
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_nll, best_epoch, best_state, history = float('inf'), 0, None, []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = torch.zeros((), device=device)
        seen = 0
        learning_rate = optimizer.param_groups[0]['lr']
        for features, labels in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(features), labels)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(labels)
            seen += len(labels)
        validation, _, _ = evaluate_logits(model, loaders['validation'], device)
        scheduler.step()
        row = {'epoch': epoch, 'learning_rate': learning_rate, 'train_loss': float(total) / seen,
            'validation_nll': validation['nll'], 'validation_accuracy': validation['accuracy'],
            'elapsed_seconds': time.time() - started}
        history.append(row)
        if validation['nll'] < best_nll - .0001:
            best_nll, best_epoch = validation['nll'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state_dict': best_state, 'classes': CLASS_NAMES, 'seed': seed,
                'best_epoch': best_epoch, 'best_validation_nll': best_nll,
                'manifest_sha256': manifest_digest, 'training_configuration': configuration,
                'provenance': provenance}, run / 'model.pt')
        write_json(run / 'history.json', history)
        log({'seed': seed, 'phase': 'training', **row, 'best_epoch': best_epoch})
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    summary = {'seed': seed, 'protocol': PROTOCOL, 'best_epoch': best_epoch,
        'best_validation_nll': best_nll, 'epochs_run': len(history),
        'early_stopped': len(history) < args.epochs, 'maximum_epochs_reached': len(history) == args.epochs,
        'training_status': 'complete', 'evaluation_status': 'pending_selection',
        'training_configuration': configuration, 'manifest_sha256': manifest_digest,
        'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'torch_threads': torch.get_num_threads(), 'cached_device_tensors': True,
        'training_wall_seconds': time.time() - started, 'provenance': provenance,
        'convergence_claim': 'requires_curve_and_cross_seed_review',
        'test_used_for_training_or_tuning': False,
        'metrics_scoring': 'stable_log_softmax_uncapped_NLL',
        'temperature_fit': 'existing bounded scalar fit_temperature on temperature role only',
        'evaluation_protocol': 'all_selection_temperature_outputs_then_frozen_decisions_then_test'}
    write_json(summary_path, summary)
    return model, summary


def evaluate_candidates(seed, model, args, run, cache, loaders, device, manifest, summary, log):
    layers = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if len(layers) != 7:
        raise ValueError('AudioDSCNN must expose all seven convolution candidates')
    baseline_bytes = weight_bytes(model, {})
    groups = {role: group_ids(manifest['roles'][role]) for role in ('selection', 'temperature', 'test')}
    if args.resume and (run / 'freeze_receipt.json').exists():
        require_test_gate(run)
        rows = json.loads((run / 'selection_candidates.json').read_text(encoding='utf-8'))
    else:
        if any((run / f'mask{i}_logits.npz').exists() for i in range(128)):
            raise RuntimeError('test outputs exist before this freeze; preserve results and investigate')
        rows = []
        for mask in range(128):
            bits = bits_from_mask(mask, layers)
            candidate = copy.deepcopy(model)
            quantize_mixed_precision(candidate, bits, default_bits=8)
            ranges = calibrate_activation_ranges(candidate, loaders['quantization'], device)
            handles = install_activation_quantization_hooks(candidate, bits=8, ranges=ranges)
            try:
                _, temperature_logits, temperature_labels = evaluate_logits(candidate, loaders['temperature'], device)
                temperature = fit_temperature(temperature_logits, temperature_labels)
                raw, selection_logits, selection_labels = evaluate_logits(candidate, loaders['selection'], device)
            finally:
                for handle in handles:
                    handle.remove()
            row = {'seed': seed, 'mask': mask, 'config': f'static:{mask}', 'bits': bits,
                'layer_names': layers, 'activation_bits': 8, 'activation_ranges': ranges,
                'temperature': temperature, 'weight_bytes': weight_bytes(model, bits),
                'flash_fraction': weight_bytes(model, bits) / baseline_bytes,
                'search_raw': raw, 'search_calibrated': classification_metrics(selection_logits / temperature, selection_labels),
                'evaluation_status': 'selection_only'}
            rows.append(row)
            np.savez_compressed(run / f'mask{mask}_selection_logits.npz',
                selection_logits=selection_logits, selection_labels=selection_labels,
                temperature_logits=temperature_logits, temperature_labels=temperature_labels,
                selection_group_ids=groups['selection'], temperature_group_ids=groups['temperature'])
            write_json(run / 'selection_candidates.json', rows)
            log({'seed': seed, 'phase': 'selection', 'mask': mask, 'completed': len(rows), 'total': 128})
            del candidate
        # A prespecified full-network W4 control; excluded from allocation search.
        uniform_bits = {name: 4 for name, module in model.named_modules() if isinstance(module, (nn.Conv2d, nn.Linear))}
        candidate = copy.deepcopy(model)
        quantize_mixed_precision(candidate, uniform_bits)
        ranges = calibrate_activation_ranges(candidate, loaders['quantization'], device)
        handles = install_activation_quantization_hooks(candidate, bits=8, ranges=ranges)
        try:
            _, t_logits, t_labels = evaluate_logits(candidate, loaders['temperature'], device)
            temperature = fit_temperature(t_logits, t_labels)
            raw, s_logits, s_labels = evaluate_logits(candidate, loaders['selection'], device)
        finally:
            for handle in handles:
                handle.remove()
        write_json(run / 'uniform_w4_selection.json', {'seed': seed, 'control': 'full_network_uniform_w4_fixed_a8',
            'bits': uniform_bits, 'activation_bits': 8, 'activation_ranges': ranges,
            'temperature': temperature, 'weight_bytes': weight_bytes(model, uniform_bits),
            'search_raw': raw, 'search_calibrated': classification_metrics(s_logits / temperature, s_labels),
            'evaluation_status': 'selection_only', 'included_in_search': False})
        np.savez_compressed(run / 'uniform_w4_selection_logits.npz',
            selection_logits=s_logits, selection_labels=s_labels,
            temperature_logits=t_logits, temperature_labels=t_labels,
            selection_group_ids=groups['selection'], temperature_group_ids=groups['temperature'])
        del candidate
        freeze_selection(run, rows)
    digest = require_test_gate(run)
    summary.update({'evaluation_status': 'frozen_before_test', 'frozen_decisions_sha256': digest,
        'selection_candidate_count': 128, 'layer_names': layers,
        'uniform_w8_weight_bytes': baseline_bytes, 'classifier_weight_bits': 8,
        'storage_scope': 'All seven Conv weights plus fixed-W8 classifier weights; excludes bias, BN, scales and activation memory.',
        'activation_quantization': 'candidate-specific fixed Conv-output A8 ranges from quantization role',
        'quantization_implementation': 'float fake quantization; no integer deployment or latency claim'})
    write_json(run / 'training_summary.json', summary)
    log({'seed': seed, 'phase': 'freeze', 'status': 'complete', 'frozen_decisions_sha256': digest})
    # This is the first loading or model evaluation of test tensors for this seed.
    test_loader = load_role(cache, 'test', device, args.batch_size)
    evaluated = []
    old_rows = {}
    if args.resume and (run / 'candidates.json').exists():
        old_rows = {r['mask']: r for r in json.loads((run / 'candidates.json').read_text(encoding='utf-8'))
                    if r.get('evaluation_status') == 'test_evaluated_after_freeze'}
    for source in rows:
        row = copy.deepcopy(source)
        mask = row['mask']
        if mask in old_rows and (run / f'mask{mask}_logits.npz').exists():
            evaluated.append(old_rows[mask])
            continue
        candidate = copy.deepcopy(model)
        quantize_mixed_precision(candidate, row['bits'], default_bits=8)
        handles = install_activation_quantization_hooks(candidate, bits=8, ranges=row['activation_ranges'])
        try:
            raw, test_logits, test_labels = evaluate_logits(candidate, test_loader, device)
        finally:
            for handle in handles:
                handle.remove()
        row.update({'eval_raw': raw,
            'eval_calibrated': classification_metrics(test_logits / row['temperature'], test_labels),
            'evaluation_status': 'test_evaluated_after_freeze'})
        with np.load(run / f'mask{mask}_selection_logits.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        np.savez_compressed(run / f'mask{mask}_logits.npz', **arrays,
            test_logits=test_logits, test_labels=test_labels, test_group_ids=groups['test'])
        evaluated.append(row)
        write_json(run / 'candidates.json', evaluated + [r for r in rows if r['mask'] > mask])
        log({'seed': seed, 'phase': 'test', 'mask': mask, 'completed': len(evaluated), 'total': 128})
        del candidate
    if require_test_gate(run) != digest:
        raise RuntimeError('decisions changed during test evaluation')
    write_json(run / 'candidates.json', evaluated)
    control = json.loads((run / 'uniform_w4_selection.json').read_text(encoding='utf-8'))
    candidate = copy.deepcopy(model)
    quantize_mixed_precision(candidate, control['bits'])
    handles = install_activation_quantization_hooks(candidate, bits=8, ranges=control['activation_ranges'])
    try:
        raw, test_logits, test_labels = evaluate_logits(candidate, test_loader, device)
    finally:
        for handle in handles:
            handle.remove()
    control.update({'test_raw': raw,
        'test_calibrated': classification_metrics(test_logits / control['temperature'], test_labels),
        'evaluation_status': 'test_evaluated_after_freeze', 'frozen_decisions_sha256': digest})
    with np.load(run / 'uniform_w4_selection_logits.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    np.savez_compressed(run / 'uniform_w4_logits.npz', **arrays,
        test_logits=test_logits, test_labels=test_labels, test_group_ids=groups['test'])
    write_json(run / 'uniform_w4.json', control)
    summary.update({'evaluation_status': 'complete', 'candidate_count': len(evaluated),
        'test_candidate_count': len(evaluated), 'complete_candidate_space': len(evaluated) == 128,
        'full_network_uniform_w4': 'uniform_w4.json',
        'prediction_group_ids': 'speaker:<speaker_id> for words; background:<dataset-relative-file> for silence',
        'silence_bootstrap_limitation': 'Only one background source per role; keep silence fixed or stratify it, do not bootstrap crops as independent sources.'})
    write_json(run / 'training_summary.json', summary)


def main():
    args = parse_args()
    output = ROOT / 'results/round2/speech'
    output.mkdir(parents=True, exist_ok=True)
    stream = (output / 'run.log').open('a', encoding='utf-8', buffering=1)
    status_path = output / 'status.json'
    started = time.time()
    status = {'protocol': PROTOCOL, 'pid': os.getpid(), 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(), 'seeds_requested': args.seeds,
        'seeds': {str(seed): {'status': 'pending'} for seed in args.seeds},
        'output_directory': 'results/round2/speech',
        'configuration': training_configuration(args), 'candidate_count_per_seed': 128}

    def log(record):
        record = {'utc': datetime.now(timezone.utc).isoformat(), **record}
        line = json.dumps(record, ensure_ascii=False)
        print(line, flush=True)
        print(line, file=stream, flush=True)
        status['latest_progress'] = record
        status['elapsed_seconds'] = time.time() - started
        write_json(status_path, status)

    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type != 'cuda':
            raise RuntimeError('this full run requires the available CUDA GPU')
        for seed in args.seeds:
            if (output / f'seed{seed}/training_summary.json').exists() and not args.resume:
                raise FileExistsError(f'seed{seed} results exist; use --resume to verify and continue')
        log({'phase': 'manifest', 'status': 'building', 'split_seed': SPLIT_SEED})
        root = ROOT / 'datasets/speech_commands/extracted'
        manifest = build_manifest(root, DEFAULT_CAPS, SPLIT_SEED)
        write_json(output / 'source_audit.json', manifest['audit'])
        write_json(output / 'metadata.json', {key: value for key, value in manifest.items() if key not in ('roles', 'source_file_sha256')})
        write_json(output / 'plan.json', {'approved_scope': 'independent second-review speech replication',
            'files_owned': ['cig_mpq/round2_speech.py', 'scripts/run_round2_speech.py', 'tests/test_round2_speech.py', 'results/round2/speech*'],
            'steps': ['fresh source-disjoint manifest and MFCC cache',
                'five independent training seeds sharing the same role split',
                'all 128 selection and temperature outputs per seed',
                'freeze budgets 1,.95,.9,.85,.8 using existing freeze_decisions',
                'evaluate all 128 tests only after freezing'],
            'group_ids': 'group_ids.json per seed and selection/test_group_ids in every final prediction NPZ',
            'hyperparameters': training_configuration(args), 'no_novel_effect_claims': True})
        log({'phase': 'manifest', 'status': 'audited', 'source_disjoint': manifest['audit']['passed'],
            'records': {r: s['records'] for r, s in manifest['role_summary'].items()},
            'underrepresented': {r: s['underrepresented'] for r, s in manifest['role_summary'].items()}})
        cache = output / 'features'
        digest = prepare_cache(root, cache, manifest, args.workers, log)
        if args.prepare_only:
            status['status'] = 'prepared_only'
            log({'phase': 'data', 'status': 'complete', 'manifest_sha256': digest})
            return
        provenance = {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / 'cig_mpq/round2_speech.py', ROOT / 'cig_mpq/audio.py',
            ROOT / 'cig_mpq/models.py', ROOT / 'cig_mpq/quantization.py',
            ROOT / 'cig_mpq/training.py', ROOT / 'cig_mpq/metrics.py',
            ROOT / 'cig_mpq/temperature.py', ROOT / 'scripts/review_reanalysis.py')}
        loaders = {role: load_role(cache, role, device, args.batch_size) for role in ROLES if role != 'test'}
        for seed in args.seeds:
            seed_everything(seed)
            run = output / f'seed{seed}'
            run.mkdir(parents=True, exist_ok=True)
            write_json(run / 'split_manifest.json', manifest)
            write_json(run / 'group_ids.json', {r: group_ids(manifest['roles'][r]).tolist() for r in ROLES})
            status['seeds'][str(seed)]['status'] = 'training'
            log({'seed': seed, 'phase': 'training', 'status': 'started'})
            model, summary = train_seed(seed, args, run, loaders, device, digest, provenance, log)
            summary['record_counts'] = {r: len(manifest['roles'][r]) for r in ROLES}
            status['seeds'][str(seed)]['status'] = 'selection_then_freeze_then_test'
            evaluate_candidates(seed, model, args, run, cache, loaders, device, manifest, summary, log)
            status['seeds'][str(seed)] = {'status': 'complete', 'epochs_run': summary['epochs_run'],
                'best_epoch': summary['best_epoch'], 'selection_candidates': 128, 'test_candidates': 128}
            log({'seed': seed, 'phase': 'seed', 'status': 'complete', 'candidate_count': 128})
            del model
        status['status'] = 'complete'
        log({'phase': 'run', 'status': 'complete', 'seeds_complete': len(args.seeds),
            'candidate_total': len(args.seeds) * 128})
    except BaseException as error:
        status['status'] = 'failed'
        status['error'] = f'{type(error).__name__}: {error}'
        log({'phase': 'run', 'status': 'failed', 'error': status['error']})
        traceback.print_exc(file=stream)
        stream.flush()
        raise
    finally:
        stream.close()


if __name__ == '__main__':
    main()
