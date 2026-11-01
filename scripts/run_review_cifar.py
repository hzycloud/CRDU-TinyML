"""Auditable converged-training/static-fake-quantization replication.

New experiment, never overwrites historical pilot results. Example:
.venv/Scripts/python.exe scripts/run_review_cifar.py --seeds 42 43 44 45 46
"""
import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cig_mpq.data import build_imagefolder
from cig_mpq.models import DSCNN
from cig_mpq.reproducibility import seed_everything, group_role_split
from cig_mpq.quantization import quantize_mixed_precision, calibrate_activation_ranges, install_activation_quantization_hooks
from cig_mpq.metrics import classification_metrics
from cig_mpq.temperature import fit_temperature
from cig_mpq.training import evaluate_logits
from scripts.run_pareto_pilot import bits_from_mask, weight_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--min-epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-configs', type=int, default=64)
    parser.add_argument('--resume', action='store_true', help='reuse an archived trained checkpoint; redo evaluation in two phases')
    args = parser.parse_args()
    if args.epochs < args.min_epochs or args.patience < 1 or not 1 <= args.max_configs <= 64:
        parser.error('epochs must reach min-epochs, patience positive, max-configs 1..64')
    torch.set_num_threads(4)
    output = ROOT / 'results' / 'review_revision' / 'cifar_static'
    output.mkdir(parents=True, exist_ok=True)
    dataset = build_imagefolder(ROOT / 'datasets/cifar10/extracted/cifar10/train', normalize=True)
    test = build_imagefolder(ROOT / 'datasets/cifar10/extracted/cifar10/test', normalize=True)
    # Cache tensors once, avoiding repeated image decoding each epoch.
    print('Caching training images', flush=True)
    inputs, labels = [], []
    for x, y in DataLoader(dataset, batch_size=512, shuffle=False):
        inputs.append(x)
        labels.append(y)
    tensors = TensorDataset(torch.cat(inputs), torch.cat(labels))
    del inputs, labels
    test_inputs, test_targets = [], []
    for x, y in DataLoader(test, batch_size=512, shuffle=False):
        test_inputs.append(x)
        test_targets.append(y)
    test_tensors = TensorDataset(torch.cat(test_inputs), torch.cat(test_targets))
    del test_inputs, test_targets
    source_ids = [str(Path(path).relative_to(dataset.root)) for path, _ in dataset.samples]
    for seed in args.seeds:
        run = output / f'seed{seed}'
        resuming = args.resume and (run / 'training_summary.json').exists()
        previous_summary = json.loads((run/'training_summary.json').read_text()) if resuming else None
        if (run / 'training_summary.json').exists() and not args.resume:
            raise FileExistsError(f'{run} already exists; preserve archived results')
        run.mkdir(parents=True, exist_ok=True)
        seed_everything(seed)
        roles = group_role_split(source_ids, {'train': len(dataset)-10000, 'validation': 4000,
            'quantization': 2000, 'temperature': 2000, 'selection': 2000}, seed)
        manifest = {'seed': seed, 'roles': {role: [source_ids[i] for i in ids] for role, ids in roles.items()},
            'test': [str(Path(p).relative_to(test.root)) for p, _ in test.samples],
            'test_namespace': 'official_test', 'training_namespace': 'official_train',
            'protocol': 'six_disjoint_roles; validation_NLL_only_early_stopping; no_test_tuning'}
        # Hash raw images to make input identity auditable, not just index order.
        manifest['file_sha256'] = {f'train/{source_ids[i]}': hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for i, (p, _) in enumerate(dataset.samples)}
        manifest['file_sha256'].update({f'test/{Path(p).relative_to(test.root)}': hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                       for p, _ in test.samples})
        manifest_path = run / 'split_manifest.json'
        if resuming:
            saved_manifest = json.loads(manifest_path.read_text())
            if saved_manifest != manifest:
                raise ValueError('resume input/role manifest differs from archived training inputs')
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        loaders = {role: DataLoader(Subset(tensors, indices), batch_size=args.batch_size,
                    shuffle=role == 'train', num_workers=0) for role, indices in roles.items()}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DSCNN(num_classes=10, in_channels=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        history, best_loss, best_epoch, state = [], float('inf'), 0, None
        if resuming:
            checkpoint = torch.load(run/'model.pt',map_location='cpu',weights_only=False)
            state, best_epoch = checkpoint['state_dict'], checkpoint['best_epoch']
            history = json.loads((run/'history.json').read_text())
            for field in ('epochs','min_epochs','patience','batch_size'):
                if previous_summary['args'][field] != getattr(args,field):
                    raise ValueError(f'resume training configuration changed: {field}')
        start = time.time()
        for epoch in ([] if resuming else range(1, args.epochs+1)):
            model.train()
            total, count = 0., 0
            for x, y in loaders['train']:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
                count += len(y)
            validation, _, _ = evaluate_logits(model, loaders['validation'], device)
            scheduler.step()
            history.append(dict(epoch=epoch, train_loss=total/count, validation_nll=validation['nll'],
                                validation_accuracy=validation['accuracy'], elapsed=time.time()-start))
            if validation['nll'] < best_loss - 1e-4:
                best_loss, best_epoch = validation['nll'], epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(dict(state_dict=state, classes=dataset.classes, args=vars(args), seed=seed,
                    best_epoch=epoch, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()), run/'model.pt')
            (run/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(json.dumps(dict(seed=seed, **history[-1])), flush=True)
            if epoch >= args.min_epochs and epoch-best_epoch >= args.patience:
                break
        model.load_state_dict(state)
        model.eval()
        summary = dict(seed=seed, best_epoch=best_epoch, epochs_run=len(history),
            early_stopped=len(history)<args.epochs, maximum_epochs_reached=len(history)==args.epochs,
            convergence_claim='requires_visual_and_cross_seed_review',
            python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            device=str(device), gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
            args=vars(args), wall_seconds=time.time()-start,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if resuming:
            summary['training_script_sha256'] = previous_summary.get('training_script_sha256',previous_summary['script_sha256'])
            summary['training_wall_seconds'] = previous_summary.get('training_wall_seconds',previous_summary['wall_seconds'])
        else:
            summary['training_script_sha256'] = summary['script_sha256']
            summary['training_wall_seconds'] = summary['wall_seconds']
        summary['evaluation_protocol'] = 'all_selection_outputs_then_frozen_decisions_then_test'
        summary['prior_partial_exploratory_test_outputs'] = bool(resuming)
        summary['metrics_scoring'] = 'stable_log_softmax_uncapped_NLL'
        (run/'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        test_loader = DataLoader(test_tensors, batch_size=args.batch_size, shuffle=False)
        layers = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)][:6]
        candidates = []
        for mask in range(args.max_configs):
            bits = bits_from_mask(mask, layers)
            candidate = copy.deepcopy(model)
            quantize_mixed_precision(candidate, bits)
            ranges = calibrate_activation_ranges(candidate, loaders['quantization'], device)
            handles = install_activation_quantization_hooks(candidate, bits=8, ranges=ranges)
            try:
                _, t_logits, t_labels = evaluate_logits(candidate, loaders['temperature'], device)
                temperature = fit_temperature(t_logits, t_labels)
                raw, s_logits, s_labels = evaluate_logits(candidate, loaders['selection'], device)
            finally:
                for handle in handles:
                    handle.remove()
            row = dict(seed=seed, mask=mask, config=f'static:{mask}', bits=bits, layer_names=layers,
                activation_bits=8, activation_ranges=ranges, temperature=temperature,
                weight_bytes=weight_bytes(model,bits), flash_fraction=weight_bytes(model,bits)/weight_bytes(model,{}),
                search_raw=raw, search_calibrated=classification_metrics(s_logits/temperature,s_labels),
                evaluation_status='selection_only')
            candidates.append(row)
            np.savez_compressed(run/f'mask{mask}_selection_logits.npz', selection_logits=s_logits, selection_labels=s_labels,
                temperature_logits=t_logits, temperature_labels=t_labels)
            (run/'selection_candidates.json').write_text(json.dumps(candidates, indent=2), encoding='utf-8')
            print(json.dumps(dict(seed=seed, phase='selection',mask=mask)), flush=True)
        from scripts.review_reanalysis import freeze_decisions
        frozen = freeze_decisions(candidates)
        decision_path = run/'frozen_decisions.json'
        decision_path.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
        summary['frozen_decisions_sha256'] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        for row in candidates:
            mask = row['mask']
            candidate = copy.deepcopy(model)
            quantize_mixed_precision(candidate,row['bits'])
            handles = install_activation_quantization_hooks(candidate,bits=8,ranges=row['activation_ranges'])
            try:
                _, test_logits, test_labels = evaluate_logits(candidate,test_loader,device)
            finally:
                for handle in handles:
                    handle.remove()
            row['eval_raw'] = classification_metrics(test_logits,test_labels)
            row['eval_calibrated'] = classification_metrics(test_logits/row['temperature'],test_labels)
            row['evaluation_status'] = 'test_evaluated_after_freeze'
            with np.load(run/f'mask{mask}_selection_logits.npz') as archive:
                selection_arrays = {key:archive[key] for key in archive.files}
            np.savez_compressed(run/f'mask{mask}_logits.npz',**selection_arrays,test_logits=test_logits,test_labels=test_labels)
            (run/'candidates.json').write_text(json.dumps(candidates, indent=2), encoding='utf-8')
            print(json.dumps(dict(seed=seed,phase='test',mask=mask,test_accuracy=row['eval_raw']['accuracy'])),flush=True)
        summary['candidate_count'] = len(candidates)
        summary['complete_candidate_space'] = len(candidates)==64
        summary['total_wall_seconds'] = time.time()-start
        (run/'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
