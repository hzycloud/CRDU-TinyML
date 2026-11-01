"""Run the CLADO allocation-objective adaptation on existing CIFAR checkpoints.

CPU only; never opens test images or logits. Results are independent of archived
candidate metrics. Run with Python -B to avoid writes outside the owned paths:
  .venv/Scripts/python.exe -B scripts/run_round2_baseline.py
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.stats import spearmanr
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cig_mpq.data import build_imagefolder
from cig_mpq.models import DSCNN
from cig_mpq.quantization import fake_quantize_per_channel
from cig_mpq.round2_baseline import (
    candidate_spec, clado_matrix, enumerate_scores, measure_clado_losses,
    project_psd, select_probe_ids,
)

OUTPUT = ROOT / 'results/round2/baselines'
CHECKPOINTS = ROOT / 'results/review_revision/cifar_static'
COMMIT = '7a139496bef94582f3ffb6790d18c00eb404bea5'
BUDGETS = (1., .95, .90, .85, .80)
METHOD = 'CLADO loss finite-difference + PSD, fixed-A8 CIFAR/backend adaptation'
ADAPTATIONS = [
    'CIFAR-10 DSCNN checkpoints seeds 42..46; six archived variable Conv layers, W4/W8 choices.',
    'Workspace symmetric per-output-channel weight fake quantizer replaces paper default per-tensor/MQBench calibration.',
    'Noncandidate Conv/Linear weights fixed W8; biases and BN stay checkpoint FP. Candidate reference weights remain FP.',
    'All probes use the same archived all-W8 static A8 output ranges calibrated on 2000 quantization-role inputs.',
    'Holding these ranges fixed defines one function during sensitivity measurement; archived candidate evaluations recalibrate ranges per candidate.',
    '256 uniformly sampled quantization-role labeled examples, disjoint from training/validation/temperature/selection and official test namespaces.',
    'Same-layer mutually exclusive choice entries zero before projection, following official Ltilde2CachedGrad.',
    'Float64 loss accumulation and matrix algebra; model and fake-quantized forward operators remain float32.',
    'Exact enumeration of 64 feasible one-hot assignments replaces the IQP solver. This is an allocation score control, not benchmark reproduction.',
    'No accuracy filter is used here. Parent may apply its matching selection-accuracy filter to these frozen scores.',
]
LIMITATIONS = [
    'Small-set loss differences approximate the directional loss Hessian only under a local quadratic/negligible-gradient assumption.',
    'The reference with fixed W8 noncandidate weights/A8 rounding need not be stationary or smooth; no claim of an exact Hessian.',
    'PSD projection can change rankings substantially and couples initially zero same-layer option entries; both raw and projected objectives are saved.',
    'Disjoint 128-example halves describe sensitivity-set instability; they are not independent training seeds or a confidence interval.',
    'No published ImageNet results reproduced, no test-based coefficient fitting, no speedup or accuracy-improvement claim.',
    'Weight payload budgets omit metadata, biases, BN, packing, activation SRAM and hardware execution overhead, matching the existing weight_bytes convention.',
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def source_manifest():
    local = ['cig_mpq/round2_baseline.py', 'scripts/run_round2_baseline.py',
             'tests/test_round2_baseline.py', 'cig_mpq/models.py', 'cig_mpq/quantization.py',
             'cig_mpq/data.py', 'scripts/run_review_cifar.py']
    primary = [
        ('clado_2307.05657v2.html', 'https://arxiv.org/html/2307.05657v2'),
        ('official_prep_mpqco_clado.py', f'https://raw.githubusercontent.com/JamesTuna/CLADO_MPQ/{COMMIT}/CLADO_SampleRun/prep_mpqco_clado.py'),
        ('official_optimize.py', f'https://raw.githubusercontent.com/JamesTuna/CLADO_MPQ/{COMMIT}/CLADO_SampleRun/optimize.py'),
    ]
    return {'official_repository_commit': COMMIT,
            'local_files_sha256': {name: sha256(ROOT / name) for name in local},
            'primary_sources': [{'path': str((OUTPUT / 'sources' / name).relative_to(ROOT)),
                                 'url': url, 'sha256': sha256(OUTPUT / 'sources' / name)}
                                for name, url in primary],
            'equations': {'diagonal': 'G[u,u] = 2*(L_u - L_ref), eq 12',
                          'cross_layer': 'G[u,v] = L_uv + L_ref - L_u - L_v, eq 13',
                          'projection': 'G_psd = U diag(max(eigenvalues(G),0)) U^T, Algorithm 1',
                          'objective': 'alpha^T G_psd alpha; one W4/W8 option per layer, eq 11'},
            'official_code_sections': ['prep_mpqco_clado.py: perturb_loss and CLADO sensitivities',
                                       'optimize.py: Ltilde2CachedGrad and MIQCP_optimize PSD branch'],
            'implementation': 'Independent implementation of primary-source equations; official code archived for inspection, never imported.'}


def objective_details(records, subset, n_layers):
    means = {name: float(values[subset].mean()) for name, values in records.items()}
    singles = [means[f'single:{i}'] for i in range(2*n_layers)]
    pairs = {(int(name.split(':')[1]), int(name.split(':')[2])): value
             for name, value in means.items() if name.startswith('pair:')}
    raw = clado_matrix(means['reference'], singles, pairs)
    projected, eigenvalues = project_psd(raw)
    return {'mean_losses': means, 'G_raw': raw.tolist(), 'G_psd': projected.tolist(),
            'raw_eigenvalues': eigenvalues.tolist(),
            'negative_eigenvalue_count': int((eigenvalues < 0).sum()),
            'negative_eigenvalue_mass': float(-eigenvalues[eigenvalues < 0].sum()),
            'projection_frobenius_norm': float(np.linalg.norm(projected - raw)),
            'scores_raw': enumerate_scores(raw, n_layers),
            'scores': enumerate_scores(projected, n_layers)}


def budget_minima(scores, costs):
    return {str(b): {'mask': min((m for m in range(64) if costs[str(m)] <= b*costs['0'] + 1e-9),
                                key=lambda m: (scores[str(m)], m))}
            for b in BUDGETS}


def run_seed(seed, args, dataset, sources):
    start = time.perf_counter()
    run = CHECKPOINTS / f'seed{seed}'
    output = OUTPUT / f'seed{seed}'
    if output.exists():
        raise FileExistsError(f'{output} exists; preserve the completed run rather than overwriting')
    checkpoint_path = run / 'model.pt'
    split_path = run / 'split_manifest.json'
    candidates_path = run / 'candidates.json'
    input_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in (checkpoint_path, split_path, candidates_path)}
    split = json.loads(split_path.read_text(encoding='utf-8'))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['seed'] != seed or split['seed'] != seed or checkpoint['manifest_sha256'] != sha256(split_path):
        raise ValueError('checkpoint seed or archived split-manifest identity mismatch')
    if checkpoint['classes'] != dataset.classes:
        raise ValueError('checkpoint and training-folder class orders differ')
    names, ranges = candidate_spec(json.loads(candidates_path.read_text(encoding='utf-8')))
    selected = select_probe_ids(split, args.samples, seed)
    index = {Path(p).relative_to(dataset.root).as_posix(): i for i, (p, _) in enumerate(dataset.samples)}
    examples, manifest_rows = [], []
    for position, source_id in enumerate(selected):
        dataset_index = index[source_id.replace('\\', '/')]
        path, label = dataset.samples[dataset_index]
        raw_hash = sha256(path)
        if raw_hash != split['file_sha256'][f'train/{source_id}']:
            raise ValueError(f'raw quantization image hash changed: {source_id}')
        tensor, observed_label = dataset[dataset_index]
        if label != observed_label:
            raise ValueError('dataset label mismatch')
        examples.append((tensor, label))
        manifest_rows.append({'source_id': source_id, 'namespace': 'official_train',
                              'role': 'quantization', 'sha256': raw_hash, 'label': label,
                              'class_name': dataset.classes[label],
                              'half': 0 if position < args.samples // 2 else 1})
    inputs = torch.stack([x for x, _ in examples])
    labels = torch.tensor([y for _, y in examples], dtype=torch.long)
    model = DSCNN(num_classes=10, in_channels=3)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    if any(not isinstance(dict(model.named_modules()).get(name), nn.Conv2d) for name in names):
        raise ValueError('candidate layers are not checkpoint Conv2d modules')
    conv_names = {name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
    if set(ranges) != conv_names:
        raise ValueError('A8 static ranges must cover every checkpoint convolution exactly')
    output.mkdir()
    sample_manifest = {'seed': seed, 'checkpoint_sha256': sha256(checkpoint_path),
                       'split_manifest_sha256': sha256(split_path), 'input_files_sha256': input_hashes,
                       'role_sizes': {name: len(ids) for name, ids in split['roles'].items()},
                       'sampling': 'default_rng(20261027+checkpoint_seed).permutation(2000)[:256]; no replacement',
                       'sampling_rng_seed': 20261027 + seed, 'sample_count': args.samples,
                       'test_images_or_logits_opened': False,
                       'candidate_read_fields': ['mask', 'layer_names', 'bits', 'activation_bits', 'mask0.activation_ranges'],
                       'normalized_input_tensor_sha256': hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
                       'labels_sha256': hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
                       'normalization': {'mean': [.4914, .4822, .4465], 'std': [.2470, .2435, .2616]},
                       'examples_in_probe_order': manifest_rows}
    save_json(output / 'probe_manifest.json', sample_manifest)

    def progress(key, done, total):
        if done == 1 or done % 12 == 0 or done == total:
            print(json.dumps({'seed': seed, 'probe': key, 'completed': done, 'total': total,
                              'wall_seconds': time.perf_counter() - start}), flush=True)

    measure_start = time.perf_counter()
    records = measure_clado_losses(model, names, inputs, labels, ranges, args.batch_size, progress)
    measurement_seconds = time.perf_counter() - measure_start
    if len(records) != 73:
        raise ValueError('expected 73 reference/single/cross-layer evaluations')
    np.savez_compressed(output / 'probe_losses.npz', probe_ids=np.asarray(list(records)),
                        source_ids=np.asarray(selected), labels=labels.numpy(),
                        losses=np.stack(list(records.values())))
    full = objective_details(records, slice(None), len(names))
    halves = [objective_details(records, slice(0, args.samples // 2), len(names)),
              objective_details(records, slice(args.samples // 2, None), len(names))]
    quantization_errors, all_weight_counts = [], {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            all_weight_counts[name] = module.weight.numel()
            if name in names:
                for bits in (4, 8):
                    error = fake_quantize_per_channel(module.weight, bits).double() - module.weight.detach().double()
                    quantization_errors.append({'layer': name, 'bits': bits, 'numel': module.weight.numel(),
                                                'frobenius_error_squared': float(error.square().sum())})
    costs = {str(mask): sum((count*(4 if name in names and mask & (1 << names.index(name)) else 8)+7)//8
                           for name, count in all_weight_counts.items()) for mask in range(64)}
    primary_minima = budget_minima(full['scores'], costs)
    half_minima = [budget_minima(half['scores'], costs) for half in halves]
    vectors = [np.array([half['scores'][str(mask)] for mask in range(64)]) for half in halves]
    correlation = float(spearmanr(*vectors).statistic)
    stability = {'policy': 'Always use full 256; both prespecified 128-example halves reported, never selected/tuned.',
                 'half_sample_counts': [args.samples // 2, args.samples - args.samples // 2],
                 'halves_spearman_all_64': correlation if np.isfinite(correlation) else None,
                 'halves_score_rmse': float(np.sqrt(np.mean((vectors[0]-vectors[1])**2))),
                 'halves_max_absolute_score_difference': float(np.max(np.abs(vectors[0]-vectors[1]))),
                 'half_budget_minima': half_minima,
                 'half_budget_mask_agreements': sum(half_minima[0][str(b)] == half_minima[1][str(b)] for b in BUDGETS),
                 'budget_count': len(BUDGETS), 'half_objectives': halves}
    details = {'seed': seed, 'method': METHOD, 'layer_names': names,
               'variable_order': [{'index': 2*i+j, 'layer': name, 'bits': bits}
                                  for i, name in enumerate(names) for j, bits in enumerate((4, 8))],
               'mask_convention': 'bit i=1 selects W4, bit i=0 selects W8; layer_names order',
               'adaptations': ADAPTATIONS, 'limitations': LIMITATIONS,
               'objective': full, 'primary_scores_field': 'objective.scores (PSD projected)',
               'quantization_errors': quantization_errors,
               'noncandidate_bits': {name: 8 for name in all_weight_counts if name not in names},
               'activation_bits': 8, 'fixed_activation_ranges': ranges,
               'weight_payload_bytes_by_mask': costs, 'budget_only_exact_minima': primary_minima,
               'sample_stability': stability,
               'sources': sources, 'input_files_sha256': input_hashes,
               'probe_manifest_sha256': sha256(output / 'probe_manifest.json'),
               'probe_losses_sha256': sha256(output / 'probe_losses.npz'),
               'runtime': {'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                           'batch_size': args.batch_size, 'samples': args.samples, 'model_evaluations': len(records),
                           'forward_batches': len(records)*((args.samples+args.batch_size-1)//args.batch_size),
                           'measurement_seconds': measurement_seconds, 'seed_wall_seconds': time.perf_counter() - start,
                           'python': platform.python_version(), 'torch': torch.__version__,
                           'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform()}}
    if source_manifest() != sources or any(sha256(ROOT / path) != digest for path, digest in input_hashes.items()):
        raise ValueError('source or input files changed during the run')
    save_json(output / 'details.json', details)
    return {'scores': full['scores'], 'details': {'path': str((output / 'details.json').relative_to(ROOT)),
                                               'sha256': sha256(output / 'details.json'),
                                               'seed_wall_seconds': details['runtime']['seed_wall_seconds'],
                                               'halves_spearman_all_64': stability['halves_spearman_all_64']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--samples', type=int, default=256, choices=[256])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if (len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= set(range(42, 47))
            or args.batch_size < 1 or not 1 <= args.threads <= 4):
        parser.error('unique seeds 42..46, positive batch size, and 1..4 CPU threads required')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    score_path = OUTPUT / 'allocation_scores.json'
    if score_path.exists():
        raise FileExistsError(f'{score_path} already exists; preserve archived results')
    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = source_manifest()
    save_json(OUTPUT / 'sources.json', sources)
    dataset = build_imagefolder(ROOT / 'datasets/cifar10/extracted/cifar10/train', normalize=True)
    payload = {'method': METHOD, 'status': 'running', 'lower_score_is_better': True,
               'started_utc': datetime.now(timezone.utc).isoformat(), 'args': vars(args),
               'adaptations': ADAPTATIONS, 'limitations': LIMITATIONS,
               'sources_manifest_sha256': sha256(OUTPUT / 'sources.json'), 'seeds': {}}
    for seed in args.seeds:
        payload['seeds'][str(seed)] = run_seed(seed, args, dataset, sources)
        payload['elapsed_seconds'] = time.perf_counter() - start
        save_json(score_path, payload)
    payload['status'] = 'complete'
    payload['completed_utc'] = datetime.now(timezone.utc).isoformat()
    payload['elapsed_seconds'] = time.perf_counter() - start
    save_json(score_path, payload)
    print(json.dumps({'status': 'complete', 'output': str(score_path),
                      'seeds': list(payload['seeds']), 'wall_seconds': payload['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
