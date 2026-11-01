"""CLADO eqs. (12), (13), one-hot objective (11), and Algorithm 1 PSD clipping.

This is a CIFAR/backend adaptation, not a published benchmark reproduction.
Reference: https://arxiv.org/html/2307.05657v2 . Same-layer different-choice
entries are zero before PSD projection, as in official Ltilde2CachedGrad.
"""
import numpy as np
import torch
from torch import nn

from cig_mpq.quantization import fake_quantize_per_channel, install_activation_quantization_hooks


def clado_matrix(reference_loss, single_losses, pair_losses, schemes_per_layer=2):
    """Recover Delta_i^T H_ij Delta_j from independently measured mean losses.

    Diagonal = 2*(L_i-L_0); off-diagonal = L_ij+L_0-L_i-L_j.
    The objective sums BOTH symmetric off-diagonal entries (no extra 1/2).
    Only one choice per layer is feasible, so its inter-choice entries are 0.
    """
    singles = np.asarray(single_losses, dtype=np.float64)
    if (singles.ndim != 1 or not len(singles) or schemes_per_layer < 1
            or len(singles) % schemes_per_layer or not np.isfinite(reference_loss)
            or not np.isfinite(singles).all()):
        raise ValueError('invalid finite reference/single losses or scheme count')
    expected = {(i, j) for i in range(len(singles)) for j in range(i + 1, len(singles))
                if i // schemes_per_layer != j // schemes_per_layer}
    if set(pair_losses) != expected:
        raise ValueError('exactly all cross-layer choice pairs are required')
    matrix = np.diag(2 * (singles - reference_loss))
    for (i, j), pair in pair_losses.items():
        if not np.isfinite(pair):
            raise ValueError('nonfinite pair loss')
        matrix[i, j] = matrix[j, i] = pair + reference_loss - singles[i] - singles[j]
    return matrix


def project_psd(matrix):
    """Algorithm 1: eigendecompose the symmetric matrix and clip e<0 to zero."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12)):
        raise ValueError('a finite symmetric square matrix is required')
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    return (projected + projected.T) / 2, eigenvalues


def enumerate_scores(matrix, n_layers):
    """Exact one-hot IQP objective for every binary mask; bit i=1 means W4.

    Choice order is (W4, W8) for each layer, with neither option treated as FP.
    Enumerating 64 possibilities replaces a solver for this six-layer search.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if n_layers < 1 or matrix.shape != (2*n_layers, 2*n_layers) or not np.isfinite(matrix).all():
        raise ValueError('matrix must have two choices per layer')
    scores = {}
    for mask in range(1 << n_layers):
        alpha = np.zeros(2*n_layers)
        for i in range(n_layers):
            alpha[2*i + (0 if mask & (1 << i) else 1)] = 1
        scores[str(mask)] = float(alpha @ matrix @ alpha)
    return scores


def candidate_spec(rows, n_layers=6):
    """Read only layer order, bits, A8 flag and mask-0 calibration ranges.

    No archived loss, accuracy, labels, temperature or mask-specific ranges
    enter the objective. All 64 mask definitions must match the same order.
    """
    by_mask = {row['mask']: row for row in rows}
    if n_layers not in (6, 7) or len(rows) != 1 << n_layers or set(by_mask) != set(range(1 << n_layers)):
        raise ValueError('a complete, unique binary candidate space is required (six or seven layers)')
    names = list(by_mask[0]['layer_names'])
    if len(names) != n_layers or len(set(names)) != n_layers:
        raise ValueError('the requested unique variable layers are required')
    for mask, row in by_mask.items():
        expected = {name: 4 if mask & (1 << i) else 8 for i, name in enumerate(names)}
        if row['layer_names'] != names or row['bits'] != expected or row['activation_bits'] != 8:
            raise ValueError('candidate bits/layer order/A8 do not match the protocol')
    ranges = {name: tuple(map(float, bounds)) for name, bounds in by_mask[0]['activation_ranges'].items()}
    if any(not np.isfinite([low, high]).all() or not low <= 0 <= high or low >= high
           for low, high in ranges.values()):
        raise ValueError('invalid static A8 ranges')
    return names, ranges


def select_probe_ids(manifest, count, seed):
    """Predetermined uniform sample without replacement from quantization only."""
    if (manifest['training_namespace'] != 'official_train'
            or manifest['test_namespace'] != 'official_test'):
        raise ValueError('unexpected data namespaces')
    roles = manifest['roles']
    if set(roles) != {'train', 'validation', 'quantization', 'temperature', 'selection'}:
        raise ValueError('the archived five training roles must all be present')
    seen = set()
    for ids in roles.values():
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('training roles overlap or contain duplicates')
        seen.update(ids)
    quantization = roles['quantization']
    if not 2 <= count <= len(quantization):
        raise ValueError('probe count must be 2..size of quantization role')
    indices = np.random.default_rng(20261027 + seed).permutation(len(quantization))[:count]
    return [quantization[int(i)] for i in indices]


def select_speech_probe_indices(manifest, count, seed):
    """Audit six-role source identities, then sample quantization feature rows.

    Crops can share sources within a role. Speakers, backgrounds, paths and
    content hashes cannot cross roles. Test labels are never consulted.
    """
    roles = manifest['roles']
    if set(roles) != {'train', 'validation', 'quantization', 'temperature', 'selection', 'test'}:
        raise ValueError('all six speech roles are required')
    seen = {key: set() for key in ('path', 'sha256', 'speaker_id', 'background_id')}
    for rows in roles.values():
        local = {key: {row[key] for row in rows if row.get(key)} for key in seen}
        for key, values in local.items():
            if seen[key].intersection(values):
                raise ValueError(f'cross-role speech {key} overlap')
            seen[key].update(values)
    size = len(roles['quantization'])
    if not 2 <= count <= size:
        raise ValueError('probe count must fit the quantization role')
    return np.random.default_rng(20261027 + seed).permutation(size)[:count].tolist()


@torch.no_grad()
def measure_clado_losses(model, layer_names, inputs, labels, activation_ranges, batch_size=64, progress=None):
    """Measure FP-candidate reference, two singles/layer, four pairs/layer pair.

    Noncandidate Conv/Linear weights are fixed W8, candidate reference weights
    remain checkpoint FP, biases/BN stay FP, A8 ranges are fixed for ALL probes.
    No calibration, training, gradients, dropout, BN updates, or test reads.
    Save per-example float64 cross entropy for prespecified split diagnostics.
    The caller's entire model state and individual module modes are restored.
    """
    modules = dict(model.named_modules())
    if (not layer_names or len(set(layer_names)) != len(layer_names)
            or any(not isinstance(modules.get(name), (nn.Conv2d, nn.Linear)) for name in layer_names)
            or len(inputs) != len(labels) or not len(inputs) or batch_size < 1):
        raise ValueError('invalid model layers, nonempty data, or batch size')
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    weights = {name: modules[name].weight.detach().clone() for name in layer_names}
    choices = [fake_quantize_per_channel(weights[name], bits) for name in layer_names for bits in (4, 8)]
    handles = []
    records = {}
    try:
        model.eval()
        for name, module in modules.items():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name not in weights:
                module.weight.copy_(fake_quantize_per_channel(module.weight, 8))
        handles = install_activation_quantization_hooks(model, bits=8, ranges=activation_ranges)
        probes = [('reference', ())]
        probes += [(f'single:{i}', (i,)) for i in range(len(choices))]
        probes += [(f'pair:{i}:{j}', (i, j)) for i in range(len(choices))
                   for j in range(i+1, len(choices)) if i // 2 != j // 2]
        device = next(model.parameters()).device
        for key, selection in probes:
            for name, weight in weights.items():
                modules[name].weight.copy_(weight)
            for index in selection:
                modules[layer_names[index // 2]].weight.copy_(choices[index])
            losses = []
            for start in range(0, len(inputs), batch_size):
                logits = model(inputs[start:start+batch_size].to(device))
                loss = nn.functional.cross_entropy(logits.double(), labels[start:start+batch_size].to(device),
                                                   reduction='none')
                losses.append(loss.cpu().numpy())
            records[key] = np.concatenate(losses)
            if not np.isfinite(records[key]).all():
                raise ValueError(f'nonfinite loss for {key}')
            if progress is not None:
                progress(key, len(records), len(probes))
    finally:
        for handle in handles:
            handle.remove()
        model.load_state_dict(state)
        for module, training in modes:
            module.training = training
    return records
