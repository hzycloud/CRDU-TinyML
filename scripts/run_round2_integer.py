"""Auxiliary host CPU INT8 execution control, separate from W4/W8 search."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cig_mpq.data import build_imagefolder
from cig_mpq.models import DSCNN
from cig_mpq.metrics import classification_metrics
from cig_mpq.temperature import fit_temperature
from cig_mpq.round2_speech import write_json


@torch.inference_mode()
def logits(model, loader):
    outputs, labels = [], []
    for x, y in loader:
        outputs.append(model(x).numpy())
        labels.append(y.numpy())
    return np.concatenate(outputs), np.concatenate(labels)


def main():
    torch.set_num_threads(4)
    torch.backends.quantized.engine = 'fbgemm'
    output = ROOT / 'results/round2/integer'
    output.mkdir(exist_ok=True)
    run = ROOT / 'results/review_revision/cifar_static/seed42'
    split = json.loads((run / 'split_manifest.json').read_text())
    data = build_imagefolder(ROOT / 'datasets/cifar10/extracted/cifar10/train', normalize=True)
    index = {Path(path).relative_to(data.root).as_posix(): i for i, (path, _) in enumerate(data.samples)}
    loaders = {role: DataLoader(Subset(data, [index[p.replace('\\', '/')] for p in split['roles'][role]]),
        batch_size=128, shuffle=False) for role in ('quantization', 'temperature')}
    test = DataLoader(build_imagefolder(ROOT / 'datasets/cifar10/extracted/cifar10/test', normalize=True), batch_size=128)
    checkpoint = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
    model = DSCNN(10, 3).eval()
    model.load_state_dict(checkpoint['state_dict'])
    example = torch.zeros(1, 3, 32, 32)
    prepared = prepare_fx(copy.deepcopy(model), get_default_qconfig_mapping('fbgemm'), (example,))
    with torch.inference_mode():
        for x, _ in loaders['quantization']:
            prepared(x)
    quantized = convert_fx(prepared).eval()
    inventory = [{'name': name, 'type': str(type(module)), 'weight_dtype': str(module.weight().dtype)}
        for name, module in quantized.named_modules()
        if isinstance(module, (torch.ao.nn.quantized.Conv2d, torch.ao.nn.quantized.Linear))]
    assert len(inventory) == 8 and all(r['weight_dtype'] == 'torch.qint8' for r in inventory)
    results = {}
    for name, current in [('fp32', model), ('int8', quantized)]:
        scripted = torch.jit.trace(current, example)
        path = output / (name + '.pt')
        torch.jit.save(scripted, str(path))
        reloaded = torch.jit.load(str(path))
        with torch.inference_mode():
            original, restored = current(example), reloaded(example)
        assert torch.isfinite(restored).all() and torch.allclose(original, restored, rtol=0, atol=1e-6)
        tlogits, ty = logits(reloaded, loaders['temperature'])
        temperature = fit_temperature(tlogits, ty)
        z, y = logits(reloaded, test)
        assert np.isfinite(z).all()
        results[name] = {'script_archive_bytes': path.stat().st_size, 'temperature': temperature,
            'raw': classification_metrics(z, y), 'calibrated': classification_metrics(z/temperature, y),
            'serialized_reload_verified': True}
        np.savez_compressed(output / (name + '_logits.npz'), temperature_logits=tlogits, temperature_labels=ty,
            test_logits=z, test_labels=y)
    write_json(output / 'report.json', {'seed': 42, 'task': 'CIFAR-10', 'backend': 'fbgemm',
        'torch': torch.__version__, 'device': 'host CPU', 'quantized_operators': inventory,
        'calibration_samples': len(loaders['quantization'].dataset), 'results': results,
        'scope': 'Auxiliary FX static INT8 control with fused BN and default observers; different from candidate fake quantization.',
        'limitations': 'No INT4 mixed execution, MCU Flash/SRAM, latency, energy, or equivalence claim.'})
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
