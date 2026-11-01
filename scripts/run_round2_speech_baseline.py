"""CLADO equation adaptation on the independent speech checkpoints.

Only quantization features/labels are used to measure scores. Frozen selection
ranges are read from mask0; no selection/test risks enter the objective.
"""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cig_mpq.models import AudioDSCNN
from cig_mpq.round2_baseline import measure_clado_losses, select_speech_probe_indices
from cig_mpq.round2_speech import write_json
from scripts.run_round2_baseline import objective_details, ADAPTATIONS, LIMITATIONS, COMMIT


def main():
    torch.set_num_threads(2)
    output = ROOT / 'results/round2/speech_baseline'
    output.mkdir(exist_ok=True)
    base = ROOT / 'results/round2/speech'
    with np.load(base / 'features/quantization.npz') as data:
        features = data['features']
        labels = data['labels']
    scores = {'method': 'CLADO loss finite-difference + PSD, speech/backend adaptation',
        'official_commit': COMMIT, 'adaptations': ADAPTATIONS,
        'speech_adaptations': ['Seven candidate convolutions, 128 exact assignments, 99 probes.',
            '256 source-audited quantization-role cached MFCC examples; no test tensor or metric reads.',
            'Same fixed all-W8 A8 ranges during all probes; main candidates use candidate-specific ranges.'],
        'limitations': LIMITATIONS, 'seeds': {}}
    for seed in range(42, 47):
        started = time.time()
        run = base / f'seed{seed}'
        manifest = json.loads((run / 'split_manifest.json').read_text(encoding='utf-8'))
        indices = select_speech_probe_indices(manifest, 256, seed)
        rows = json.loads((run / 'candidates.json').read_text())
        assert len(rows) == 128
        reference = next(r for r in rows if r['mask'] == 0)
        names = reference['layer_names']
        checkpoint = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
        model = AudioDSCNN()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        records = measure_clado_losses(model, names, torch.from_numpy(features[indices]).float().unsqueeze(1),
            torch.from_numpy(labels[indices]).long(), reference['activation_ranges'], 64)
        assert len(records) == 99
        full = objective_details(records, slice(None), len(names))
        halves = [objective_details(records, slice(i*128, (i+1)*128), len(names)) for i in range(2)]
        seedout = output / f'seed{seed}'
        seedout.mkdir(exist_ok=True)
        np.savez_compressed(seedout / 'probe_losses.npz', probe_ids=np.array(list(records)),
            indices=indices, losses=np.stack(list(records.values())))
        write_json(seedout / 'details.json', {'seed': seed, 'objective': full, 'halves': halves,
            'probe_rows': [manifest['roles']['quantization'][i] for i in indices],
            'model_sha256': hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest(),
            'layer_names': names, 'activation_ranges': reference['activation_ranges'],
            'model_evaluations': len(records), 'seconds': time.time()-started})
        scores['seeds'][str(seed)] = {'scores': full['scores']}
        write_json(output / 'allocation_scores.json', scores)
        print(json.dumps({'seed': seed, 'complete': True, 'seconds': time.time()-started}), flush=True)


if __name__ == '__main__':
    main()
