"""Fixed full-network W4 control on archived new replication checkpoints."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cig_mpq.data import build_imagefolder
from cig_mpq.models import DSCNN
from cig_mpq.quantization import quantize_mixed_precision, calibrate_activation_ranges, install_activation_quantization_hooks
from cig_mpq.training import evaluate_logits
from cig_mpq.temperature import fit_temperature
from cig_mpq.metrics import classification_metrics


def main():
    torch.set_num_threads(4)
    train=build_imagefolder(ROOT/'datasets/cifar10/extracted/cifar10/train',normalize=True)
    test=build_imagefolder(ROOT/'datasets/cifar10/extracted/cifar10/test',normalize=True)
    source_ids=[str(Path(p).relative_to(train.root)) for p,_ in train.samples]
    by_name={name:i for i,name in enumerate(source_ids)}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_tensors=None
    for path in sorted((ROOT/'results/review_revision/cifar_static').glob('seed*/training_summary.json')):
        summary=json.loads(path.read_text())
        if not summary.get('complete_candidate_space'):
            raise RuntimeError('finish all seeds before adding the fixed uniform-W4 control')
        run=path.parent
        manifest=json.loads((run/'split_manifest.json').read_text())
        loaders={}
        for role in ('quantization','temperature','selection'):
            data=[train[by_name[name]] for name in manifest['roles'][role]]
            loaders[role]=DataLoader(TensorDataset(torch.stack([x for x,_ in data]),torch.tensor([y for _,y in data])),batch_size=256)
        model=DSCNN(num_classes=10,in_channels=3).to(device)
        checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        quantize_mixed_precision(model,{},default_bits=4)
        ranges=calibrate_activation_ranges(model,loaders['quantization'],device)
        handles=install_activation_quantization_hooks(model,bits=8,ranges=ranges)
        try:
            _,logits,labels=evaluate_logits(model,loaders['temperature'],device)
            temperature=fit_temperature(logits,labels)
            raw_selection,s_logits,s_labels=evaluate_logits(model,loaders['selection'],device)
            # Fixed baseline identity precedes test evaluation; no policy tuning.
            record=dict(seed=summary['seed'],baseline='full_network_uniform_w4',activation_bits=8,
                temperature=temperature,activation_ranges=ranges,
                weight_bytes=sum((m.weight.numel()*4+7)//8 for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))),
                selection_raw=raw_selection,selection_calibrated=classification_metrics(s_logits/temperature,s_labels))
            (run/'uniform_w4_selection.json').write_text(json.dumps(record,indent=2))
            if test_tensors is None:
                data=[test[i] for i in range(len(test))]
                test_tensors=TensorDataset(torch.stack([x for x,_ in data]),torch.tensor([y for _,y in data]))
                del data
            raw_test,t_logits,t_labels=evaluate_logits(model,DataLoader(test_tensors,batch_size=256),device)
        finally:
            for handle in handles:handle.remove()
        record.update(test_raw=raw_test,test_calibrated=classification_metrics(t_logits/temperature,t_labels))
        (run/'uniform_w4.json').write_text(json.dumps(record,indent=2))
        np.savez_compressed(run/'uniform_w4_logits.npz',selection_logits=s_logits,selection_labels=s_labels,
                            test_logits=t_logits,test_labels=t_labels)
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
