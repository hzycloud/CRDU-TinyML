"""Analyze only complete new independent static-quantization replications."""
import csv
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.review_reanalysis import objectives, interactions, additive_prediction, write_csv, freeze_decisions
from cig_mpq.metrics import softmax, expected_calibration_error, adaptive_calibration_error, brier_decomposition


def main():
    root=ROOT/'results/review_revision/cifar_static'
    summaries, diagnostics, decisions, hashes, residuals = [], [], [], {}, {}
    ece=[]
    for path in sorted(root.glob('seed*/training_summary.json')):
        summary=json.loads(path.read_text())
        if not summary.get('complete_candidate_space'):
            continue
        run=path.parent
        rows=json.loads((run/'candidates.json').read_text())
        if len(rows)!=64:
            raise ValueError('complete flag inconsistent with candidate count')
        frozen=json.loads((run/'frozen_decisions.json').read_text())
        if frozen != freeze_decisions(rows):
            raise ValueError('test-stage records do not reproduce frozen selection-only decisions')
        if hashlib.sha256((run/'frozen_decisions.json').read_bytes()).hexdigest()!=summary['frozen_decisions_sha256']:
            raise ValueError('frozen decision hash mismatch')
        for input_path in (path,run/'candidates.json',run/'split_manifest.json',run/'history.json',run/'model.pt',run/'frozen_decisions.json'):
            hashes[str(input_path.relative_to(ROOT))]=hashlib.sha256(input_path.read_bytes()).hexdigest()
        seed=summary['seed']
        reference=rows[0]
        summaries.append(dict(seed=seed,epochs=summary['epochs_run'],best_epoch=summary['best_epoch'],
            maximum_epochs_reached=summary['maximum_epochs_reached'],
            preliminary_test_exposure=summary.get('prior_partial_exploratory_test_outputs',False),
            feasible_budgets=len([r for r in frozen['budgets'] if r['eligible_masks']]),
            w8_accuracy=reference['eval_raw']['accuracy'],w8_ece=reference['eval_calibrated']['ece'],
            w8_nll=reference['eval_calibrated']['nll']))
        values={key:{r['mask']:objectives(r)[key] for r in rows} for key in objectives(reference)}
        raw=np.array(list(values['raw_nll'].values()))
        raw_inter=interactions(values['raw_nll'],6)
        residuals[seed]={}
        for key,value in values.items():
            inter=interactions(value,6)
            X=np.column_stack([np.ones(len(raw_inter)),raw_inter])
            residual=inter-X@np.linalg.lstsq(X,inter,rcond=None)[0]
            residuals[seed][key]=residual
            total=np.sum((inter-inter.mean())**2)
            diagnostics.append(dict(seed=seed,objective=key,
                preliminary_test_exposure=summary.get('prior_partial_exploratory_test_outputs',False),
                candidate_rho=float(spearmanr(raw,list(value.values())).statistic),
                interaction_r2=float(1-np.sum(residual**2)/total) if total>0 else None))
        for budget in (1.,.95,.9,.85,.8):
            eligible=[r for r in rows if r['flash_fraction']<=budget+1e-12 and
                      r['search_raw']['accuracy']>=reference['search_raw']['accuracy']-.01]
            if not eligible:
                continue
            task=min(eligible,key=lambda r:(r['search_raw']['nll'],r['mask']))
            policies={key:min(eligible,key=lambda r:(value[r['mask']],r['mask'])) for key,value in values.items()}
            policies['additive_raw_nll']=min(eligible,key=lambda r:(additive_prediction(r['mask'],values['raw_nll'],6),r['mask']))
            ece_eligible=[r for r in eligible if r['search_calibrated']['ece']<=reference['search_calibrated']['ece']]
            if ece_eligible:
                policies['ece_constrained']=min(ece_eligible,key=lambda r:(r['search_raw']['nll'],r['mask']))
            for key,row in policies.items():
                decisions.append(dict(seed=seed,budget=budget,policy=key,mask=row['mask'],
                    preliminary_test_exposure=summary.get('prior_partial_exploratory_test_outputs',False),
                    bits=''.join(map(str,row['bits'].values())),changed=int(row['mask']!=task['mask']),
                    **{f'delta_{m}':row['eval_calibrated'][m]-task['eval_calibrated'][m] for m in ('accuracy','ece','nll','brier')}))
                archive=run/f"mask{row['mask']}_logits.npz"
                hashes[str(archive.relative_to(ROOT))]=hashlib.sha256(archive.read_bytes()).hexdigest()
                with np.load(archive) as data:
                    p=softmax(data['test_logits']/row['temperature'])
                    y=data['test_labels']
                for bins in (10,15,20):
                    ece.append(dict(seed=seed,budget=budget,policy=key,bins=bins,
                        ece=expected_calibration_error(p,y,bins),adaptive_ece=adaptive_calibration_error(p,y,bins)))
        history=json.loads((run/'history.json').read_text())
        fig,axes=plt.subplots(1,2,figsize=(9,3),layout='constrained')
        epochs=[r['epoch'] for r in history]
        axes[0].plot(epochs,[r['train_loss'] for r in history],label='Train NLL')
        axes[0].plot(epochs,[r['validation_nll'] for r in history],label='Validation NLL')
        axes[0].axvline(summary['best_epoch'],color='black',linestyle=':',label='Selected checkpoint')
        axes[0].set(xlabel='Epoch',ylabel='Negative log-likelihood')
        axes[0].legend(fontsize=8)
        axes[1].plot(epochs,[r['validation_accuracy'] for r in history])
        axes[1].set(xlabel='Epoch',ylabel='Validation accuracy')
        fig.savefig(root/f'training_seed{seed}.png',dpi=200)
        plt.close(fig)
    write_csv(root/'training_summary.csv',summaries)
    write_csv(root/'risk_diagnostics.csv',diagnostics)
    write_csv(root/'budget_decisions.csv',decisions)
    write_csv(root/'ece_sensitivity.csv',ece)
    stability={}
    for key in ('ts_nll','ts_brier','mix_0.5'):
        correlations=[float(spearmanr(residuals[a][key],residuals[b][key]).statistic) for a,b in combinations(residuals,2)]
        stability[key]=dict(seed_pairs=len(correlations),median_residual_rho=float(np.median(correlations)) if correlations else None)
    policies={}
    for policy in dict.fromkeys(r['policy'] for r in decisions):
        subset=[r for r in decisions if r['policy']==policy]
        seeds=sorted({r['seed'] for r in subset})
        entry=dict(cases=len(subset),changed=sum(r['changed'] for r in subset),seed_count=len(seeds),
            seeds=seeds,coverage_by_seed={str(seed):[r['budget'] for r in subset if r['seed']==seed] for seed in seeds},
            estimand='seed-equal paired difference conditional on this policy being feasible; not common support across policies',
            prior_exposure_seeds=sorted({r['seed'] for r in subset if r['preliminary_test_exposure']}))
        for metric in ('accuracy','ece','nll','brier'):
            per_seed=[np.mean([r[f'delta_{metric}'] for r in subset if r['seed']==s]) for s in sorted({r['seed'] for r in subset})]
            rng=np.random.default_rng(20261027)
            bootstrap=np.asarray(per_seed)[rng.integers(0,len(per_seed),(10000,len(per_seed)))].mean(axis=1)
            entry[metric]=dict(mean=float(np.mean(per_seed)),
                ci95=np.quantile(bootstrap,[.025,.975]).tolist() if len(per_seed)>=2 else None,
                per_seed=dict(zip(map(str,seeds),map(float,per_seed))))
        policies[policy]=entry
    output=dict(status='independent_protocol_static_fake_quantization_not_integer',seeds=[r['seed'] for r in summaries],
        w8_accuracy_mean=float(np.mean([r['w8_accuracy'] for r in summaries])) if summaries else None,
        w8_accuracy_sd=float(np.std([r['w8_accuracy'] for r in summaries],ddof=1)) if len(summaries)>1 else None,
        stability=stability,policies=policies,source_hashes=hashes,
        seed_provenance={str(r['seed']):{'prior_partial_test_exposure':r['preliminary_test_exposure']} for r in summaries},
        limitations=['Data count, training, splits and ranges all changed; not a causal static-vs-dynamic ablation',
                     'Validation NLL checkpoint can precede minimum epoch; inspect overfitting',
                     'Five seeds still limit inference; no integer inference or external strong baseline'])
    (root/'analysis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(dict(seeds=output['seeds'],w8_accuracy_mean=output['w8_accuracy_mean'],
                         w8_accuracy_sd=output['w8_accuracy_sd'],stability=stability),indent=2))


if __name__=='__main__':
    main()
