"""Post-review reanalysis of frozen pilot records; never claims new training.

Run: .venv/Scripts/python.exe scripts/review_reanalysis.py
All allocation decisions use search_* only. eval_* is read after selection.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cig_mpq.metrics import expected_calibration_error, adaptive_calibration_error, brier_decomposition

LAMBDAS = (0., .1, .5, 1., 2.)


def objectives(row):
    raw, ts = row['search_raw'], row['search_calibrated']
    return {'raw_nll': raw['nll'], 'ts_nll': ts['nll'], 'ts_brier': ts['brier'],
            'ece': ts['ece'], **{f'mix_{w:g}': ts['nll'] + w * ts['brier'] for w in LAMBDAS}}


def additive_prediction(mask, values, n_layers):
    return values[0] + sum(values[1 << i] - values[0] for i in range(n_layers) if mask & (1 << i))


def interactions(values, n_layers):
    return np.array([values[(1 << i) | (1 << j)] - values[1 << i] - values[1 << j] + values[0]
                     for i in range(n_layers) for j in range(i + 1, n_layers)])


def freeze_decisions(rows, budgets=(1., .95, .9, .85, .8)):
    """Serialize selection-only decisions before opening held-out test data."""
    reference = next(row for row in rows if row['mask'] == 0)
    layers = len(reference['bits'])
    values = {key: {r['mask']: objectives(r)[key] for r in rows} for key in objectives(reference)}
    frozen = []
    for budget in budgets:
        eligible = [r for r in rows if r['flash_fraction'] <= budget + 1e-12 and
                    r['search_raw']['accuracy'] >= reference['search_raw']['accuracy'] - .01]
        record = {'budget': budget, 'eligible_masks': [r['mask'] for r in eligible], 'policies': {}}
        if eligible:
            for name, value in values.items():
                record['policies'][name] = min(eligible, key=lambda r: (value[r['mask']],r['mask']))['mask']
            if all((1 << i) in values['raw_nll'] for i in range(layers)):
                record['policies']['additive_raw_nll'] = min(eligible,key=lambda r: (
                    additive_prediction(r['mask'],values['raw_nll'],layers),r['mask']))['mask']
            constrained = [r for r in eligible if r['search_calibrated']['ece'] <= reference['search_calibrated']['ece']]
            record['policies']['ece_constrained'] = min(constrained,key=lambda r:(r['search_raw']['nll'],r['mask']))['mask'] if constrained else None
        frozen.append(record)
    return {'protocol': 'selection_only_no_test_fields', 'budgets': frozen}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    output = ROOT / 'results' / 'review_revision'
    output.mkdir(parents=True, exist_ok=True)
    hashes, diagnostics, decisions, basic, ece_rows, decomposition = {}, [], [], [], [], []
    for task in ('cifar', 'speech'):
        result_dir = ROOT / 'results' / f'{task}_pareto_m5'
        for path in sorted(result_dir.glob('seed*_candidates.json')):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = sorted(json.loads(path.read_text(encoding='utf-8')), key=lambda r: r['mask'])
            if task == 'speech':
                for row in rows:
                    row['search_raw'] = {'nll': row['search_raw_nll'], 'accuracy': row['search_calibrated']['accuracy']}
                    row['eval_calibrated'] = row['test_calibrated_metrics']
            seed = rows[0]['seed']
            by_mask = {r['mask']: r for r in rows}
            reference = by_mask[0]
            layers = len(reference['bits'])
            values = {name: {r['mask']: objectives(r)[name] for r in rows} for name in objectives(reference)}
            # Scale-adjusted combination uses selection-space SD only. Exploratory,
            # not tuned on test outcomes and not claimed as a new algorithm.
            nll_sd = np.std(list(values['ts_nll'].values()))
            brier_sd = np.std(list(values['ts_brier'].values()))
            scale_ratio = float(nll_sd / max(brier_sd, 1e-12))
            values['mix_sd_normalized'] = {r['mask']: r['search_calibrated']['nll'] +
                .5 * scale_ratio * r['search_calibrated']['brier'] for r in rows}
            raw_vec = np.array(list(values['raw_nll'].values()))
            raw_inter = interactions(values['raw_nll'], layers)
            for objective, candidate_values in values.items():
                vec = np.array(list(candidate_values.values()))
                inter = interactions(candidate_values, layers)
                fitted = np.column_stack([np.ones(len(raw_inter)), raw_inter])
                residual = inter - fitted @ np.linalg.lstsq(fitted, inter, rcond=None)[0]
                total = np.sum((inter - inter.mean())**2)
                diagnostics.append(dict(task=task, seed=seed, objective=objective,
                    candidate_rho=float(spearmanr(raw_vec, vec).statistic),
                    interaction_rho=float(spearmanr(raw_inter, inter).statistic),
                    interaction_r2=float(1 - np.sum(residual**2) / total) if total > 0 else None,
                    nll_to_brier_sd_ratio=scale_ratio))
            for label, mask in [('uniform_w8', 0), ('candidate_w4_fixed_w8', (1 << layers) - 1)]:
                row = by_mask[mask]
                basic.append(dict(task=task, seed=seed, baseline=label, mask=mask,
                    weight_bytes=row['weight_bytes'], **row['eval_calibrated']))
            # Include every attainable storage threshold. Existing coarse budgets
            # are retained, but repeated decisions are not independent samples.
            budgets = sorted(set([1., .95, .9, .85] + ([.8] if task == 'speech' else []) +
                [float(r['flash_fraction']) for r in rows]))
            for budget in budgets:
                eligible = [r for r in rows if r['flash_fraction'] <= budget + 1e-12 and
                    r['search_raw']['accuracy'] >= reference['search_raw']['accuracy'] - .01]
                if not eligible:
                    continue
                task_choice = min(eligible, key=lambda r: (values['raw_nll'][r['mask']], r['mask']))
                policies = {name: min(eligible, key=lambda r: (v[r['mask']], r['mask'])) for name, v in values.items()}
                policies['additive_raw_nll'] = min(eligible, key=lambda r: (
                    additive_prediction(r['mask'], values['raw_nll'], layers), r['mask']))
                # Exhaustive mean is the exact expectation of a single uniform
                # random draw from the common feasible set, no best-of-test draw.
                policies['random_feasible_expectation'] = None
                for label, chosen in policies.items():
                    test = chosen['eval_calibrated'] if chosen is not None else {
                        k: float(np.mean([r['eval_calibrated'][k] for r in eligible])) for k in ('accuracy','ece','nll','brier')}
                    decisions.append(dict(task=task, seed=seed, budget=budget, policy=label,
                        eligible_count=len(eligible), mask=chosen['mask'] if chosen else '',
                        bits=''.join(map(str, chosen['bits'].values())) if chosen else '',
                        changed=int(chosen['mask'] != task_choice['mask']) if chosen else '',
                        weight_bytes=chosen['weight_bytes'] if chosen else float(np.mean([r['weight_bytes'] for r in eligible])),
                        **{k: test[k] for k in ('accuracy','ece','nll','brier')},
                        **{f'delta_{k}': test[k] - task_choice['eval_calibrated'][k] for k in ('accuracy','ece','nll','brier')}))
        prediction_dir = result_dir / 'predictions'
        index_path = prediction_dir / 'index.json'
        hashes[str(index_path.relative_to(ROOT))] = hashlib.sha256(index_path.read_bytes()).hexdigest()
        index = json.loads(index_path.read_text(encoding='utf-8'))
        for seed, record in index.items():
            arrays = {}
            for budget, policies in record['budgets'].items():
                for policy, mask in policies.items():
                    if mask is None:
                        continue
                    if mask not in arrays:
                        path = prediction_dir / f'seed{seed}_mask{mask}.npz'
                        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                        with np.load(path, allow_pickle=False) as data:
                            p, y = data['probabilities'].astype(float), data['labels']
                        arrays[mask] = (p, y)
                    p, y = arrays[mask]
                    for bins in (10, 15, 20):
                        ece_rows.append(dict(task=task, seed=int(seed), budget=float(budget), policy=policy,
                            mask=mask, bins=bins, ece=expected_calibration_error(p, y, bins),
                            adaptive_ece=adaptive_calibration_error(p, y, bins)))
                    decomposition.append(dict(task=task, seed=int(seed), budget=float(budget), policy=policy,
                        mask=mask, **brier_decomposition(p, y, 15)))
    write_csv(output / 'risk_diagnostics.csv', diagnostics)
    write_csv(output / 'budget_decisions.csv', decisions)
    write_csv(output / 'uniform_baselines.csv', basic)
    write_csv(output / 'ece_sensitivity.csv', ece_rows)
    write_csv(output / 'brier_decomposition.csv', decomposition)
    summary = {}
    for task in ('cifar', 'speech'):
        summary[task] = {}
        for policy in dict.fromkeys(r['policy'] for r in decisions):
            coarse = [r for r in decisions if r['task'] == task and r['policy'] == policy and
                      any(abs(r['budget'] - b) < 1e-10 for b in ([1.,.95,.9,.85] + ([.8] if task == 'speech' else [])))]
            if coarse:
                seeds = sorted(set(r['seed'] for r in coarse))
                entry = dict(cases=len(coarse), changed=sum(r['changed'] for r in coarse) if policy != 'random_feasible_expectation' else None)
                for metric in ('accuracy', 'ece', 'nll', 'brier'):
                    per_seed = np.array([np.mean([r[f'delta_{metric}'] for r in coarse if r['seed'] == s]) for s in seeds])
                    rng = np.random.default_rng(20261027)
                    bootstrap = per_seed[rng.integers(0, len(seeds), (10000, len(seeds)))].mean(axis=1)
                    entry[metric] = dict(mean=float(per_seed.mean()), ci95=np.quantile(bootstrap, [.025,.975]).tolist(),
                                         per_seed=dict(zip(map(str,seeds),per_seed.tolist())))
                summary[task][policy] = entry
    sensitivity_summary = {}
    for task in ('cifar', 'speech'):
        sensitivity_summary[task] = {}
        for policy in ('calibration_only', 'calibration_constrained'):
            sensitivity_summary[task][policy] = {}
            for metric in ('ece', 'adaptive_ece'):
                for bins in (10,15,20):
                    paired = []
                    for r in ece_rows:
                        if r['task'] != task or r['policy'] != policy or r['bins'] != bins:
                            continue
                        base = next(x for x in ece_rows if x['task'] == task and x['seed'] == r['seed'] and
                                    x['budget'] == r['budget'] and x['bins'] == bins and x['policy'] == 'task_only')
                        paired.append((r['seed'], r[metric] - base[metric]))
                    means = [np.mean([v for s,v in paired if s == seed]) for seed in sorted({s for s,_ in paired})]
                    sensitivity_summary[task][policy][f'{metric}_{bins}'] = float(np.mean(means))
    # All-candidate held-out clouds are descriptive only; no test-optimal policy
    # is selected from these plots. Each seed remains a separate experimental unit.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in ('cifar', 'speech'):
        fig, axes = plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
        for path in sorted((ROOT/'results'/f'{task}_pareto_m5').glob('seed*_candidates.json')):
            rows=json.loads(path.read_text(encoding='utf-8'))
            metric_key='eval_calibrated' if task=='cifar' else 'test_calibrated_metrics'
            for axis, metric, factor in zip(axes,('accuracy','nll','ece'),(100,1,100)):
                axis.scatter([r['weight_bytes'] for r in rows], [factor*r[metric_key][metric] for r in rows],
                             s=8,alpha=.45,label=f"Seed {rows[0]['seed']}")
                axis.set(xlabel='Weight proxy (bytes)',ylabel=metric+(' (%)' if factor==100 else ''))
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{task}: historical candidate trade-offs (not deployment measurements)')
        fig.savefig(output/f'{task}_memory_tradeoffs.png',dpi=200)
        plt.close(fig)
    report = dict(status='exploratory_reanalysis_of_short_training_dynamic_fake_quantization',
        source_hashes=hashes, summary=summary, ece_sensitivity_seed_equal=sensitivity_summary,
        boundaries=['No new independent training replication in these tables',
                    'Random baseline is an exact uniform-draw expectation over accuracy-filtered candidates',
                    'Candidate-W4 leaves noncandidate weights at W8; not a fully uniform W4 network',
                    'All methods still use exhaustive measured selection accuracy; no search speedup',
                    'Adaptive ECE ties use stable order; binning remainder retained in Brier decomposition',
                    'Five seed clusters only; intervals exploratory and not multiplicity adjusted'])
    (output / 'reanalysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(output), decisions=len(decisions), inputs=len(hashes),
                         ece_sensitivity=sensitivity_summary), indent=2))


if __name__ == '__main__':
    main()
