"""Small, auditable scientific calculations for the second revision."""
import numpy as np


def feasible_masks(rows, budget, tolerance=.01):
    reference=next(r for r in rows if r['mask']==0)
    return [r['mask'] for r in rows if r['flash_fraction']<=budget+1e-12 and
            r['search_raw']['accuracy']>=reference['search_raw']['accuracy']-tolerance-1e-12]


def surface_prediction(mask, values, layers, pairwise=True):
    result=values[0]+sum(values[1<<i]-values[0] for i in range(layers) if mask&(1<<i))
    if pairwise:
        result+=sum(values[(1<<i)|(1<<j)]-values[1<<i]-values[1<<j]+values[0]
                    for i in range(layers) for j in range(i+1,layers)
                    if mask&(1<<i) and mask&(1<<j))
    return float(result)


def ece_from_indices(confidence, correct, indices, bins=15):
    confidence=np.asarray(confidence)[:,indices]
    correct=np.asarray(correct)[:,indices]
    ids=np.minimum((confidence*bins).astype(int),bins-1)
    # Sum unnormalized signed errors inside each bin then take the absolute
    # value; empty bins contribute zero. Matches the existing top-label ECE.
    return np.array([np.abs(np.bincount(b,weights=c-a,minlength=bins)).sum()/len(indices)
                     for b,c,a in zip(ids,correct,confidence)])


def paired_summary(records, policy, metric):
    rows=[r for r in records if r['policy']==policy]
    seeds=sorted({r['seed'] for r in rows})
    means=np.array([np.mean([r['delta_'+metric] for r in rows if r['seed']==seed]) for seed in seeds])
    rng=np.random.default_rng(20261027)
    ci=None
    if len(seeds)>1:
        resamples=means[rng.integers(0,len(seeds),(10000,len(seeds)))].mean(axis=1)
        ci=np.quantile(resamples,[.025,.975]).tolist()
    return {'mean':float(means.mean()) if len(means) else None,'ci95':ci,'seed_count':len(seeds),
            'cases':len(rows),'per_seed':dict(zip(map(str,seeds),means.tolist())),
            'estimand':'seed-equal paired difference conditional on policy feasibility'}
