"""Unified static-range analysis; no training, test-optimal allocation or tuning.

Run --task cifar or --task speech. Input records and logits remain unchanged.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cig_mpq.metrics import softmax,expected_calibration_error,adaptive_calibration_error,brier_decomposition
from cig_mpq.round2_analysis import feasible_masks,surface_prediction,ece_from_indices,paired_summary
from scripts.review_reanalysis import objectives,write_csv,interactions

BUDGETS=(1.,.95,.9,.85,.8)
TOLERANCES=(0.,.005,.01,.02,.03)
GATE={'selection_frequency':.8,'ece_gain':.001,'max_accuracy_loss':.005,'max_nll_increase':.005,
      'repeats':100,'seed':20261027,'status':'exploratory_post_search_screen_not_general_utility_predictor'}


def safe_rho(a,b):
    return float(spearmanr(a,b).statistic) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None


def masks_for_metric(eligible,values):
    return min(eligible,key=lambda mask:(values[mask],mask))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=('cifar','speech'),required=True)
    parser.add_argument('--input-dir',type=Path)
    parser.add_argument('--repeats',type=int,default=100)
    args=parser.parse_args()
    inputs=args.input_dir or (ROOT/'results/review_revision/cifar_static' if args.task=='cifar' else ROOT/'results/round2/speech')
    out=ROOT/'results/round2'/f'analysis_{args.task}'
    out.mkdir(parents=True,exist_ok=True)
    gate_path=ROOT/'results/round2/gate_protocol.json'
    if gate_path.exists():
        if json.loads(gate_path.read_text()) != GATE:
            raise ValueError('existing exploratory screen parameters differ')
    else:
        gate_path.write_text(json.dumps(GATE,indent=2),encoding='utf-8')
    result={name:[] for name in ('coverage','decisions','diagnostics','surrogates','stability','ece_sensitivity','brier_components','uniform','configuration_metrics')}
    hashes={}
    residual_vectors={}
    baseline_path=ROOT/('results/round2/baselines/allocation_scores.json' if args.task=='cifar' else 'results/round2/speech_baseline/allocation_scores.json')
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    if baseline:hashes[str(baseline_path.relative_to(ROOT))]=hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    for path in sorted(inputs.glob('seed*/candidates.json')):
        run=path.parent
        rows=sorted(json.loads(path.read_text()),key=lambda r:r['mask'])
        layers=len(rows[0]['bits'])
        if len(rows)!=(1<<layers) or any('eval_calibrated' not in r for r in rows):
            continue
        seed=rows[0]['seed']
        reference=rows[0]
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        archive_names=[run/f'mask{r["mask"]}_logits.npz' for r in rows]
        selected_logits=[];test_logits=[];y_sel=y_test=None
        group_ids=None
        for archive in archive_names:
            hashes[str(archive.relative_to(ROOT))]=hashlib.sha256(archive.read_bytes()).hexdigest()
            with np.load(archive,allow_pickle=False) as data:
                if y_sel is not None:
                    assert np.array_equal(y_sel,data['selection_labels']) and np.array_equal(y_test,data['test_labels'])
                y_sel=data['selection_labels'];y_test=data['test_labels']
                selected_logits.append(data['selection_logits'].astype(float))
                test_logits.append(data['test_logits'].astype(float))
                if 'selection_group_ids' in data.files:group_ids=data['selection_group_ids'].astype(str)
        selected_logits=np.stack(selected_logits)
        test_logits=np.stack(test_logits)
        temperatures=np.array([r['temperature'] for r in rows])
        calibrated_selection=selected_logits/temperatures[:,None,None]
        calibrated_test=test_logits/temperatures[:,None,None]
        p_sel=np.stack([softmax(z) for z in calibrated_selection])
        p_test=np.stack([softmax(z) for z in calibrated_test])
        correct=p_sel.argmax(axis=2)==y_sel
        confidence=p_sel.max(axis=2)
        raw_nll=logsumexp(selected_logits,axis=2)-np.take_along_axis(selected_logits,y_sel[None,:,None],axis=2)[:,:,0]
        ts_nll=logsumexp(calibrated_selection,axis=2)-np.take_along_axis(calibrated_selection,y_sel[None,:,None],axis=2)[:,:,0]
        ts_brier=np.sum((p_sel-np.eye(p_sel.shape[2])[y_sel][None,:,:])**2,axis=2)
        for row in rows:
            mask=row['mask']
            result['configuration_metrics'].append(dict(seed=seed,mask=mask,weight_bytes=row['weight_bytes'],
                flash_fraction=row['flash_fraction'],selection_accuracy=row['search_raw']['accuracy'],
                raw_nll=row['search_raw']['nll'],ts_nll=row['search_calibrated']['nll'],
                ts_brier=row['search_calibrated']['brier'],selection_ece=row['search_calibrated']['ece'],
                **{'test_'+k:row['eval_calibrated'][k] for k in ('accuracy','ece','nll','brier')}))
        values={key:{r['mask']:objectives(r)[key] for r in rows} for key in objectives(reference)}
        ratio=np.std(list(values['ts_nll'].values()))/max(np.std(list(values['ts_brier'].values())),1e-12)
        values['mix_scaled']={r['mask']:values['ts_nll'][r['mask']]+.5*ratio*values['ts_brier'][r['mask']] for r in rows}
        main_scores={key:v for key,v in values.items() if key in ('raw_nll','ts_nll','ts_brier','ece','mix_0.5')}
        main_scores['additive_task']={r['mask']:surface_prediction(r['mask'],values['raw_nll'],layers,False) for r in rows}
        for name,key in (('pair_task','raw_nll'),('pair_cal','mix_0.5')):
            main_scores[name]={r['mask']:surface_prediction(r['mask'],values[key],layers,True) for r in rows}
        if baseline and str(seed) in baseline['seeds']:
            main_scores['published_control']={int(k):float(v) for k,v in baseline['seeds'][str(seed)]['scores'].items()}
        residual_vectors[seed]={}
        raw_inter=interactions(values['raw_nll'],layers)
        for key,v in values.items():
            inter=interactions(v,layers)
            X=np.column_stack([np.ones(len(raw_inter)),raw_inter])
            e=inter-X@np.linalg.lstsq(X,inter,rcond=None)[0]
            residual_vectors[seed][key]=e
            total=np.sum((inter-inter.mean())**2)
            result['diagnostics'].append(dict(seed=seed,objective=key,
                candidate_rho=safe_rho(list(values['raw_nll'].values()),list(v.values())),
                interaction_rho=safe_rho(raw_inter,inter),
                interaction_r2=float(1-np.sum(e**2)/total) if total>0 else None))
            heldout=[m for m in v if m.bit_count()>=3]
            for pair in (False,True):
                errors=[(surface_prediction(m,v,layers,pair)-v[m])**2 for m in heldout]
                result['surrogates'].append(dict(seed=seed,objective=key,model='pairwise' if pair else 'additive',
                    heldout_configurations=len(heldout),rmse=float(np.sqrt(np.mean(errors)))))
        for label,mask in [('W8',0),('candidate_W4',(1<<layers)-1)]:
            r=rows[mask]
            result['uniform'].append(dict(seed=seed,baseline=label,weight_bytes=r['weight_bytes'],**r['eval_calibrated']))
        full_path=run/'uniform_w4.json'
        if full_path.exists():
            c=json.loads(full_path.read_text())
            result['uniform'].append(dict(seed=seed,baseline='full_W4',weight_bytes=c['weight_bytes'],**c['test_calibrated']))
        # Fixed 50% complementary selection splits. For speech, split word
        # speakers as groups; silence is within-one-source resampled separately
        # and cannot provide independent background-source uncertainty.
        rng=np.random.default_rng(20261027+seed)
        splits=[]
        for repeat in range(args.repeats):
            if group_ids is None:
                perm=rng.permutation(len(y_sel));first=perm[:len(perm)//2];second=perm[len(perm)//2:]
            else:
                silence=np.flatnonzero(y_sel==10)
                words=np.flatnonzero(y_sel!=10)
                groups=np.unique(group_ids[words]);rng.shuffle(groups)
                first_word=words[np.isin(group_ids[words],groups[:len(groups)//2])]
                second_word=words[np.isin(group_ids[words],groups[len(groups)//2:])]
                rng.shuffle(silence);first=np.r_[first_word,silence[:len(silence)//2]];second=np.r_[second_word,silence[len(silence)//2:]]
            splits.append((first,second))
        repeated=[]
        for first,second in splits:
            repeated.append(dict(a=first,b=second,ece_a=ece_from_indices(confidence,correct,first),
                ece_b=ece_from_indices(confidence,correct,second),accuracy_a=correct[:,first].mean(axis=1),
                accuracy_b=correct[:,second].mean(axis=1),raw_a=raw_nll[:,first].mean(axis=1),
                nll_b=ts_nll[:,second].mean(axis=1)))
        for tolerance in TOLERANCES:
            for budget in BUDGETS:
                budget_masks=[r['mask'] for r in rows if r['flash_fraction']<=budget+1e-12]
                eligible=feasible_masks(rows,budget,tolerance)
                task=masks_for_metric(eligible,values['raw_nll']) if eligible else None
                chosen={name:masks_for_metric(eligible,v) for name,v in values.items()} if eligible else {}
                category='empty' if not eligible else 'singleton' if len(eligible)==1 else 'multiple'
                raw_order=sorted(eligible,key=lambda m:(values['raw_nll'][m],m))
                cal_order=sorted(eligible,key=lambda m:(values['mix_0.5'][m],m))
                result['coverage'].append(dict(seed=seed,budget=budget,tolerance=tolerance,
                    budget_count=len(budget_masks),feasible_count=len(eligible),category=category,
                    task_mask=task,cal_mask=chosen.get('mix_0.5'),
                    feasible_rho=safe_rho([values['raw_nll'][m] for m in eligible],[values['mix_0.5'][m] for m in eligible]),
                    raw_gap=values['raw_nll'][raw_order[1]]-values['raw_nll'][raw_order[0]] if len(eligible)>1 else None,
                    cal_gap=values['mix_0.5'][cal_order[1]]-values['mix_0.5'][cal_order[0]] if len(eligible)>1 else None))
                if not eligible:continue
                policies={name:masks_for_metric(eligible,v) for name,v in main_scores.items()}
                cap=[m for m in eligible if values['ece'][m]<=values['ece'][0]]
                if cap:policies['ece_cap']=masks_for_metric(cap,values['raw_nll'])
                # Full lambda sweep remains a sensitivity analysis, never
                # selecting a lambda according to the final test performance.
                for name in values:
                    if name.startswith('mix_'):policies[name]=masks_for_metric(eligible,values[name])
                gate_frequency=None;gate_validation_success=None
                if tolerance==.01:
                    stable_choices=[];validations=[]
                    for mode in ('fixed_set','recomputed_set'):
                        picked=[];success=[];gains=[];selection_ece=[];complement_ece=[]
                        for replicate in repeated:
                            candidates=eligible if mode=='fixed_set' else [m for m in budget_masks
                                if replicate['accuracy_a'][m]>=replicate['accuracy_a'][0]-tolerance-1e-12]
                            if not candidates:continue
                            t=min(candidates,key=lambda m:(replicate['raw_a'][m],m))
                            c=min(candidates,key=lambda m:(replicate['ece_a'][m],m))
                            picked.append(c)
                            gain=replicate['ece_b'][t]-replicate['ece_b'][c]
                            gains.append(float(gain));selection_ece.append(float(replicate['ece_a'][c]));complement_ece.append(float(replicate['ece_b'][c]))
                            success.append(gain>=GATE['ece_gain'] and
                                replicate['accuracy_b'][c]-replicate['accuracy_b'][t]>=-GATE['max_accuracy_loss'] and
                                replicate['nll_b'][c]-replicate['nll_b'][t]<=GATE['max_nll_increase'])
                        counts=Counter(picked)
                        modal,frequency=counts.most_common(1)[0] if counts else (None,0)
                        result['stability'].append(dict(seed=seed,budget=budget,mode=mode,valid_repeats=len(picked),
                            modal_mask=modal,modal_frequency=frequency/len(picked) if picked else None,
                            full_ece_mask=policies['ece'],full_choice_frequency=counts.get(policies['ece'],0)/len(picked) if picked else None,
                            validation_success=float(np.mean(success)) if success else None,
                            complementary_ece_gain=float(np.mean(gains)) if gains else None,
                            chosen_selection_ece=float(np.mean(selection_ece)) if selection_ece else None,
                            chosen_complement_ece=float(np.mean(complement_ece)) if complement_ece else None,
                            reference_ece_half_sd=float(np.std([v['ece_a'][0] for v in repeated],ddof=1)),
                            full_choice_ece_half_sd=float(np.std([v['ece_a'][policies['ece']] for v in repeated],ddof=1)),
                            selected_mask_counts=json.dumps(dict(counts)),sampling='speaker-group words and within-source silence' if group_ids is not None else 'complementary sample halves'))
                        if mode=='fixed_set':
                            gate_frequency=counts.get(policies['ece'],0)/len(picked) if picked else 0
                            gate_validation_success=float(np.mean(success)) if success else 0
                    gate_pass=len(eligible)>1 and gate_frequency>=GATE['selection_frequency'] and gate_validation_success>=.8
                    policies['stability_gate']=policies['ece'] if gate_pass else task
                for policy,mask in policies.items():
                    r=rows[mask];test=r['eval_calibrated'];base=rows[task]['eval_calibrated']
                    result['decisions'].append(dict(seed=seed,budget=budget,tolerance=tolerance,policy=policy,mask=mask,
                        bits=''.join(map(str,r['bits'].values())),feasible_count=len(eligible),changed=int(mask!=task),
                        weight_bytes=r['weight_bytes'],**{m:test[m] for m in ('accuracy','ece','nll','brier')},
                        **{'delta_'+m:test[m]-base[m] for m in ('accuracy','ece','nll','brier')}))
                    if tolerance==.01 and policy in ('raw_nll','mix_0.5','ece','ece_cap','stability_gate'):
                        for bins in (10,15,20):
                            result['ece_sensitivity'].append(dict(seed=seed,budget=budget,policy=policy,bins=bins,
                                ece=expected_calibration_error(p_test[mask],y_test,bins),
                                adaptive_ece=adaptive_calibration_error(p_test[mask],y_test,bins)))
                        result['brier_components'].append(dict(seed=seed,budget=budget,policy=policy,
                            **brier_decomposition(p_test[mask],y_test)))
                random_test={m:float(np.mean([rows[mask]['eval_calibrated'][m] for mask in eligible])) for m in ('accuracy','ece','nll','brier')}
                result['decisions'].append(dict(seed=seed,budget=budget,tolerance=tolerance,policy='random_expectation',mask=-1,bits='',
                    feasible_count=len(eligible),changed=-1,weight_bytes=float(np.mean([rows[m]['weight_bytes'] for m in eligible])),
                    **random_test,**{'delta_'+m:random_test[m]-rows[task]['eval_calibrated'][m] for m in random_test}))
        print(json.dumps(dict(task=args.task,seed=seed,candidates=len(rows),repeats=args.repeats)),flush=True)
        del p_sel,p_test,selected_logits,test_logits
    for name,records in result.items():write_csv(out/(name+'.csv'),records)
    primary=[r for r in result['decisions'] if r['tolerance']==.01]
    summary={}
    for policy in dict.fromkeys(r['policy'] for r in primary):
        r=[x for x in primary if x['policy']==policy]
        summary[policy]={'cases':len(r),'changed':sum(x['changed'] for x in r) if policy!='random_expectation' else None,
            'metrics':{m:paired_summary(primary,policy,m) for m in ('accuracy','ece','nll','brier')},
            'coverage':{str(s):[x['budget'] for x in r if x['seed']==s] for s in sorted({x['seed'] for x in r})}}
    residual_stability={}
    for key in ('ts_nll','ts_brier','mix_0.5','ece'):
        scores=[safe_rho(residual_vectors[a][key],residual_vectors[b][key]) for a,b in combinations(residual_vectors,2)]
        scores=[x for x in scores if x is not None]
        residual_stability[key]=float(np.median(scores)) if scores else None
    cov=[r for r in result['coverage'] if r['tolerance']==.01]
    report={'task':args.task,'seeds':sorted(residual_vectors),'primary_tolerance':.01,'budgets':BUDGETS,
        'coverage_counts':dict(Counter(r['category'] for r in cov)),
        'residual_stability':residual_stability,'policy_summary':summary,'source_hashes':hashes,
        'gate':GATE,'limitations':['Exploratory analysis; fixed CIFAR tests already observed previously',
        'Five training seeds do not equal five independent settings','Seed-cluster percentile intervals unadjusted',
        'No integer inference or MCU resource claims; random expectation requires exhaustive measured feasibility',
        'Stability gate uses all candidate selection data and is a post-search screen; no search saving claimed']}
    (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
