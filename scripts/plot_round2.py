"""Publication figures drawn only from executed round-two CSV/JSON records."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/round2/figures'
TASKS=('cifar','speech')
COLORS={'cifar':'#235789','speech':'#A14A28'}
LABELS={'raw_nll':'Task','mix_0.5':'Cal','ece':'ECE-min','ece_cap':'ECE-cap',
        'additive_task':'Additive','pair_task':'Pair-task','pair_cal':'Pair-cal',
        'published_control':'CLADO control','random_expectation':'Random mean','stability_gate':'Stability screen'}


def read(task,name):
    return list(csv.DictReader((ROOT/'results/round2'/f'analysis_{task}'/(name+'.csv')).open(encoding='utf-8-sig')))


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.2})
    fig,ax=plt.subplots(figsize=(9,3.1));ax.axis('off')
    labels=['Training\nFit network','Validation\nChoose checkpoint','Quantization\nFreeze A8 ranges',
            'Temperature\nFit scalar T','Selection\nRank and freeze','Test\nEvaluate decisions']
    for i,label in enumerate(labels):
        x=.01+i*.166
        box=FancyBboxPatch((x,.40),.15,.37,boxstyle='round,pad=0.008',edgecolor=COLORS['cifar'],facecolor='#EAF0F6',transform=ax.transAxes)
        ax.add_patch(box);ax.text(x+.075,.585,label,ha='center',va='center',transform=ax.transAxes,fontsize=8.5)
        if i<5:ax.annotate('',xy=(x+.163,.58),xytext=(x+.151,.58),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#4B5563'})
    ax.text(.5,.23,'Distinct data roles; speaker and background sources cannot cross speech roles',ha='center',transform=ax.transAxes)
    ax.text(.5,.08,'Risk ranking  →  feasible decision opportunity  →  held-out utility and selection stability',ha='center',transform=ax.transAxes,fontweight='bold')
    save(fig,'fig1_protocol')
    fig,axes=plt.subplots(2,2,figsize=(9,5.6),layout='constrained')
    for row,task in enumerate(TASKS):
        base=ROOT/'results/review_revision/cifar_static' if task=='cifar' else ROOT/'results/round2/speech'
        for i,path in enumerate(sorted(base.glob('seed*/history.json'))):
            history=json.loads(path.read_text());summary=json.loads((path.parent/'training_summary.json').read_text())
            epochs=[h['epoch'] for h in history]
            train=[h['train_loss'] for h in history]
            validation=[h.get('validation_nll',h.get('validation_metrics',{}).get('nll')) for h in history]
            accuracy=[h.get('validation_accuracy',h.get('validation_metrics',{}).get('accuracy')) for h in history]
            color=plt.cm.tab10(i)
            axes[row,0].plot(epochs,validation,color=color,label=f"Seed {summary['seed']}")
            axes[row,0].plot(epochs,train,color=color,alpha=.35,linestyle='--')
            selected=summary['best_epoch'];axes[row,0].plot([selected],[validation[selected-1]],'o',color=color,ms=4)
            axes[row,1].plot(epochs,accuracy,color=color)
        axes[row,0].set(title=task.upper()+' loss',xlabel='Epoch',ylabel='NLL (solid: validation; dashed: train)')
        axes[row,1].set(title=task.upper()+' validation accuracy',xlabel='Epoch',ylabel='Accuracy')
    axes[0,0].legend(ncol=2,fontsize=7)
    save(fig,'fig2_training')
    fig,axes=plt.subplots(2,2,figsize=(9,5.7),layout='constrained')
    for row,task in enumerate(TASKS):
        cov=read(task,'coverage');main=[r for r in cov if abs(float(r['tolerance'])-.01)<1e-9]
        seeds=sorted({int(r['seed']) for r in main});budgets=[1.,.95,.9,.85,.8]
        values=np.array([[int(next(r['feasible_count'] for r in main if int(r['seed'])==seed and float(r['budget'])==budget)) for budget in budgets] for seed in seeds])
        axes[row,0].imshow(values,cmap='Blues',aspect='auto',vmin=0)
        for i in range(len(seeds)):
            for j in range(5):axes[row,0].text(j,i,str(values[i,j]),ha='center',va='center',color='black')
        axes[row,0].set(xticks=range(5),xticklabels=budgets,yticks=range(len(seeds)),yticklabels=seeds,xlabel='Storage budget fraction',ylabel='Training seed',title=task.upper()+' feasible candidates')
        for budget in budgets:
            tolerances=sorted({float(r['tolerance']) for r in cov})
            means=[np.mean([int(r['feasible_count']) for r in cov if float(r['budget'])==budget and float(r['tolerance'])==tol]) for tol in tolerances]
            axes[row,1].plot(np.array(tolerances)*100,means,'o-',label=f'{budget:.2f}')
        axes[row,1].set(xlabel='Accuracy tolerance (percentage points)',ylabel='Mean feasible candidate count',title=task.upper()+' constraint sensitivity')
        axes[row,1].legend(title='Budget',fontsize=7,ncol=2)
    save(fig,'fig3_feasibility')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    objectives=('ts_nll','ts_brier','mix_0.5','ece')
    for ax,task in zip(axes,TASKS):
        records=read(task,'diagnostics')
        for i,obj in enumerate(objectives):
            vals=[float(r['candidate_rho']) for r in records if r['objective']==obj]
            ax.scatter(np.full(len(vals),i),vals,s=30,color=COLORS[task],alpha=.7)
            ax.plot([i-.2,i+.2],[np.median(vals)]*2,color='black',lw=2)
        ax.set(xticks=range(4),xticklabels=['TS-NLL','TS-Brier','Combined','ECE'],ylabel='Spearman correlation with raw NLL',ylim=(-.3,1.05),title=task.upper())
    save(fig,'fig4_rankings')
    fig,axes=plt.subplots(2,3,figsize=(10,5.4),layout='constrained')
    policies=['mix_0.5','ece','ece_cap','additive_task','pair_task','published_control','stability_gate']
    for row,task in enumerate(TASKS):
        report=json.loads((ROOT/'results/round2'/f'analysis_{task}'/'analysis.json').read_text())
        present=[p for p in policies if p in report['policy_summary']]
        for col,metric in enumerate(('accuracy','ece','nll')):
            for i,policy in enumerate(present):
                info=report['policy_summary'][policy]['metrics'][metric];factor=100 if metric in ('accuracy','ece') else 1
                mean=info['mean']*factor;ci=info['ci95']
                if ci:axes[row,col].plot(np.array(ci)*factor,[i,i],color=COLORS[task],lw=1)
                axes[row,col].plot(mean,i,'o',color=COLORS[task],ms=4)
            axes[row,col].axvline(0,color='gray',linestyle='--',lw=.8)
            axes[row,col].set(yticks=range(len(present)),yticklabels=[LABELS[p] for p in present] if col==0 else [],
                xlabel='Difference from Task'+(' (pp)' if factor==100 else ''),title=task.upper()+' '+metric)
    save(fig,'fig5_paired_effects')
    fig,axes=plt.subplots(2,3,figsize=(9,5.3),layout='constrained')
    for row,task in enumerate(TASKS):
        records=read(task,'surrogates')
        for col,(objective,label) in enumerate((('raw_nll','Task risk'),('mix_0.5','Combined risk'),('ece','ECE risk'))):
            ax=axes[row,col]
            seeds=sorted({r['seed'] for r in records})
            for seed in seeds:
                vals=[float(next(r['rmse'] for r in records if r['seed']==seed and r['objective']==objective and r['model']==m)) for m in ('additive','pairwise')]
                ax.plot([0,1],vals,'o-',alpha=.6,color=COLORS[task])
            ax.set(xticks=[0,1],xticklabels=['Additive','Pairwise'],ylabel='Held-out RMSE' if col==0 else '',title=task.upper()+' '+label)
    save(fig,'fig6_surrogates')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for ax,task in zip(axes,TASKS):
        records=read(task,'stability')
        for mode,marker in (('fixed_set','o'),('recomputed_set','x')):
            selected=[r for r in records if r['mode']==mode and r['modal_frequency']]
            ax.scatter([float(r['modal_frequency']) for r in selected],[100*float(r['complementary_ece_gain']) for r in selected],marker=marker,s=34,label=mode.replace('_',' '),alpha=.75)
        ax.axhline(0,color='gray',linestyle='--');ax.axvline(.8,color='gray',linestyle=':')
        ax.set(xlabel='Most frequent configuration selection rate',ylabel='Complementary-half ECE gain (pp)',title=task.upper(),xlim=(0,1.03));ax.legend(fontsize=7)
    save(fig,'fig7_selection_stability')
    (OUT/'provenance.json').write_text(json.dumps({'script':str(Path(__file__).relative_to(ROOT)),
        'input_directories':['results/round2/analysis_cifar','results/round2/analysis_speech'],
        'note':'All numerical plots use executed analysis records; fig1 is conceptual.'},indent=2))
    print('Wrote seven figures as PNG and PDF')


if __name__=='__main__':main()
