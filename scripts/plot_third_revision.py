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
OUT=ROOT/'results/round2/figures_v3'
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
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':12.5,
        'axes.labelsize':12,'xtick.labelsize':11.5,'ytick.labelsize':11.5,
        'legend.fontsize':11.5,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.2})
    fig,ax=plt.subplots(figsize=(10,3.2));ax.axis('off')
    labels=['Training\nFit network','Validation\nChoose\ncheckpoint','Quantization\nFreeze A8\nranges',
            'Temperature\nFit T','Selection\nRank and\nfreeze','Test\nEvaluate\ndecisions']
    for i,label in enumerate(labels):
        x=.01+i*.166
        box=FancyBboxPatch((x,.40),.15,.37,boxstyle='round,pad=0.008',edgecolor=COLORS['cifar'],facecolor='#EAF0F6',transform=ax.transAxes)
        ax.add_patch(box);ax.text(x+.075,.585,label,ha='center',va='center',transform=ax.transAxes,fontsize=10.5)
        if i<5:ax.annotate('',xy=(x+.163,.58),xytext=(x+.151,.58),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':'#4B5563'})
    ax.text(.5,.23,'Distinct data roles; speaker and background sources cannot cross speech roles',ha='center',transform=ax.transAxes)
    ax.text(.5,.08,'Risk ranking  →  feasible decision opportunity  →  held-out utility and selection stability',ha='center',transform=ax.transAxes,fontweight='bold')
    save(fig,'fig1_protocol')
    fig,axes=plt.subplots(2,2,figsize=(7.4,5.0),layout='constrained')
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
        axes[row,0].set(title=task.upper()+' loss',xlabel='Epoch',ylabel='NLL')
        axes[row,1].set(title=task.upper()+' validation accuracy',xlabel='Epoch',ylabel='Accuracy')
    axes[0,0].legend(ncol=2,fontsize=11.5,frameon=True)
    save(fig,'fig2_training')
    fig,axes=plt.subplots(2,2,figsize=(7.4,4.2),layout='constrained')
    for row,task in enumerate(TASKS):
        cov=read(task,'coverage');main=[r for r in cov if abs(float(r['tolerance'])-.01)<1e-9]
        seeds=sorted({int(r['seed']) for r in main});budgets=[1.,.95,.9,.85,.8]
        values=np.array([[int(next(r['feasible_count'] for r in main if int(r['seed'])==seed and float(r['budget'])==budget)) for budget in budgets] for seed in seeds])
        axes[row,0].imshow(values,cmap='Blues',aspect='auto',vmin=0)
        for i in range(len(seeds)):
            for j in range(5):
                color='white' if values[i,j] >= .55*values.max() else '#111111'
                axes[row,0].text(j,i,str(values[i,j]),ha='center',va='center',color=color,fontweight='bold' if color=='white' else 'normal')
        axes[row,0].set(xticks=range(5),xticklabels=budgets,yticks=range(len(seeds)),yticklabels=seeds,xlabel='Storage budget',ylabel='Training seed',title=task.upper()+' feasible candidates')
        for budget in budgets:
            tolerances=sorted({float(r['tolerance']) for r in cov})
            means=[np.mean([int(r['feasible_count']) for r in cov if float(r['budget'])==budget and float(r['tolerance'])==tol]) for tol in tolerances]
            axes[row,1].plot(np.array(tolerances)*100,means,'o-',label=f'{budget:.2f}')
        axes[row,1].set(xlabel='Accuracy tolerance (pp)',ylabel='Mean candidates',title=task.upper()+' constraint sensitivity')
        axes[row,1].legend(title='Budget',fontsize=11,ncol=2)
    save(fig,'fig3_feasibility')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.6),layout='constrained')
    objectives=('ts_nll','ts_brier','mix_0.5','ece')
    for ax,task in zip(axes,TASKS):
        records=read(task,'diagnostics')
        for i,obj in enumerate(objectives):
            vals=[float(r['candidate_rho']) for r in records if r['objective']==obj]
            offsets=np.linspace(-.12,.12,len(vals))
            ax.scatter(i+offsets,vals,s=34,color=COLORS[task],alpha=.75)
            ax.plot([i-.18,i+.18],[np.median(vals)]*2,color='black',lw=2.4)
        ax.set(xticks=range(4),xticklabels=['TS-NLL','TS-Brier','Cal','ECE-min'],ylabel='Spearman correlation with raw NLL',ylim=(-.3,1.05),title=task.upper())
    save(fig,'fig4_rankings')
    fig,axes=plt.subplots(2,3,figsize=(7.6,5.4),layout='constrained')
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
                xlabel='Difference from Task'+(' (pp)' if factor==100 else ''),title=task.upper()+' '+('TS-NLL' if metric=='nll' else metric))
    save(fig,'fig5_paired_effects')
    fig,axes=plt.subplots(2,3,figsize=(7.6,5.0),layout='constrained')
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
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.6),layout='constrained')
    for ax,task in zip(axes,TASKS):
        records=read(task,'stability')
        coverage={(r['seed'],r['budget']):r['category'] for r in read(task,'coverage') if abs(float(r['tolerance'])-.01)<1e-9}
        for mode,marker in (('fixed_set','o'),('recomputed_set','s')):
            selected=[r for r in records if r['mode']==mode and r['full_choice_frequency'] and r['validation_success'] is not None]
            edgecolors=['none' if coverage.get((r['seed'],r['budget']))=='multiple' else 'black' for r in selected]
            linewidths=[0 if coverage.get((r['seed'],r['budget']))=='multiple' else .9 for r in selected]
            ax.scatter([float(r['full_choice_frequency']) for r in selected],[float(r['validation_success']) for r in selected],marker=marker,s=50,label=mode.replace('_',' '),alpha=.78,edgecolors=edgecolors,linewidths=linewidths)
        ax.axhline(.8,color='gray',linestyle='--',lw=1);ax.axvline(.8,color='gray',linestyle=':',lw=1)
        ax.text(.805,.025,'0.8',color='#666666',fontsize=10.5,ha='left',va='bottom')
        ax.set(xlabel='Final ECE candidate reselection frequency',ylabel='Half-split utility success rate',title=task.upper(),xlim=(-.03,1.03),ylim=(-.03,1.03));ax.legend(fontsize=11)
    save(fig,'fig7_selection_stability')
    (OUT/'provenance.json').write_text(json.dumps({'script':str(Path(__file__).relative_to(ROOT)),
        'input_directories':['results/round2/analysis_cifar','results/round2/analysis_speech'],
        'note':'Revision three: all numerical plots use executed analysis records; fig1 is conceptual. Figure 7 uses full-choice reselection frequency and the recorded half-split utility success rate.'},indent=2))
    print('Wrote seven figures as PNG and PDF')


if __name__=='__main__':main()
