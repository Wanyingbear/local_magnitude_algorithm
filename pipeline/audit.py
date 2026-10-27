"""Independently check held-out predictions and write compact k50 results."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from .common import (ROOT, DATA, CACHE, RESULTS, read_json, save_json, settings,
                     dataset, splits, feature_signature, run_signature,
                     checkpoint_path, mean_valid, sha, environment)


def scores(y, prediction):
    if y.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError('Invalid predictions')
    pcc = None if len(y)<2 or np.ptp(y)<1e-12 or np.ptp(prediction)<1e-12 else float(pearsonr(y, prediction).statistic)
    residual = prediction-y
    return dict(pcc=pcc, rmse=float(np.sqrt(np.mean(residual**2))), mae=float(np.mean(np.abs(residual))))


def check_task1():
    meta = read_json(DATA/'datasets.json')
    prep = read_json(CACHE/'preparation.json')
    expected = pd.read_csv(RESULTS/'task1_metrics.csv')
    if len(expected) != 2*len(meta['task1_proteins']) or expected.duplicated(['protein','method']).any():
        raise ValueError('Missing or repeated Task1 rows')
    rows = []
    for protein in meta['task1_proteins']:
        if sha(DATA/'task1'/f'{protein}.npz') != prep['task1_data_sha256'][protein]:
            raise ValueError('Task1 input changed; run prepare first: '+protein)
        with np.load(DATA/'task1'/f'{protein}.npz') as z:
            y = z['y']
        with np.load(CACHE/'task1'/f'{protein}.npz') as z:
            for case in ['global', 'local']:
                features = z[case+'_features']
                if features.shape != (len(y),12) or not np.isfinite(features).all():
                    raise ValueError('Invalid Task1 features: '+protein)
                result = scores(y, z[case+'_prediction'])
                original = expected[(expected.protein==protein)&(expected.method==case+'12')]
                if len(original)!=1:
                    raise ValueError('Task1 row missing')
                np.testing.assert_allclose(list(result.values()), original[['pcc','rmse','mae']].to_numpy()[0],
                                           rtol=1e-8, atol=1e-8, equal_nan=True)
                rows.append(dict(protein=protein,method=case+'12',n=len(y),**result))
    table = pd.DataFrame(rows)
    expected_summary = pd.read_csv(RESULTS/'task1_summary.csv')
    for subset, ids in meta['task1_subsets'].items():
        for method, g in table[table.protein.isin(ids)].groupby('method'):
            original=expected_summary[(expected_summary.subset==subset)&(expected_summary.method==method)]
            actual=[g.pcc.mean(),np.sqrt(np.average(g.rmse**2,weights=g.n)),np.average(g.mae,weights=g.n)]
            if len(original)!=1:
                raise ValueError('Task1 subset row missing')
            np.testing.assert_allclose(actual,original[['mean_pcc','rmse','mae']].to_numpy()[0],atol=1e-8,rtol=1e-8)
    return len(rows)


def summarize(protocols=None, verify_task1=True):
    from .train import initialize, STATE
    RESULTS.mkdir(parents=True, exist_ok=True)
    if verify_task1:
        save_json(RESULTS/'validation.json',dict(passed=False,status='audit in progress'))
    initialize()
    data=STATE['data'];y=data['y'];groups=data['groups'];proteins=pd.unique(groups)
    config=settings();protocols=config['protocols'] if protocols is None else protocols
    if not protocols or len(set(protocols))!=len(protocols):
        raise ValueError('Choose distinct nonempty protocols')
    task1_checks=check_task1() if verify_task1 else 0
    meta=read_json(DATA/'datasets.json')
    records=[];protein_records=[];count=0;imported=0;environment_differences=0
    current_env=environment()
    for protocol in protocols:
        for case in config['cases']:
            for name in config['models']:
                sig=run_signature(STATE['signatures'][case],protocol,name)
                pred=np.full(len(y),np.nan);baseline=np.full(len(y),np.nan);coverage=np.zeros(len(y),int)
                correlations=[]
                for fold,tr,te in splits(groups,protocol):
                    if np.intersect1d(tr,te).size:
                        raise ValueError('Train/test row overlap')
                    if protocol!='atom_cv' and np.intersect1d(groups[tr],groups[te]).size:
                        raise ValueError('Train/test protein overlap')
                    path=checkpoint_path(protocol,case,name,fold)
                    with np.load(path,allow_pickle=False) as z:
                        checkpoint=json.loads(str(z['metadata'].item()))
                        if checkpoint['signature']!=sig:
                            raise ValueError('Checkpoint signature mismatch: '+str(path))
                        if any(checkpoint[k]!=v for k,v in dict(protocol=protocol,case=case,model=name,fold=fold).items()):
                            raise ValueError('Checkpoint identity mismatch')
                        np.testing.assert_array_equal(z['test_indices'],te)
                        p=z['prediction'];measured=scores(y[te],p)
                        np.testing.assert_allclose(z['training_mean'],y[tr].mean(),rtol=1e-12,atol=1e-12)
                        pred[te]=p;baseline[te]=float(z['training_mean']);coverage[te]+=1
                    correlations.append(measured['pcc']);count+=1
                    imported+=checkpoint['origin']=='imported'
                    environment_differences+=checkpoint.get('environment')!=current_env
                if not np.all(coverage==1):
                    raise ValueError('Every target must occur once in testing')
                per=[]
                for protein in proteins:
                    selected=groups==protein
                    row=dict(protocol=protocol,case=case,model=name,protein=protein,
                             n=int(selected.sum()),**scores(y[selected],pred[selected]))
                    per.append(row);protein_records.append(row)
                mean_protein=mean_valid([row['pcc'] for row in per])
                mean_fold=mean_valid(correlations);pooled=scores(y,pred);b=scores(y,baseline)
                records.append(dict(protocol=protocol,case=case,model=name,
                    primary_pcc=mean_protein if protocol=='lopo' else mean_fold,
                    mean_fold_pcc=mean_fold,mean_protein_pcc=mean_protein,pooled_pcc=pooled['pcc'],
                    rmse=pooled['rmse'],mae=pooled['mae'],baseline_rmse=b['rmse'],
                    baseline_mae=b['mae'],proteins=len(proteins),rows=len(y)))
    metrics=pd.DataFrame(records);per=pd.DataFrame(protein_records)
    # Preserve completed, unselected protocols in the summary tables.
    def merge_existing(frame,path,keys):
        if path.exists():
            old=pd.read_csv(path)
            old=old[~old.protocol.isin(protocols)]
            frame=pd.concat([old,frame],ignore_index=True)
        return frame.sort_values(keys).reset_index(drop=True)
    metrics=merge_existing(metrics,RESULTS/'task2_metrics.csv',['protocol','case','model'])
    per=merge_existing(per,RESULTS/'task2_per_protein.csv',['protocol','case','model','protein'])
    comparisons=[]
    for (protocol,name),g in metrics.groupby(['protocol','model']):
        both=g.set_index('case')
        a,b=both.loc['global49'],both.loc['local49']
        comparisons.append(dict(protocol=protocol,model=name,global_pcc=a.primary_pcc,local_pcc=b.primary_pcc,
            delta_pcc=b.primary_pcc-a.primary_pcc,global_rmse=a.rmse,local_rmse=b.rmse,global_mae=a.mae,local_mae=b.mae))
    comparisons=pd.DataFrame(comparisons)
    subset_rows=[]
    for (case,name),g in per[per.protocol=='lopo'].groupby(['case','model']):
        for subset,ids in meta['task2_subsets'].items():
            s=g[g.protein.isin(ids)]
            subset_rows.append(dict(case=case,model=name,subset=subset,proteins=len(s),pcc=mean_valid(s.pcc)))
    metrics.to_csv(RESULTS/'task2_metrics.csv',index=False)
    per.to_csv(RESULTS/'task2_per_protein.csv',index=False)
    comparisons.to_csv(RESULTS/'comparison.csv',index=False)
    if subset_rows:
        pd.DataFrame(subset_rows).to_csv(RESULTS/'lopo_subsets.csv',index=False)
    write_report(comparisons,metrics)
    if verify_task1:
        save_json(RESULTS/'validation.json',dict(passed=True,scope=protocols,
                  complete=set(protocols)==set(config['protocols']),task1_fits_checked=task1_checks,
                  task2_proteins=len(proteins),task2_rows=len(y),fold_prediction_files_checked=count,
                  imported_checkpoints=imported,checkpoints_from_different_or_unknown_environment=environment_differences,
                  all_rows_covered_once_per_protocol=True,protein_split_overlap=0,
                  note='Cached predictions are verified, not retrained by audit. Imported checkpoints retain their original lineage.'))
    print(comparisons.to_string(index=False),flush=True)
    print(f'Checked {count} prediction checkpoints; results: {RESULTS}',flush=True)
    return metrics


def write_report(comparisons,metrics):
    def table(frame):
        columns=list(frame.columns)
        lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        for row in frame.itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
        return '\n'.join(lines)
    task1=pd.read_csv(RESULTS/'task1_summary.csv')
    text='# Global and local k50 results\n\n'
    text+='Local: k=50 including the center, p=1,2, alpha=0.1,...,0.6. Global: p=1,2, alpha=0.3,...,0.8. Task1 uses 12 geometry features; Task2 adds the same 37 auxiliary inputs.\n\n'
    text+='## Task1: within-protein fitting\n\n'+table(task1)+'\n\n'
    text+='## Task2: held-out prediction\n\n'+table(comparisons)+'\n\n'
    text+='PCC averages test folds for ten-fold protocols and proteins for LOPO. RMSE/MAE pool all target rows. Atom CV allows other residues of the same protein in training. Protein CV and LOPO separate proteins, not homology clusters.\n\n'
    text+='## Calibration check\n\n'+table(metrics[['protocol','case','model','rmse','baseline_rmse']])+'\n\n'
    text+='One seed is evaluated. The configuration was selected in earlier exploration on overlapping proteins; these results are not independent external validation. A higher PCC does not guarantee lower error on every protein or useful absolute calibration. The two alpha banks differ, so changes are not solely attributable to locality.\n\n'
    text+='Detailed numeric outputs are in this directory. Resumable predictions are in cache/k50/task2. Set FORCE_RETRAIN=True in spyder_train.py for new fits; valid existing checkpoints otherwise resume without training. Audit verifies saved predictions rather than proving a fresh training run occurred.\n'
    (RESULTS/'REPORT.md').write_text(text,encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocols',nargs='+',choices=['protein_cv','atom_cv','lopo'])
    summarize(**vars(parser.parse_args()))
