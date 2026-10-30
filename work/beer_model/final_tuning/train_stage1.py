"""Approved implementation: two internal time folds, 48 first-stage fits only.
No outer-validation or test outcome participates in candidate choice.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys
import json
import hashlib
import time
import importlib.util
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
APP=ROOT.parents[2]/'quietrisk'
# ROOT is ML/work/beer_model/final_tuning, so ML is parents[2].
sys.path.insert(0,str(APP))
from beer_pipeline import SnapshotPreprocessor,customer_weights,policy,snapshot
for d in ['models','scores','tables','figures']:
    (ROOT/d).mkdir(exist_ok=True)
SEED=20261029
FOLDS=[{'name':'F1','train_end':'2020-11-08','validation':['2021-05-08','2021-06-08','2021-07-08','2021-08-08','2021-09-08']},
       {'name':'F2','train_end':'2021-04-08','validation':['2021-10-08','2021-11-08','2021-12-08','2022-01-08','2022-02-08']}]
GRID={'LR':[{'C':c} for c in [.1,1,10]],
      'RF':[{'max_depth':8,'min_samples_leaf':30,'max_features':.5},
            {'max_depth':None,'min_samples_leaf':30,'max_features':.5},
            {'max_depth':8,'min_samples_leaf':80,'max_features':.5}]}
CONDITIONS={'A':(37,'all'),'B':(30,'all'),'C':(37,'quiet'),'D':(30,'quiet')}
CHECKS=[]


def check(name,ok):
    assert ok,name
    CHECKS.append({'check':name,'passed':bool(ok)})


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


def csv(df,name):
    df.to_csv(ROOT/'tables'/name,index=False,encoding='utf-8-sig')


def estimator(family,params,features):
    model=(LogisticRegression(C=params['C'],l1_ratio=0,solver='lbfgs',max_iter=3000,
                              class_weight=None,random_state=SEED) if family=='LR' else
           RandomForestClassifier(**params,n_estimators=400,bootstrap=True,oob_score=False,
                                  n_jobs=4,class_weight=None,random_state=SEED))
    return Pipeline([('preprocess',SnapshotPreprocessor(tuple(features),scale=family=='LR')),('model',model)])


def evaluate(rows,scores,budgets,baseline):
    months,picked=policy(rows,scores,budgets)
    bm,bp=baseline
    months['recency_hits']=bm.hits.to_numpy()
    months['delta']=months.hits-months.recency_hits
    hits=set(picked.loc[picked.y_D180.eq(1),'Customer_ID'])
    basehits=set(bp.loc[bp.y_D180.eq(1),'Customer_ID'])
    common=hits&basehits
    # Pair actual event for lead-time comparison, independent of composition.
    a=picked[picked.y_D180.eq(1)][['Customer_ID','event_D180','prediction_date']]
    b=bp[bp.y_D180.eq(1)][['Customer_ID','event_D180','prediction_date']]
    paired=a.merge(b,on=['Customer_ID','event_D180'],suffixes=('_model','_rule'))
    advance=(paired.prediction_date_rule-paired.prediction_date_model).dt.days
    out={'hits':len(hits),'recency_hits':len(basehits),'delta_unique':len(hits)-len(basehits),
         'model_only':len(hits-basehits),'rule_only':len(basehits-hits),'both':len(common),
         'selected':sum(budgets),'precision':len(hits)/sum(budgets),
         'recall':len(hits)/rows.y_D180.sum(),'AP':float(average_precision_score(rows.y_D180,scores)),
         'worst_month':int(months.delta.min()),'negative_months':int(months.delta.lt(0).sum()),
         'matched_events':len(paired),'paired_advance_median':float(advance.median()) if len(advance) else None}
    return out,months,picked


def choose_candidates(lr,rf):
    """Approved order: primary mean gain, stability, then simplicity for small gaps."""
    difference=rf['mean_delta']-lr['mean_delta']
    if difference>5:
        return rf,'RF has a clear mean unique-customer gain advantage (>5); primary metric first'
    if difference < -5:
        return lr,'LR has a clear mean unique-customer gain advantage (>5) and simpler operation'
    if lr['min_fold_delta']<0 and rf['min_fold_delta']>=0:
        return rf,'Small mean gap, but only RF is nonnegative on both internal folds'
    return lr,'Mean gap within 5 customers; prefer simpler LR'


def main():
    started=time.time()
    source=ROOT.parent/'definition_comparison/prepared/common_snapshots_train_validation.csv.gz'
    frame=pd.read_csv(source,dtype={'Customer_ID':str},parse_dates=['prediction_date','outcome_end','event_D180'])
    # Do not expose existing validation rows to internal tuning.
    train=frame[frame.split.eq('train')].copy()
    del frame
    defs=json.loads((ROOT.parent/'steps_08_14/feature_plans.json').read_text(encoding='utf-8'))
    features={37:defs['original37'],30:defs['deduplicated30']}
    check('Common training cohort unchanged',len(train)==116594 and train.recency_days.lt(90).all())
    check('Fixed feature counts',len(features[37])==37 and len(features[30])==30)
    checksums={'raw_common_snapshots':hashlib.sha256(source.read_bytes()).hexdigest()}
    # Verify reused as-of feature formulas and future exclusion on two known dates.
    p=pd.read_csv(ROOT.parents[1]/'beer_data_audit/beer_purchases.csv',dtype={'Customer_ID':str})
    c=pd.read_csv(ROOT.parents[1]/'beer_data_audit/customer_complaints.csv',dtype={'Customer_ID':str})
    bad=~p.Customer_ID.str.fullmatch(r'\d+')
    check('Original damaged IDs quarantined',int(bad.sum())==54)
    p=p[~bad].copy();p['date']=pd.to_datetime(p.Date,format='%m/%d/%Y')
    c['record_date']=pd.to_datetime(c.Date,format='%m/%d/%Y')
    for t in ['2020-11-08','2021-04-08']:
        date=pd.Timestamp(t)
        current=snapshot(p,c,date)
        truncated=snapshot(p[p.date.le(date)],c[c.record_date.le(date)],date)
        pd.testing.assert_frame_equal(current,truncated)
        raw=train[train.prediction_date.eq(date)].reset_index(drop=True)
        cols=[f for f in features[37] if not f.endswith('__missing')]
        pd.testing.assert_frame_equal(current[['Customer_ID']+cols],raw[['Customer_ID']+cols],check_dtype=False)
        check(t+' snapshot matches approved cache and ignores future',True)
    fold_data={};counts=[];quota_rows=[]
    for fold in FOLDS:
        tr=train[train.prediction_date.le(fold['train_end'])].copy()
        va=train[train.prediction_date.isin(pd.to_datetime(fold['validation']))&train.has_record_before_cutoff.eq(0)].copy().reset_index(drop=True)
        check(fold['name']+' labels mature before validation',tr.outcome_end.max()<va.prediction_date.min())
        check(fold['name']+' internal validation outcomes remain inside training observation',va.outcome_end.max()<=pd.Timestamp('2022-11-04'))
        quotas=[]
        for t,g in va.groupby('prediction_date',sort=True):
            # Outcome-blind, conservative capacity bound works for EVERY ranking.
            idx=len(quotas)
            cap=[378,391,370,370,314][idx]
            q=min(cap,int(np.ceil(.2*len(g))),max(0,len(g)-sum(quotas)))
            check(fold['name']+' positive affordable quota '+str(t.date()),q>0)
            quotas.append(q)
            quota_rows.append({'fold':fold['name'],'prediction_date':str(t.date()),'population':len(g),
                               'budget':q,'fraction':q/len(g),'capacity_lower_bound':len(g)-sum(quotas[:-1])})
        for cond,(n,regime) in CONDITIONS.items():
            x=tr if regime=='all' else tr[tr.has_record_before_cutoff.eq(0)]
            counts.append({'fold':fold['name'],'condition':cond,'features':n,'regime':regime,
                'train_rows':len(x),'train_customers':int(x.Customer_ID.nunique()),
                'train_positive_rows':int(x.y_D180.sum()),'train_first':str(x.prediction_date.min().date()),
                'train_last':str(x.prediction_date.max().date()),'train_outcome_end':str(x.outcome_end.max().date()),
                'validation_rows':len(va),'validation_customers':int(va.Customer_ID.nunique()),
                'validation_first':str(va.prediction_date.min().date()),'validation_last':str(va.prediction_date.max().date()),
                'validation_outcome_end':str(va.outcome_end.max().date()),'budget_total':sum(quotas)})
        fold_data[fold['name']]=(tr,va,quotas,policy(va,va.recency_days.to_numpy(),quotas))
    csv(pd.DataFrame(counts),'fold_counts.csv');csv(pd.DataFrame(quota_rows),'internal_budgets.csv')
    dump(ROOT/'execution_settings.json',{'folds':FOLDS,'grid':GRID,'conditions':CONDITIONS,'seed':SEED,
         'primary':'mean of per-fold net unique D180 churn customers vs same-policy Recency',
         'budget_rule':'min(original slot cap, ceil(20% current common quiet pool), worst-case remaining capacity)',
         'policy':'all selections blocked 180 days; separate empty campaign history per fold and method',
         'no_stage2_reason':'prioritize complete reproducible service; no optimization to exceed old +27',
         'outer_validation_not_used_for_selection':True,'test_evaluation':False,'source_hashes':checksums})
    print('FOLD COUNTS\n'+pd.DataFrame(counts).to_string(index=False),flush=True)
    print('BUDGETS\n'+pd.DataFrame(quota_rows).to_string(index=False),flush=True)
    records=[]
    for fold in FOLDS:
        tr,va,quotas,base=fold_data[fold['name']]
        for cond,(n,regime) in CONDITIONS.items():
            x=tr if regime=='all' else tr[tr.has_record_before_cutoff.eq(0)]
            weights=customer_weights(x)
            check(fold['name']+cond+' equal customer total weights',
                  np.allclose(pd.DataFrame({'id':x.Customer_ID.to_numpy(),'w':weights}).groupby('id').w.sum(),len(x)/x.Customer_ID.nunique()))
            for family,grid in GRID.items():
                for params in grid:
                    token=json.dumps([fold['name'],cond,family,params],sort_keys=True)
                    key=hashlib.sha256(token.encode()).hexdigest()[:16]
                    path=ROOT/'scores'/f'{key}.json'
                    if path.exists():
                        result=json.loads(path.read_text(encoding='utf-8'))
                    else:
                        print('FIT',fold['name'],cond,family,params,flush=True)
                        before=time.time()
                        model=estimator(family,params,features[n])
                        with threadpool_limits(limits=4):
                            model.fit(x,x.y_D180,model__sample_weight=weights)
                            scores=model.predict_proba(va)[:,1]
                        metrics,months,picked=evaluate(va,scores,quotas,base)
                        result={'key':key,'fold':fold['name'],'condition':cond,'family':family,
                            'params':params,'feature_count':n,'regime':regime,'fit_seconds':round(time.time()-before,3),**metrics}
                        # Preprocessing must reproduce TRAIN-only medians exactly.
                        proc=model.named_steps['preprocess']
                        pd.testing.assert_series_equal(proc.medians_,proc.raw(x).median())
                        check(key+' preprocessing training-only and finite',np.isfinite(proc.transform(va)).all())
                        if family=='LR':
                            check(key+' LR convergence',model.named_steps['model'].n_iter_.max()<3000)
                        joblib.dump(model,ROOT/'models'/f'{key}.joblib',compress=1)
                        np.savez_compressed(ROOT/'scores'/f'{key}.npz',scores=scores)
                        months.to_csv(ROOT/'scores'/f'{key}_monthly.csv',index=False)
                        dump(path,result)
                        print('DONE',key,'delta',metrics['delta_unique'],'sec',result['fit_seconds'],flush=True)
                    records.append(result)
                    csv(pd.DataFrame(records),'stage1_progress.csv')
    results=pd.DataFrame(records)
    csv(results,'stage1_all_folds.csv')
    aggregate=[]
    for (cond,family,paramstr),g in results.assign(paramstr=results.params.map(lambda d:json.dumps(d,sort_keys=True))).groupby(['condition','family','paramstr']):
        assert len(g)==2
        aggregate.append({'condition':cond,'family':family,'params':json.loads(paramstr),
            'feature_count':int(g.feature_count.iloc[0]),'regime':g.regime.iloc[0],
            'mean_delta':float(g.delta_unique.mean()),'min_fold_delta':int(g.delta_unique.min()),
            'F1_delta':int(g.loc[g.fold.eq('F1'),'delta_unique'].iloc[0]),
            'F2_delta':int(g.loc[g.fold.eq('F2'),'delta_unique'].iloc[0]),
            'worst_month':int(g.worst_month.min()),'negative_months':int(g.negative_months.sum()),
            'mean_AP':float(g.AP.mean()),'mean_precision':float(g.precision.mean())})
    agg=pd.DataFrame(aggregate).sort_values(['mean_delta','min_fold_delta','worst_month','mean_AP'],ascending=False)
    csv(agg,'internal_candidate_ranking.csv')
    lr=agg[agg.family.eq('LR')].iloc[0].to_dict()
    rf=agg[agg.family.eq('RF')].iloc[0].to_dict()
    selected,reason=choose_candidates(lr,rf)
    decision={'selected':selected,'best_LR':lr,'best_RF':rf,'reason':reason,
              'internal_both_folds_nonnegative':selected['min_fold_delta']>=0,
              'stage2_performed':False,'stage2_reason':'Bounded first-stage complete; reproducible service prioritized, no extra search',
              'outer_validation_used_for_selection':False,'test_used':False}
    dump(ROOT/'model_decision_before_outer_validation.json',decision)
    csv(pd.DataFrame(CHECKS),'verification_checks.csv')
    dump(ROOT/'stage1_manifest.json',{'status':'internal_selection_complete','fits':len(records),
         'checks':len(CHECKS),'elapsed_seconds':round(time.time()-started,2),'decision':decision})
    print('INTERNAL DECISION',json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
