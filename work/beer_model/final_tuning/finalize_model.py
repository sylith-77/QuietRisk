"""Refit internally chosen candidates; outer confirmation cannot change decision."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import train_stage1 as stage
from beer_pipeline import policy,customer_weights,snapshot

ROOT=stage.ROOT
APP=stage.APP
ART=APP/'beer_artifacts'
ART.mkdir(exist_ok=True)


def main():
    decision=json.loads((ROOT/'model_decision_before_outer_validation.json').read_text(encoding='utf-8'))
    frozen_hash=hashlib.sha256((ROOT/'model_decision_before_outer_validation.json').read_bytes()).hexdigest()
    frame=pd.read_csv(ROOT.parent/'definition_comparison/prepared/common_snapshots_train_validation.csv.gz',
                       dtype={'Customer_ID':str},parse_dates=['prediction_date','outcome_end','event_D180'])
    train=frame[frame.split.eq('train')].copy()
    val=frame[frame.split.eq('validation')&frame.has_record_before_cutoff.eq(0)].copy().reset_index(drop=True)
    defs=json.loads((ROOT.parent/'steps_08_14/feature_plans.json').read_text(encoding='utf-8'))
    budgets=[378,391,370,370,314]
    baseline=policy(val,val.recency_days.to_numpy(),budgets)
    results=[];scores_by_family={};models={}
    for family,selection in [('LR',decision['best_LR']),('RF',decision['best_RF'])]:
        features=defs['original37' if selection['feature_count']==37 else 'deduplicated30']
        rows=train if selection['regime']=='all' else train[train.has_record_before_cutoff.eq(0)]
        model=stage.estimator(family,selection['params'],features)
        print('REFIT FROZEN',family,selection['condition'],selection['params'],len(rows),flush=True)
        with threadpool_limits(limits=4):
            model.fit(rows,rows.y_D180,model__sample_weight=customer_weights(rows))
            score=model.predict_proba(val)[:,1]
        metrics,monthly,picked=stage.evaluate(val,score,budgets,baseline)
        monthly.to_csv(ART/f'validation_monthly_{family}.csv',index=False,encoding='utf-8-sig')
        picked.to_csv(ART/f'validation_selected_{family}.csv.gz',index=False)
        result={'family':family,'condition':selection['condition'],'features':len(features),
                'regime':selection['regime'],'params':selection['params'],'train_rows':len(rows),
                'internal_mean_delta':selection['mean_delta'],'internal_min_fold_delta':selection['min_fold_delta'],**metrics}
        results.append(result);scores_by_family[family]=score;models[family]=model
        bundle={'pipeline':model,'features':features,'family':family,'condition':selection['condition'],
                'params':selection['params'],'seed':stage.SEED,'D':180,'H':180,'recency_eligibility':90,
                'cooldown_days':180,'train_cutoff_end':'2022-05-08','label_observation_end':'2022-11-04',
                'selection_basis':'two internal time folds only','decision_sha256':frozen_hash}
        joblib.dump(bundle,ART/f'candidate_{family}.joblib',compress=1)
        restored=joblib.load(ART/f'candidate_{family}.joblib')
        with threadpool_limits(limits=4):
            assert np.allclose(restored['pipeline'].predict_proba(val)[:,1],score,atol=1e-12)
        print('OUTER CONFIRMATION',family,json.dumps(metrics),flush=True)
    selected_family=decision['selected']['family']
    shutil.copy2(ART/f'candidate_{selected_family}.joblib',ART/'model.joblib')
    selected_result=next(r for r in results if r['family']==selected_family)
    # Even a negative outer check cannot trigger a new search or switch families.
    assert frozen_hash==hashlib.sha256((ROOT/'model_decision_before_outer_validation.json').read_bytes()).hexdigest()
    pd.DataFrame(results).to_csv(ART/'outer_confirmation.csv',index=False,encoding='utf-8-sig')
    reference=baseline[1]
    reference.to_csv(ART/'validation_selected_Recency.csv.gz',index=False)
    baseline[0].to_csv(ART/'validation_monthly_Recency.csv',index=False,encoding='utf-8-sig')
    for family,score in scores_by_family.items():
        val['score_'+family]=score
    val['score']=scores_by_family[selected_family]
    chosen=pd.read_csv(ART/f'validation_selected_{selected_family}.csv.gz',dtype={'Customer_ID':str},parse_dates=['prediction_date'])
    model_keys=set(zip(chosen.Customer_ID,chosen.prediction_date))
    base_keys=set(zip(reference.Customer_ID,reference.prediction_date))
    val['selected_model']=[(cid,t) in model_keys for cid,t in zip(val.Customer_ID,val.prediction_date)]
    val['selected_recency']=[(cid,t) in base_keys for cid,t in zip(val.Customer_ID,val.prediction_date)]
    val.to_csv(ART/'validation_snapshots.csv.gz',index=False)
    # Portable raw files and latest no-outcome scoring preview.
    data=APP/'beer_data';data.mkdir(exist_ok=True)
    source=ROOT.parents[1]/'beer_data_audit'
    for name in ['beer_purchases.csv','customer_complaints.csv']:
        shutil.copy2(source/name,data/name)
    p=pd.read_csv(data/'beer_purchases.csv',dtype={'Customer_ID':str})
    p=p[p.Customer_ID.str.fullmatch(r'\d+')].copy();p['date']=pd.to_datetime(p.Date,format='%m/%d/%Y')
    c=pd.read_csv(data/'customer_complaints.csv',dtype={'Customer_ID':str});c['record_date']=pd.to_datetime(c.Date,format='%m/%d/%Y')
    latest_date=p.date.max()
    latest=snapshot(p,c,latest_date)
    latest=latest[latest.has_record_before_cutoff.eq(0)].copy()
    with threadpool_limits(limits=4):
        latest['score']=models[selected_family].predict_proba(latest)[:,1]
    latest.to_csv(ART/'latest_scores_no_outcomes.csv.gz',index=False)
    # Independent comparison of production feature code with historical cached fields.
    expected=val[val.prediction_date.eq(pd.Timestamp('2023-03-08'))].reset_index(drop=True)
    actual=snapshot(p,c,'2023-03-08');actual=actual[actual.has_record_before_cutoff.eq(0)].reset_index(drop=True)
    columns=[f for f in defs['original37'] if not f.endswith('__missing')]
    pd.testing.assert_frame_equal(actual[['Customer_ID']+columns],expected[['Customer_ID']+columns],check_dtype=False)
    truncated=snapshot(p[p.date.le(latest_date)],c[c.record_date.le(latest_date)],latest_date)
    compare=truncated[truncated.has_record_before_cutoff.eq(0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(latest.drop(columns='score').reset_index(drop=True),compare)
    # Preserve original UCI service, keep its data/artifacts untouched.
    for name in ['app.py','README.md']:
        backup=APP/(name.replace('.py','_iranian_backup.py') if name.endswith('.py') else 'README_iranian_backup.md')
        if not backup.exists():
            shutil.copy2(APP/name,backup)
    (ART/'internal_candidate_ranking.csv').write_bytes((ROOT/'tables/internal_candidate_ranking.csv').read_bytes())
    (ART/'fold_counts.csv').write_bytes((ROOT/'tables/fold_counts.csv').read_bytes())
    (ART/'internal_budgets.csv').write_bytes((ROOT/'tables/internal_budgets.csv').read_bytes())
    stage1=json.loads((ROOT/'stage1_manifest.json').read_text(encoding='utf-8'))
    metrics={'selected_family':selected_family,'selected_condition':decision['selected']['condition'],
        'selection_reason':decision['reason'],'internal_selection':decision['selected'],'internal_best_LR':decision['best_LR'],
        'internal_best_RF':decision['best_RF'],'outer_confirmation':results,'selected_outer':selected_result,
        'recency_outer_hits':int(reference.y_D180.sum()),'validation_rows':len(val),'validation_customers':int(val.Customer_ID.nunique()),
        'budgets':budgets,'budget_total':sum(budgets),'latest_date':str(latest_date.date()),'latest_quiet_customers':len(latest),
        'primary_recommended_strategy':'ML' if decision['selected']['min_fold_delta']>=0 and decision['selected']['mean_delta']>0 else 'Recency',
        'phase1_fits':48,'full_refits':2,'stage2_performed':False,
        'test_evaluated':False,'test_independent_claim':False,
        'old_test_exposure':'2024-02-08 was evaluated in an earlier experiment; never describe it as a wholly new independent test.',
        'checks':{'internal_preprocessing_and_temporal_checks':stage1['checks'],'saved_models_reproduce_scores':True,
                  'asof_features_match_approved_history':True,'latest_features_ignore_future_complaints':True,
                  'no_repeat_selected_customers':chosen.Customer_ID.nunique()==len(chosen),
                  'model_decision_frozen_before_outer':True},
        'dataset':{'purchase_raw_rows':188683,'valid_purchase_rows':len(p),'valid_purchase_customers':int(p.Customer_ID.nunique()),
                   'complaint_rows':len(c),'complaint_customers':int(c.Customer_ID.nunique()),'quarantined_rows':54,
                   'purchase_start':str(p.date.min().date()),'purchase_end':str(p.date.max().date())},
        'source_urls':['https://github.com/Datakortex/Datasets/tree/main/customer_churn_in_beer_industry'],
        'limitations':['Actual collection provenance and complaint receipt/processing-date semantics unverified.',
          'No-record means no observed record by cutoff, not proof of no dissatisfaction.',
          'Existing outer validation has been repeatedly inspected; confirmation is exploratory.',
          'Only two internal periods; recurring customers and overlapping outcome windows remain.',
          'Scores are ranking scores, not calibrated real churn probabilities.',
          'Latest scoring preview has no future outcome labels or validated campaign allocation.']}
    stage.dump(ART/'metrics.json',metrics)
    stage.dump(ROOT/'completion_manifest.json',{'status':'model_saved_and_reproduced','selected_family':selected_family,
        'selected_condition':decision['selected']['condition'],'model_sha256':hashlib.sha256((ART/'model.joblib').read_bytes()).hexdigest(),
        'source_decision_sha256':frozen_hash,'outer_used_for_selection':False,'new_test_evaluation':False,
        'stage1_fits':48,'full_refits':2,'stage2':False})
    print('FINAL MODEL',selected_family,decision['selected']['condition'],decision['reason'],flush=True)


if __name__=='__main__':
    main()
