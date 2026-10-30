"""QuietRisk beer snapshot preprocessing and fixed-budget policy, version 1.

Feature formulas reuse the approved steps_01_07/features_asof implementation.
Model inputs never include target/event/prediction date/customer identifiers.
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SEED=20261029
LOG_COLUMNS=['purchase_days_30','spend_30','purchase_days_90','spend_90',
             'purchase_days_180','spend_180','purchase_days_365','spend_365',
             'purchase_days_previous90','spend_previous90','lifetime_purchase_days','lifetime_spend']


class SnapshotPreprocessor(BaseEstimator,TransformerMixin):
    def __init__(self,features,scale=False):
        self.features=features
        self.scale=scale

    def raw(self,frame):
        x=pd.DataFrame(index=frame.index)
        for name in self.features:
            if name.endswith('__missing'):
                x[name]=frame[name.removesuffix('__missing')].isna().astype(float)
            else:
                x[name]=pd.to_numeric(frame[name],errors='raise')
                if name in LOG_COLUMNS:
                    if x[name].dropna().lt(0).any():
                        raise ValueError('Negative value in log1p input: '+name)
                    x[name]=np.log1p(x[name])
        return x.astype(float)

    def fit(self,X,y=None):
        x=self.raw(X)
        self.medians_=x.median()
        if self.medians_.isna().any():
            raise ValueError('Entirely missing training feature: '+str(self.medians_[self.medians_.isna()].index.tolist()))
        filled=x.fillna(self.medians_)
        self.mean_=filled.mean()
        self.std_=filled.std(ddof=0).replace(0,1)
        self.feature_names_in_=np.array(self.features,dtype=object)
        return self

    def transform(self,X):
        x=self.raw(X).fillna(self.medians_)
        if self.scale:
            x=(x-self.mean_)/self.std_
        result=x.to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite transformed input')
        return result

    def get_feature_names_out(self,input_features=None):
        return np.array(self.features,dtype=object)


def customer_weights(frame):
    w=1/frame.Customer_ID.map(frame.Customer_ID.value_counts()).to_numpy()
    return w/w.mean()


def tie_break(ids):
    return np.array([int.from_bytes(hashlib.sha256(f'{SEED}:{cid}'.encode()).digest()[:8],'big') for cid in ids],dtype=np.uint64)


def policy(rows,scores,budgets,targets=True):
    """Independent campaign, all selected customers excluded for 180 days."""
    rows=rows.reset_index(drop=True)
    scores=np.asarray(scores)
    ties=tie_break(rows.Customer_ID)
    history={}
    selections=[]
    monthly=[]
    dates=sorted(rows.prediction_date.unique())
    assert len(dates)==len(budgets) and len(rows)==len(scores)
    for date,k in zip(dates,budgets):
        ix=np.flatnonzero(rows.prediction_date.eq(date).to_numpy())
        eligible=np.array([i for i in ix if rows.Customer_ID.iat[i] not in history or
                           (date-history[rows.Customer_ID.iat[i]]).days>=180],dtype=int)
        if len(eligible)<k:
            raise ValueError(f'Budget infeasible on {date}: {len(eligible)} < {k}')
        chosen=eligible[np.lexsort((ties[eligible],-scores[eligible]))[:k]]
        part=rows.iloc[chosen].copy()
        part['score']=scores[chosen]
        selections.append(part)
        m={'prediction_date':str(pd.Timestamp(date).date()),'population':len(ix),'eligible':len(eligible),'selected':k}
        if targets:
            m['hits']=int(part.y_D180.sum())
            m['precision']=float(part.y_D180.mean())
            m['positives_in_pool']=int(rows.iloc[ix].y_D180.sum())
        monthly.append(m)
        for cid in part.Customer_ID:
            history[cid]=date
    picked=pd.concat(selections,ignore_index=True)
    assert picked.Customer_ID.nunique()==len(picked)
    return pd.DataFrame(monthly),picked


def snapshot(p,c,t):
    """Approved as-of formulas; common eligibility recency<90 applied at end."""
    t=pd.Timestamp(t)
    h=p[p.date<=t]
    days=h[['Customer_ID','date']].drop_duplicates().sort_values(['Customer_ID','date'])
    last=days.groupby('Customer_ID').date.max()
    ids=last[(t-last).dt.days<90].index.sort_values()
    x=pd.DataFrame(index=ids)
    x['recency_days']=(t-last.reindex(ids)).dt.days
    x['tenure_days']=(t-days.groupby('Customer_ID').date.min().reindex(ids)).dt.days
    x['lifetime_purchase_days']=days.groupby('Customer_ID').size().reindex(ids)
    x['lifetime_spend']=h.groupby('Customer_ID').Revenues.sum().reindex(ids)
    gaps=days.assign(gap=days.groupby('Customer_ID').date.diff().dt.days)
    for stat in ['median','mean','std']:
        x['gap_'+stat]=getattr(gaps.groupby('Customer_ID').gap,stat)().reindex(ids)
    x['gap_available']=x.gap_median.notna().astype(int)
    x['recency_to_median_gap']=x.recency_days/x.gap_median.replace(0,np.nan)
    for window in [30,90,180,365]:
        recent=h[h.date>t-pd.Timedelta(days=window)]
        dd=days[days.date>t-pd.Timedelta(days=window)]
        x[f'purchase_days_{window}']=dd.groupby('Customer_ID').size().reindex(ids,fill_value=0)
        x[f'spend_{window}']=recent.groupby('Customer_ID').Revenues.sum().reindex(ids,fill_value=0)
    lower,upper=t-pd.Timedelta(days=180),t-pd.Timedelta(days=90)
    dd=days[(days.date>lower)&(days.date<=upper)]
    prev=h[(h.date>lower)&(h.date<=upper)]
    x['purchase_days_previous90']=dd.groupby('Customer_ID').size().reindex(ids,fill_value=0)
    x['spend_previous90']=prev.groupby('Customer_ID').Revenues.sum().reindex(ids,fill_value=0)
    for prefix,now,before in [('frequency','purchase_days_90','purchase_days_previous90'),('spend','spend_90','spend_previous90')]:
        x[prefix+'_change90']=x[now]-x[before]
        x[prefix+'_ratio90']=x[now]/x[before].replace(0,np.nan)
        x[prefix+'_previous90_zero']=x[before].eq(0).astype(int)
    recent=h[h.date>lower]
    for cat in ['Beer','Snacks','Accessories']:
        amount=recent[recent.Category==cat].groupby('Customer_ID').Revenues.sum().reindex(ids,fill_value=0)
        x['category_share180_'+cat]=amount/x.spend_180.replace(0,np.nan)
    x['category_count180']=recent.groupby('Customer_ID').Category.nunique().reindex(ids,fill_value=0)
    total=recent.groupby('Customer_ID').size().reindex(ids)
    digital=recent[recent.Purchase_Channel=='Digital'].groupby('Customer_ID').size().reindex(ids,fill_value=0)
    x['digital_row_share180']=digital/total
    x['has_record_before_cutoff']=x.index.isin(set(c[(c.record_date<=t)&c.Complaints.gt(0)].Customer_ID)).astype(int)
    x.index.name='Customer_ID'
    x=x.reset_index()
    x['prediction_date']=t
    return x
