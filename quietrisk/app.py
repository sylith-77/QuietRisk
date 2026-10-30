"""QuietRisk beer customer service. Copied to app.py after model finalization."""
from pathlib import Path
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from threadpoolctl import threadpool_limits
from beer_pipeline import SEED,tie_break

ROOT=Path(__file__).resolve().parent
ART=ROOT/'beer_artifacts'
st.set_page_config(page_title='QuietRisk · 구매 중단 위험',page_icon='◉',layout='wide')
st.markdown('''<style>
.block-container{max-width:1400px;padding-top:2rem} h1{letter-spacing:-1.3px}
[data-testid="stMetric"]{background:#edf5f2;border-radius:10px;padding:16px}
[data-testid="stMetricValue"]{font-size:1.8rem}.stCaption{line-height:1.6}
</style>''',unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(ART/'model.joblib')


@st.cache_data
def load_data():
    report=json.loads((ART/'metrics.json').read_text(encoding='utf-8'))
    val=pd.read_csv(ART/'validation_snapshots.csv.gz',dtype={'Customer_ID':str},
                    parse_dates=['prediction_date','event_D180'])
    latest=pd.read_csv(ART/'latest_scores_no_outcomes.csv.gz',dtype={'Customer_ID':str},parse_dates=['prediction_date'])
    p=pd.read_csv(ROOT/'beer_data/beer_purchases.csv',dtype={'Customer_ID':str})
    p=p[p.Customer_ID.str.fullmatch(r'\d+')].copy()
    p['date']=pd.to_datetime(p.Date,format='%m/%d/%Y')
    c=pd.read_csv(ROOT/'beer_data/customer_complaints.csv',dtype={'Customer_ID':str})
    c['record_date']=pd.to_datetime(c.Date,format='%m/%d/%Y')
    return report,val,latest,p,c


if not (ART/'model.joblib').exists():
    st.error('모델 준비가 완료되지 않았습니다. 배포 파일의 beer_artifacts 폴더를 확인하세요.')
    st.stop()
bundle=load_model()
report,val,latest,p,c=load_data()
family=report['selected_family']
outer=report['selected_outer']
dates=sorted(val.prediction_date.dt.strftime('%Y-%m-%d').unique())
date_choice=st.sidebar.selectbox('분석 기준일',dates+['최신 점수 조회 ('+report['latest_date']+')'],index=len(dates)-1)
is_latest=date_choice.startswith('최신')
date=pd.Timestamp(report['latest_date'] if is_latest else date_choice)
pool=latest.copy() if is_latest else val[val.prediction_date.eq(date)].copy()
pool['_tie']=tie_break(pool.Customer_ID)
pool=pool.sort_values(['score','_tie'],ascending=[False,True]).reset_index(drop=True)
pool['risk_rank']=np.arange(1,len(pool)+1)
pool['recency_rank']=pool.recency_days.rank(method='min',ascending=False).astype(int)
pool['risk_band']=pd.cut(pool.risk_rank/len(pool),[0,.2,.5,1],labels=['우선 검토','관찰','일반'],include_lowest=True)
st.sidebar.title('QuietRisk')
st.sidebar.caption('불만 기록 없는 구매 중단 위험 고객')
st.sidebar.markdown(f'**모델** {family}\n\n**주 정의** D180/H180\n\n**모집단** 구매 경과일 90일 미만\n\n**재선정 제한** 180일')
st.sidebar.caption('무불만은 기준일까지 관측된 불만 기록이 없다는 뜻입니다.')
st.title('QuietRisk')
st.write('구매 활동으로 불만 기록이 없는 고객의 구매 중단 위험을 살펴보고, 같은 관리 예산에서 Recency 규칙과 비교합니다.')
st.caption(f'기준일 {date.date()} · 고객별 점수는 해당 날짜까지의 기록만 사용 · 위험 점수는 실제 이탈확률로 보정하지 않았습니다.')
overview,records,ranking,customer,evaluation=st.tabs(['서비스 개요','구매·불만 기록','관리 우선순위','고객별 위험 분석','모델 검증 결과'])

with overview:
    a,b,d,e=st.columns(4)
    a.metric('무불만 고객',f'{len(pool):,}명')
    b.metric('연속 미구매 기준','180일')
    d.metric('총 선정 예산','1,823건')
    e.metric('모델',family)
    st.subheader('어떤 고객을 찾는 서비스인가요?')
    st.write('기준일 현재 최근 90일 안에 구매했고 불만 기록이 없는 고객 중, 이후 180일 안에 연속 미구매 180일에 도달할 고객을 찾습니다. 사건 이후 재구매해도 과거 이탈 사건은 유지합니다.')
    st.info('검증에서는 선정한 고객 전원을 180일 동안 다시 선정하지 않았습니다. 규칙과 모델은 각자의 선정 이력을 관리하며 월별 같은 인원을 선정합니다.')
    st.subheader('결과를 읽는 방법')
    st.write('관리 우선순위는 점수 순위입니다. 고위험 표시는 고객이 불만족한다는 진단이 아닙니다. 검증 성능은 관측 자료의 고객 발견 결과이며 실제 접촉에 따른 이탈 방지 효과를 뜻하지 않습니다.')
    st.caption('자료: Datakortex 맥주 구매·불만 사례. 실제 수집 여부와 불만 Date의 접수일·처리일 의미는 공개 설명으로 확정되지 않았습니다.')
    st.link_button('원본 데이터 출처',report['source_urls'][0])

with records:
    ph=p[p.date.le(date)]
    ch=c[c.record_date.le(date)]
    a,b,d=st.columns(3)
    a.metric('기준일까지 구매 기록',f'{len(ph):,}건')
    b.metric('구매 고객',f'{ph.Customer_ID.nunique():,}명')
    d.metric('불만 기록 고객',f'{ch.Customer_ID.nunique():,}명')
    st.caption('불만 기록 고객에는 구매 파일에서 연결되지 않는 고객도 포함됩니다. 손상된 고객 ID 54행은 구매 분석에서 격리했습니다.')
    left,right=st.columns(2)
    with left:
        st.subheader('월별 구매금액')
        st.line_chart(ph.set_index('date').Revenues.resample('MS').sum().rename('구매금액'))
    with right:
        st.subheader('상품군별 구매금액')
        st.bar_chart(ph.groupby('Category').Revenues.sum().rename('구매금액'))
    st.subheader('원본 기록 예시')
    st.dataframe(ph[['Customer_ID','Date','Category','Revenues','Purchase_Channel']].head(15),hide_index=True,width='stretch')
    with st.expander('불만 기록 예시와 해석'):
        st.dataframe(ch[['Customer_ID','Date','Complaints']].head(15),hide_index=True,width='stretch')
        st.write('Date는 불만 기록일이라는 중립적 의미로 사용합니다. 반복된 날짜 목록을 임의로 다른 고객의 중복으로 삭제하지 않았고, 횟수·정확한 불만 간격은 예측 입력에서 제외했습니다.')

with ranking:
    st.subheader('불만 무기록 고객의 위험 순위')
    if is_latest:
        st.info('최신 시점은 점수 조회용입니다. 이후 180일 정답과 새로운 관리 예산은 확정되지 않아 관리 대상 선정 성과를 표시하지 않습니다.')
        display=pool.copy()
    else:
        show=st.radio('목록 범위',['전체 위험 순위','모델 관리 대상','Recency 관리 대상'],horizontal=True)
        if show=='모델 관리 대상':
            display=pool[pool.selected_model].copy()
        elif show=='Recency 관리 대상':
            display=pool[pool.selected_recency].copy()
        else:
            display=pool.copy()
        index=dates.index(date_choice)
        st.caption(f'이번 기준일 공통 예산 {report["budgets"][index]}명. 관리 대상은 앞선 선정 이력을 포함해 시간순으로 계산한 결과입니다. 높은 점수라도 180일 제한으로 이번 선정에서 제외될 수 있습니다.')
    columns=['Customer_ID','risk_rank','risk_band','score','recency_days','purchase_days_90','spend_90','frequency_ratio90','recency_rank']
    names={'Customer_ID':'고객 ID','risk_rank':'모델 순위','risk_band':'검토 구간','score':'위험 점수',
           'recency_days':'최근 구매 경과일','purchase_days_90':'90일 구매일 수','spend_90':'90일 구매금액',
           'frequency_ratio90':'직전 대비 구매 빈도 비율','recency_rank':'Recency 순위'}
    view=display[columns].rename(columns=names)
    st.dataframe(view.head(300).style.format({'위험 점수':'{:.3f}','90일 구매금액':'{:,.2f}',
                                           '직전 대비 구매 빈도 비율':'{:.2f}'},na_rep='계산 불가'),hide_index=True,width='stretch')
    st.download_button('현재 목록 전체 CSV 내려받기',view.to_csv(index=False).encode('utf-8-sig'),
                       f'quietrisk_{date.date()}.csv','text/csv')
    if not is_latest and st.checkbox('실제 결과 함께 보기 — 사후 검증'):
        st.dataframe(display[['Customer_ID','score','y_D180','event_D180','selected_model','selected_recency']].head(100),hide_index=True,width='stretch')

with customer:
    st.subheader('고객별 위험 점수와 구매 활동')
    cid=st.selectbox('고객 선택',pool.Customer_ID.tolist(),format_func=lambda x:x)
    row=pool[pool.Customer_ID.eq(cid)].iloc[0]
    with threadpool_limits(limits=4):
        reproduced=float(bundle['pipeline'].predict_proba(pool[pool.Customer_ID.eq(cid)])[:,1][0])
    a,b,d,e=st.columns(4)
    a.metric('위험 점수',f'{reproduced*100:.1f} / 100')
    b.metric('모델 순위',f'{int(row.risk_rank):,} / {len(pool):,}')
    d.metric('최근 구매 경과일',f'{int(row.recency_days)}일')
    e.metric('최근 90일 구매일 수',int(row.purchase_days_90))
    st.caption('저장 모델로 현재 고객의 과거 피처를 다시 예측한 값입니다. 점수 자체를 실제 이탈확률로 해석하지 않습니다.')
    hist=p[p.Customer_ID.eq(cid)&p.date.le(date)].copy()
    activity=hist.set_index('date').Revenues.resample('MS').agg(['sum','count']).rename(columns={'sum':'구매금액','count':'거래 기록 수'})
    left,right=st.columns(2)
    with left:
        st.write('월별 구매금액')
        st.bar_chart(activity[['구매금액']])
    with right:
        st.write('월별 거래 기록 수')
        st.line_chart(activity[['거래 기록 수']])
    change=pd.DataFrame({'구간':['직전 90일','최근 90일'],
                         '구매일 수':[row.purchase_days_previous90,row.purchase_days_90],
                         '구매금액':[row.spend_previous90,row.spend_90]}).set_index('구간')
    st.dataframe(change.style.format({'구매일 수':'{:.0f}','구매금액':'{:,.2f}'}),width='stretch')
    st.write(f'기준일까지 관측된 불만 기록: {int(row.has_record_before_cutoff)} · 과거 구매 간격 중앙값: '+
             (f'{row.gap_median:.1f}일' if pd.notna(row.gap_median) else '이력 부족'))
    st.dataframe(hist[['Date','Category','Revenues','Purchase_Channel']].sort_index().tail(20),hide_index=True,width='stretch')

with evaluation:
    st.subheader('내부 시간순 검증으로 모델 선택')
    st.write(f'선정 모델: {family}, 조건 {report["selected_condition"]}. 기존 검증 구간을 열기 전에 내부 두 폴드 결과로 고정했습니다.')
    rank=pd.read_csv(ART/'internal_candidate_ranking.csv')
    st.dataframe(rank[['condition','family','feature_count','regime','params','F1_delta','F2_delta','mean_delta','worst_month']].head(12),hide_index=True,width='stretch')
    st.caption('F1·F2는 별도 캠페인입니다. 내부 표본 수가 달라 선정 예산도 폴드별로 다르지만 같은 폴드의 모든 방법은 동일합니다.')
    with st.expander('내부 폴드와 예산 상세'):
        st.dataframe(pd.read_csv(ART/'fold_counts.csv'),hide_index=True,width='stretch')
        st.dataframe(pd.read_csv(ART/'internal_budgets.csv'),hide_index=True,width='stretch')
    st.subheader('기존 검증 구간의 제한적 재확인')
    a,b,d,e=st.columns(4)
    a.metric('Recency 고유 적중',f'{report["recency_outer_hits"]}명')
    b.metric('선정 모델 고유 적중',f'{outer["hits"]}명',f'{outer["delta_unique"]:+d}명')
    d.metric('모델만 발견',f'{outer["model_only"]}명')
    e.metric('규칙만 발견',f'{outer["rule_only"]}명')
    cm=pd.read_csv(ART/f'validation_monthly_{family}.csv')
    st.dataframe(cm,hide_index=True,width='stretch')
    st.bar_chart(cm.set_index('prediction_date')[['hits','recency_hits']].rename(columns={'hits':'선정 모델','recency_hits':'Recency'}),stack=False)
    st.warning('이 검증 구간은 여러 실험에서 이미 사용했습니다. 독립적인 테스트 성능이 아니며, 이번에 최종 테스트를 추가 평가하지 않았습니다.')
    st.write(f'같은 사건을 양쪽이 찾은 {outer["matched_events"]}건의 선정 시점 차이 중앙값: {outer["paired_advance_median"]}일. 양수면 모델이 더 일찍 선정했습니다.')
    st.subheader('범위와 한계')
    st.write('실수집 여부·불만 기록일 의미는 미확인입니다. 이탈은 180일 연속 미구매 사건이며 실제 해지 상태가 아닙니다. 과거 2024-02-08 테스트를 본 이력이 있어 해당 구간을 완전히 새로운 독립 테스트라고 부르지 않습니다.')
    st.caption('보조 D90/H180 및 이전 기본 모델의 정책 비교는 제출 보고서에 보존했습니다. 이번 본 튜닝은 D180/H180만 사용했습니다.')
    report_path=ROOT/'outputs/QuietRisk_과제보고서.html'
    if report_path.exists():
        st.download_button('과제 보고서 HTML 내려받기',report_path.read_bytes(),'QuietRisk_과제보고서.html','text/html')
