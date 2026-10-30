# QuietRisk · 불만 무기록 고객의 구매 중단 위험 분석

동일한 고객 관리 예산으로 최근 구매일(Recency) 규칙보다 더 많은 **불만 기록 없는 고유 구매 중단 위험 고객**을 찾을 수 있는지 검증하는 머신러닝 서비스입니다.

## 제출 링크

- GitHub 소스: https://github.com/sylith-77/QuietRisk
- Website / 배포 URL: https://quietrisk-sylith-77.streamlit.app/
- README: https://github.com/sylith-77/QuietRisk/blob/main/README.md

## 서비스 기능

1. 서비스 목적·이탈 정의·분석 모집단 안내
2. 기준일까지의 고객 구매 및 불만 기록 현황
3. 불만 무기록 고객의 위험 순위와 모델·Recency 관리 대상 비교
4. 고객별 저장 모델 예측 및 월별 구매 활동 변화
5. 내부 시간순 검증과 기존 검증 재확인 결과

## 검증 조건

| 항목 | 설정 |
|---|---|
| 주 정의 | D180/H180: 이후 180일 내 연속 180일 미구매 사건 발생 |
| 보조 정의 | D90/H180 기존 비교 결과 유지 |
| 평가 고객 | 기준일 구매 경과일 90일 미만·불만 기록 없음 |
| 선정 정책 | 선정 고객 전원 180일 재선정 제한, 방법별 별도 이력 |
| 기존 검증 예산 | 월별 378·391·370·370·314명, 총 1,823건 |
| 우선 지표 | 동일 정책 Recency 대비 고유 이탈 고객 순증 |
| 최종 모델 | RF / 37개 피처 / 무불만 전용 학습 |

## 저장 결과

내부 시간순 검증 F1·F2에서 각각 Recency 대비 **+47명, +36명**으로 후보를 선택했습니다.
선택을 고정한 뒤 반복 사용된 기존 검증 구간을 제한적으로 재확인했습니다.

| 기존 검증 재확인 | 고유 적중 고객 |
|---|---:|
| Recency | 860명 |
| 선정 RF | 901명 |
| 순증 | +41명 |

위 +41명은 독립 테스트 성능이나 실제 이탈 방지 효과가 아닙니다.
기존 검증 구간은 여러 실험에 사용되었고, 과거 테스트 평가 이력도 있습니다.
배포 준비 중 추가 학습·튜닝·최종 테스트 평가는 수행하지 않았습니다.

## 로컬 실행

Python **3.12** 환경에서 저장소 루트에서 실행합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run quietrisk/app.py
```

일반 서비스 실행은 저장된 모델을 불러오며 재학습하지 않습니다.
모델 파일은 이 저장소의 출처를 확인한 뒤 로드하세요.

## Streamlit Community Cloud 설정

- Repository: `sylith-77/QuietRisk`
- Branch: `main`
- Main file path: `quietrisk/app.py`
- Advanced settings → Python version: `3.12`
- 의존성: 루트 `requirements.txt`
- 테마: 루트 `.streamlit/config.toml`
- API 키·DB·Secrets 불필요

참고: [공식 배포 문서](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

## 파일 구성

```text
quietrisk/
  app.py                    # Streamlit 진입점
  beer_pipeline.py          # 기준일 피처·전처리·고객 가중치·선정 정책
  beer_artifacts/           # 저장 RF와 평가·예측 산출물
  beer_data/                # 공개 사례 데이터 사본
  QuietRisk_저장결과_검토.ipynb  # 저장 결과를 읽는 미실행 검토 노트북
work/
  beer_model/final_tuning/  # 당시 학습·모델 선택 구현과 기록
  beer_model/definition_comparison/prepared/ # 기존 학습·검증 스냅숏
  beer_model/steps_08_14/feature_plans.json   # 37·30개 피처 정의
  beer_data_audit/          # 학습 구현에서 사용하는 원본 사본
requirements.txt
README.md
```

학습 구현은 실행 서비스와 구분되어 있습니다. 일반 실행 시 호출되지 않습니다.
재학습은 별도 실험 버전과 승인을 전제로 하며, 저장 모델의 재현에는 필요하지 않습니다.
최종 테스트 자료는 배포에 포함하지 않습니다.

## 데이터 출처와 한계

[Datakortex 공개 맥주 사례 데이터](https://github.com/Datakortex/Datasets/tree/main/customer_churn_in_beer_industry)를 사용했습니다.
원자료는 원 제공자의 권리·이용 조건을 따릅니다. 이 저장소는 원자료의 별도 사용권을 부여하지 않습니다.
실제 기업 수집 여부와 불만 Date의 접수일·처리일 의미는 확인되지 않았습니다.
무불만은 기준일까지 관측된 불만 기록이 없다는 뜻이며 실제 불만족 부재를 보장하지 않습니다.
이탈은 구매 행동으로 정의한 사건이며 실제 해지 상태와 다릅니다.
위험 점수는 순위 비교용이며 보정된 실제 이탈확률이 아닙니다.
최신 시점은 이후 정답이 없어 점수 조회만 제공합니다.
