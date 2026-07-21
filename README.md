# Adult Census Income 팀 프로젝트

UCI `adult.data`를 이용해 데이터 준비 → EDA → 시각화 → 통계 검정 → 머신러닝 → 모델 저장 → 자동 보고서 생성을 한 번에 수행하는 End-to-End 프로젝트입니다.

## 핵심 분석 기준

- 원본 컬럼명(`education-num`, `hours-per-week` 등)을 유지합니다.
- `?`는 결측값으로 읽되 결측 행은 삭제하지 않습니다.
- 완전히 같은 중복 행만 분석용 데이터에서 제거하고 `adult.data` 원본은 변경하지 않습니다.
- EDA에는 원본 결측을 유지하며, 모델의 중앙값·최빈값 대체 기준은 80% train 데이터에서만 학습합니다.
- 실제로 가능한 극단값과 희소 범주는 제거하거나 임의로 변환하지 않습니다.
- EDA에서는 `education`, 모델에서는 `education-num`을 사용합니다.
- `income`은 EDA에서 문자열로 유지하고 모델 학습 직전에 `<=50K: 0`, `>50K: 1`로 매핑합니다.
- Logistic Regression에는 `StandardScaler`, `OneHotEncoder`, `class_weight="balanced"`를 사용합니다.

## 실행 방법

Python 3.11 이상 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 광주_3반_한형준_day2종합실습.py
```

데이터가 현재 폴더에 없으면 UCI 원본 URL에서 내려받습니다. 별도 데이터와 출력 폴더도 지정할 수 있습니다.

```bash
python 광주_3반_한형준_day2종합실습.py \
  --data /path/to/adult.data \
  --output-dir /path/to/day2_outputs \
  --benchmark-repeats 3
```

## 산출물

- [자동 분석 보고서](day2_outputs/report.md)
- [43개 기준 점검표](day2_outputs/criteria_checklist.md)
- `day2_outputs/figures/`: Seaborn/Matplotlib 정적 차트 5개와 Plotly 인터랙티브 차트 6개
- `day2_outputs/tables/`: EDA·상관·통계 검정·모델 평가 CSV 11개
- `day2_outputs/adult_income_pipeline.joblib`: 전처리기와 분류기 전체 Pipeline, 입력 컬럼, 타깃 매핑을 담은 bundle
- `day2_outputs/adult_income_model_metadata.json`: 데이터 분할, train에서 학습된 결측 대체값, 평가 지표 등의 메타데이터

현재 재현 실행에서는 `fnlwgt` 제외 모델이 최종 선택됐습니다. 테스트 성능은 Accuracy 0.8090, F1 0.6846, ROC-AUC 0.9094이며, 실행 환경에 따라 로딩 시간만 달라질 수 있습니다.

## 저장 모델 사용 예

```python
import joblib
import pandas as pd

bundle = joblib.load("day2_outputs/adult_income_pipeline.joblib")
new_data = pd.read_csv("new_adult_rows.csv")
features = new_data[bundle["feature_columns"]]
prediction = bundle["pipeline"].predict(features)
```

모델에는 `race`, `sex` 같은 민감 변수가 포함되어 있습니다. 과거 인구·노동 구조의 편향이 예측에 반영될 수 있으므로 실제 의사결정에 바로 사용하지 말고 집단별 오류율과 공정성을 별도로 검토해야 합니다.

## 코드 품질 확인

```bash
ruff check .
ruff format --check .
python -m py_compile 광주_3반_한형준_day2종합실습.py
```
