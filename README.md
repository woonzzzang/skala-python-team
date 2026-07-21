# Adult Census Income End-to-End 분석

UCI `adult.data`를 이용해 데이터 준비, EDA, 시각화, 통계 검정, 머신러닝, 모델 저장, 자동 보고서 생성까지 하나의 Python 프로그램으로 수행하는 팀 프로젝트입니다.

## 프로젝트 문서

- [분석 보고서](보고서.md): 분석 과정, 실제 실행 결과와 해석
- [팀 논의사항](팀_논의사항.md): 전처리·통계·모델링 선택과 최종 결정 근거
- [43개 기준 체크리스트](day2_outputs/criteria_checklist.md): 요구사항별 구현 위치와 상태
- [자동 생성 보고서](day2_outputs/report.md): 프로그램을 실행할 때 실제 숫자로 다시 작성되는 보고서

## 분석 목표

- Pandas와 Polars의 데이터 로드 결과, 결측치, 중복, 그룹 집계, 시간과 메모리를 비교합니다.
- 수치형·범주형 EDA와 Seaborn/Matplotlib·Plotly 시각화로 데이터 특성을 확인합니다.
- 상관분석, 독립표본 t-test, 카이제곱 검정과 효과 크기를 함께 해석합니다.
- 전처리와 Logistic Regression을 하나의 sklearn `Pipeline`으로 구성합니다.
- 평가 결과, Pipeline, 입력 컬럼, 타깃 매핑과 메타데이터를 저장합니다.

## 데이터와 핵심 기준

- 데이터: UCI Adult Census Income `adult.data` 32,561행, 15개 컬럼
- 타깃: `income` (`<=50K`, `>50K`)
- 원본 하이픈 컬럼명(`education-num`, `hours-per-week` 등) 유지
- `?`를 결측값으로 인식하되 결측 행은 삭제하지 않음
- 완전 중복 24행만 분석용 데이터에서 제거하고 원본 파일은 보존
- 결측 대체값은 train 데이터에서만 학습하고 test에는 같은 값으로 변환만 수행
- 실제로 가능한 극단값과 희소 범주는 제거·클리핑·통합하지 않음
- EDA에서는 `education`, 모델에서는 `education-num` 사용
- EDA에서는 `income` 문자열을 유지하고 모델링 복사본만 0·1로 매핑

## 프로젝트 구조

```text
skala-python-team/
├── adult.data
├── 광주_3반_한형준_day2종합실습.py
├── README.md
├── 보고서.md
├── 팀_논의사항.md
├── requirements.txt
└── day2_outputs/
    ├── report.md
    ├── criteria_checklist.md
    ├── adult_income_pipeline.joblib
    ├── adult_income_model_metadata.json
    ├── figures/                 # 정적 PNG 5개, Plotly HTML 6개
    └── tables/                  # 분석 결과 CSV 11개
```

## 설치 및 실행

Python 3.11 이상 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 광주_3반_한형준_day2종합실습.py
```

Windows PowerShell에서는 가상환경을 다음처럼 활성화할 수 있습니다.

```powershell
.venv\Scripts\Activate.ps1
```

데이터와 출력 폴더를 직접 지정할 수도 있습니다.

```bash
python 광주_3반_한형준_day2종합실습.py \
  --data /path/to/adult.data \
  --output-dir /path/to/day2_outputs \
  --benchmark-repeats 3
```

로컬 데이터가 없으면 UCI 원본 URL에서 자동으로 내려받습니다.

## 분석 흐름

1. Pandas와 Polars로 동일한 `adult.data` 로드 및 결과 검증
2. `?` 결측 인식, 완전 중복 제거, 원본 문자열 정규화
3. 수치형·범주형 EDA와 소득 그룹 비교
4. Pearson·Spearman 상관분석
5. 정적 차트 5개와 Plotly 인터랙티브 차트 6개 생성
6. 독립표본 t-test 5개, 카이제곱 검정 7개와 효과 크기 계산
7. 80:20 stratified split 이후 train에서만 전처리 학습
8. Logistic Regression Pipeline 세 가지 비교
9. 최종 Pipeline과 메타데이터 저장·재로딩 검증
10. CSV, 체크리스트와 Markdown 보고서 자동 생성

## 주요 실행 결과

### 데이터 준비

- 원본: 32,561행
- 완전 중복: 24행
- 분석 데이터: 32,537행
- 결측치: `workclass` 1,836건, `occupation` 1,843건, `native-country` 583건
- 중복 제거 후 소득 분포: `<=50K` 24,698건, `>50K` 7,839건

### 모델 비교

| 모델 | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| without_fnlwgt | 0.8090 | 0.5685 | 0.8603 | 0.6846 | 0.9094 |
| all_features | 0.8081 | 0.5670 | 0.8603 | 0.6836 | 0.9096 |
| without_sensitive | 0.8079 | 0.5669 | 0.8597 | 0.6832 | 0.9093 |

최종 모델은 F1 차이가 0.005 이내일 때 설명이 쉬운 모델을 선택한다는 기준에 따라 `without_fnlwgt`입니다. 최종 Confusion Matrix는 `[[3916, 1024], [219, 1349]]`입니다.

## 저장 모델 사용 예

저장 파일은 Pipeline만이 아니라 입력 컬럼과 타깃 매핑을 함께 담은 bundle입니다.

```python
import joblib
import pandas as pd

bundle = joblib.load("day2_outputs/adult_income_pipeline.joblib")
new_data = pd.read_csv("new_adult_rows.csv")
features = new_data[bundle["feature_columns"]]
prediction = bundle["pipeline"].predict(features)
```

## 코드 품질 확인

```bash
ruff check .
ruff format --check .
python -m py_compile 광주_3반_한형준_day2종합실습.py
```

## 윤리와 한계

Adult 데이터는 과거의 사회·노동 구조와 편향을 포함합니다. `race`, `sex`, `native-country` 같은 민감 속성이 포함된 결과를 채용·대출·보험 등 실제 의사결정에 바로 사용해서는 안 됩니다. 높은 Accuracy나 F1이 공정성을 의미하지 않으며, 실제 활용 전에는 집단별 Precision·Recall·오류율과 대리 변수 영향을 별도로 점검해야 합니다.
