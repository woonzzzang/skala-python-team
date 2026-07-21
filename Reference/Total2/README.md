# Adult Census Income End-to-End Analysis

## 프로젝트 소개
UCI Adult Census Income 데이터셋을 이용해 데이터 수집부터 정제, EDA, 통계 검정,
시각화, 머신러닝 모델링, 결과 보고서 생성까지 전체 데이터 분석 과정을 하나의
실행 가능한 프로젝트로 구현한 교육용 실습입니다.

## 분석 목표
- 개인의 인구조사 정보(나이, 학력, 직업 등)로 연 소득이 `<=50K`인지 `>50K`인지 분류하는
  머신러닝 파이프라인을 만든다.
- 그 과정에서 Pandas/Polars 비교, 결측치·중복 처리, 통계 검정(t-test, 카이제곱),
  정적/인터랙티브 시각화를 함께 수행하고 결과를 report.md로 자동 정리한다.

## 사용 데이터
- UCI Machine Learning Repository, Adult Data Set
  (<https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data>)
- 원본 파일에는 헤더가 없고, 결측값은 `?`로 표시되어 있습니다.
- 로컬에 `data/raw/adult.data`가 있으면 그 파일을 그대로 사용하고,
  없을 때만 위 URL에서 새로 다운로드합니다.

## 데이터 컬럼 설명
| 컬럼 | 설명 |
|---|---|
| age | 나이 |
| workclass | 고용 형태 (Private, Self-emp 등) |
| fnlwgt | 인구조사 가중치(census weight, 개인을 식별하는 값이 아님) |
| education | 최종 학력 |
| education-num | 학력을 숫자로 표현한 값 (education과 정보가 중복될 수 있음) |
| marital-status | 결혼 상태 |
| occupation | 직업 |
| relationship | 가족 내 관계 |
| race | 인종 |
| sex | 성별 |
| capital-gain | 자본 이득 |
| capital-loss | 자본 손실 |
| hours-per-week | 주당 근무시간 |
| native-country | 출신 국가 |
| income | 타깃. `<=50K` 또는 `>50K` |

## 프로젝트 구조
```
Total2/
├── data/
│   ├── raw/adult.data              # 원본 CSV (다운로드 또는 로컬 파일)
│   └── processed/adult_cleaned.csv # 정제된 데이터
├── output/
│   ├── figures/        # 정적 차트 PNG 5종
│   ├── interactive/    # Plotly 인터랙티브 차트 HTML
│   ├── models/         # 학습된 sklearn Pipeline (.pkl)
│   └── report.md       # 자동 생성 분석 보고서
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # 다운로드/로드, Pandas·Polars 비교
│   ├── preprocessing.py   # 점검, 결측치 방법론, 정제
│   ├── eda.py              # 수치형/범주형/타깃 EDA
│   ├── stats_analysis.py   # 상관분석, t-test, 카이제곱 검정
│   ├── visualization.py    # Seaborn/Matplotlib 정적 차트 + Plotly
│   ├── modeling.py         # sklearn Pipeline 구성/학습/평가/저장
│   ├── report.py           # report.md 자동 생성
│   └── main.py              # 전체 파이프라인 실행 진입점
├── requirements.txt
├── README.md
└── .gitignore
```

> 참고: 원래 계획한 폴더 구조에서는 통계 모듈 이름이 `statistics.py`였지만,
> 이 프로젝트는 `python src/main.py`처럼 스크립트를 직접 실행하기 때문에
> `src/` 폴더가 `sys.path`의 맨 앞에 놓입니다. 이 상태에서 `statistics.py`라는
> 이름을 쓰면 파이썬 표준 라이브러리 `statistics` 모듈을 가려버려서 seaborn 같은
> 라이브러리의 import가 깨지는 문제가 실제로 발생했습니다. 그래서 이름 충돌을
> 피하기 위해 `stats_analysis.py`로 변경했습니다.

## 설치 방법

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
# Mac / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 패키지 설치
pip install -r requirements.txt
# 또는 uv를 쓴다면
uv pip install -r requirements.txt
```

## 실행 방법

```bash
python src/main.py
```

실행 위치는 이 `Total2` 폴더 기준입니다. 경로는 모두 `pathlib.Path`와
`Path(__file__).resolve().parent.parent`로 계산하므로, 어느 위치에서
`python src/main.py`로 실행해도(터미널의 현재 폴더가 달라도) 안전하게 동작합니다.

(선택) `src/main.py`에는 `schedule` 라이브러리로 매일 정해진 시각에
자동 실행하는 `schedule_daily_run()` 예시 함수도 들어 있습니다. 기본 실행에서는
호출되지 않으며, 필요할 때만 별도로 실행합니다.

## 분석 과정
1. 데이터 다운로드 또는 로컬 파일 로드
2. Pandas / Polars로 각각 로드 후 행/열 수, 결측치, income 분포, 중복 제거 결과 비교
3. 기본 점검(shape/info/결측치/중복) + 결측치 처리 방법론 설명 + 형식 정제(공백 제거, income 정규화, 중복 제거)
4. 수치형/범주형/타깃(income) EDA
5. Seaborn/Matplotlib 정적 시각화 5종 PNG 저장
6. Pearson 상관분석 + 상관관계 히트맵
7. t-test(income별 hours-per-week) + 카이제곱 독립성 검정(선택 추가 분석)
8. Plotly 인터랙티브 산점도 HTML 저장
9. sklearn Pipeline(전처리+LogisticRegression) 학습 및 평가, confusion matrix 시각화
10. joblib으로 Pipeline 저장 및 재로딩 검증
11. report.md 자동 생성

## 생성 결과물
- `data/processed/adult_cleaned.csv`
- `output/figures/income_distribution.png`
- `output/figures/numeric_eda.png`
- `output/figures/categorical_eda.png`
- `output/figures/correlation_heatmap.png`
- `output/figures/confusion_matrix.png`
- `output/interactive/adult_income_analysis.html`
- `output/models/adult_income_pipeline.pkl`
- `output/report.md`

## 주요 분석 결과
- 전체 32,561행 중 완전 중복 24행을 제거해 32,537행을 사용했습니다.
- 결측치는 workclass(5.6%), occupation(5.7%), native-country(1.8%) 세 범주형 컬럼에만 있습니다.
- income은 `<=50K` 75.9% vs `>50K` 24.1%로 불균형합니다(약 3.15배).
- income 그룹별 hours-per-week 평균 차이는 통계적으로 유의했습니다(Welch t-test, p < 0.001).
- education, occupation, sex는 모두 income과 통계적으로 유의한 연관성이 있었습니다(카이제곱 검정).

## 모델 평가
- 모델: `LogisticRegression(class_weight="balanced")` + `ColumnTransformer` 전처리를 하나의
  `Pipeline`으로 연결
- Accuracy 0.809 / Precision 0.568 / Recall 0.861 / F1-score 0.685 / ROC-AUC 0.910 (>50K 기준)
- income 클래스가 불균형하기 때문에 Accuracy만으로 성능을 판단하면 안 되며,
  Precision/Recall/F1-score를 함께 봐야 합니다. 실제 값은 실행할 때마다
  `output/report.md`에 다시 자동으로 기록됩니다.

## 결측치 처리 방법
- 수치형 컬럼: 이상치의 영향을 덜 받는 **중앙값(median)**으로 대체
- 범주형 컬럼: **최빈값(most_frequent)**으로 대체 (또는 결측 자체가 의미 있다면 'Unknown' 범주로 두는 방법도 대안)
- 결측 행을 무조건 `dropna()`로 삭제하지 않고, 학습 데이터 보존을 위해 대체를 기본으로 선택
- 결측 원인(MCAR/MAR/MNAR)은 데이터만으로 단정할 수 없으므로 임의로 확정하지 않음
- **중요**: 전체 데이터에 미리 대체값을 채우면 데이터 누수가 생길 수 있으므로,
  실제 대체는 `sklearn Pipeline` 내부에서 train 데이터 기준으로만 수행합니다.
  EDA에서 쓰는 정제 DataFrame은 형식만 정리한 것이고, 결측치 대체는 하지 않습니다.

## 통계 검정 방법
- **t-test**: `scipy.stats.ttest_ind(..., equal_var=False)` (Welch t-test, 양측 검정)로
  income 그룹별 hours-per-week 평균 차이를 검정하고, 유의수준 0.05 기준으로 코드에서 직접 판단합니다.
  p >= 0.05인 경우에도 "두 그룹이 같다"라고 단정하지 않고 "유의한 차이를 확인하지 못했다"라고 표현합니다.
- **카이제곱 검정**: `pd.crosstab` + `scipy.stats.chi2_contingency`로 education/occupation/sex와
  income의 독립성을 검정합니다(선택적 추가 분석). 범주가 많은 컬럼은 상위 10개 외 나머지를 'Other'로
  묶어 기대빈도가 너무 작아지는 문제를 완화합니다.

## 윤리 및 한계
- 이 프로젝트는 교육용 데이터 분석 실습 결과물이며, 실제 채용·대출·보험·복지 대상 선정 등
  사람에게 실질적 영향을 주는 의사결정에 그대로 사용해서는 안 됩니다.
- Adult 데이터는 과거 시점 인구조사 데이터라, 그 시절의 사회적 편향이 모델에 재현될 수 있습니다.
- Accuracy/F1이 높다고 해서 모델이 공정하다는 뜻은 아닙니다.
- race, sex, native-country 같은 민감 속성을 feature에서 제외해도 education, occupation 같은
  다른 변수가 대리 변수(proxy) 역할을 해서 간접적으로 편향이 남을 수 있습니다.
- 이 모델은 인과관계를 설명하거나 개인의 능력·가치를 판단하는 모델이 아니라, 주어진 데이터에서
  나타난 통계적 패턴을 학습한 결과일 뿐입니다.

## GitHub 업로드 시 주의사항
- `.gitignore`에 `.venv/`, `__pycache__/`, `*.pyc`, `.env`, `.DS_Store`를 포함했습니다.
- 원본/정제 데이터(`data/raw/`, `data/processed/`)와 학습된 모델(`output/models/*.pkl`)은
  기본적으로 `.gitignore`에서 제외 처리했습니다. 용량이 크고, 데이터는 URL에서, 모델은
  코드 실행으로 언제든 다시 만들 수 있기 때문입니다.
- 과제 제출 기준상 결과물을 함께 제출해야 한다면 `.gitignore`에서 해당 줄을 지우고
  같이 커밋해도 됩니다. 일반적인 GitHub 공개 프로젝트라면 대용량 원본 데이터는 제외하고
  README에 다운로드 URL과 받는 방법만 적어두는 방식을 권장합니다.
