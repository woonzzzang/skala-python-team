# Adult Census Income 분석 보고서

## 1. 프로젝트 개요
- 분석 목적: UCI Adult Census Income 데이터를 이용해 개인의 income이 <=50K인지 >50K인지 예측하는
  End-to-End 데이터 분석/모델링 파이프라인을 구현하고, 그 과정에서 데이터 품질과 통계적 특성을 확인한다.
- 사용 데이터: UCI Adult Census Income 데이터셋 (adult.data)
- 분석 대상: 인구조사 응답자 개인 단위 레코드, 총 32561행

## 2. 데이터 품질
- 원본 데이터: 32561행 x 15열
- 정제 후 데이터: 32537행 x 15열
- 완전 중복 행: 24건 (정제 과정에서 제거)
- 주요 결측 컬럼별 결측치 수:
  - workclass: 1836건
  - occupation: 1843건
  - native-country: 583건
- 처리 방법: 이 데이터셋의 결측치는 workclass, occupation, native-country 세 범주형 컬럼에서만 발견된다. 결측이 발생한 정확한 이유(MCAR/MAR/MNAR 여부)는 데이터만으로 단정할 수 없으므로 원인을 임의로 확정하지 않는다.
- 수치형 컬럼: 이상치의 영향을 덜 받는 중앙값(median)으로 대체
- 범주형 컬럼: 최빈값(most_frequent)으로 대체하되, 결측 자체가 의미 있다면 'Unknown' 범주로 별도 처리하는 방법도 대안이 될 수 있음
- 결측 행을 무조건 dropna()로 삭제하지 않고, 학습 데이터 보존을 위해 대체를 기본으로 선택
- 단, 통계 검정처럼 특정 컬럼만 필요한 분석에서는 해당 컬럼의 결측 행만 일시적으로 제거
- 전체 데이터에 미리 대체값을 채우면 train/test 분리 후에도 test 정보가 섞여 데이터 누수가 생길 수 있으므로, 실제 대체는 sklearn Pipeline 내부에서 train 데이터 기준으로만 수행한다.

## 3. EDA 주요 결과

### income 분포
- <=50K: 24698건 (75.91%)
- >50K : 7839건 (24.09%)
- 클래스 비율(다수/소수): 3.15배로 불균형이 존재하므로,
  모델 평가 시 Accuracy 외에 Precision/Recall/F1-score를 함께 봐야 한다.

### 주요 수치형 변수 기술통계
| column         |   count |      mean |       std |   min |     Q1 |   median |     Q3 |            max |   missing |   skewness |   kurtosis |
|:---------------|--------:|----------:|----------:|------:|-------:|---------:|-------:|---------------:|----------:|-----------:|-----------:|
| age            |   32537 |     38.59 |     13.64 |    17 |     28 |       37 |     48 |    90          |         0 |       0.56 |      -0.17 |
| fnlwgt         |   32537 | 189781    | 105556    | 12285 | 117827 |   178356 | 236993 |     1.4847e+06 |         0 |       1.45 |       6.22 |
| education-num  |   32537 |     10.08 |      2.57 |     1 |      9 |       10 |     12 |    16          |         0 |      -0.31 |       0.62 |
| capital-gain   |   32537 |   1078.44 |   7387.96 |     0 |      0 |        0 |      0 | 99999          |         0 |      11.95 |     154.68 |
| capital-loss   |   32537 |     87.37 |    403.1  |     0 |      0 |        0 |      0 |  4356          |         0 |       4.59 |      20.36 |
| hours-per-week |   32537 |     40.44 |     12.35 |     1 |     40 |       40 |     45 |    99          |         0 |       0.23 |       2.92 |

- capital-gain과 capital-loss는 0값 비율이 매우 높고 오른쪽 꼬리가 긴 분포라서,
  큰 값을 단순 오류로 보고 삭제하지 않았다(시각화에서는 log1p 변환으로 확인).
- fnlwgt는 인구조사 가중치(census weight)이므로 값이 크더라도 측정 오류가 아니다.

## 4. 통계 분석

### 상관관계 (숫자형 컬럼, income 제외)
| 변수1           | 변수2            |   상관계수 |
|:--------------|:---------------|-------:|
| education-num | hours-per-week |  0.148 |
| education-num | capital-gain   |  0.123 |
| education-num | capital-loss   |  0.08  |

- 상관관계는 인과관계를 의미하지 않는다.
- education과 education-num은 같은 학력 정보를 다른 형태로 표현하므로 서로 상관이 높을 수 있고,
  모델 feature로 함께 쓸 때 정보 중복에 유의해야 한다.

### t-test: income 그룹별 hours-per-week
- H0: 두 income 그룹의 평균 hours-per-week에는 차이가 없다.
- H1: 두 income 그룹의 평균 hours-per-week에는 차이가 있다.
- >50K 그룹: n=7839, 평균=45.47
- <=50K 그룹: n=24698, 평균=38.84
- t 통계량: 45.0950
- p-value: 0.000000
- Cohen's d: 0.567
- 결론: 귀무가설을 기각하며, 두 income 그룹의 평균 주당 근무시간에 통계적으로 유의한 차이가 있다.
- 주의: 표본 수가 크면 작은 차이도 매우 작은 p-value로 나타날 수 있어, 통계적 유의성과
  실질적 효과 크기(Cohen's d)를 함께 고려해야 한다.

### 카이제곱 독립성 검정 (선택적 추가 분석)
  - education x income: chi2=3593.21, p-value=0.0000, 유의함
  - occupation x income: chi2=3672.96, p-value=0.0000, 유의함
  - sex x income: chi2=1516.54, p-value=0.0000, 유의함

## 5. 모델링
- 사용한 feature: 수치형 ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'], 범주형 ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
- target: income (feature에서는 제외하여 데이터 누수를 방지)
- 전처리: SimpleImputer + StandardScaler(수치형), SimpleImputer + OneHotEncoder(범주형)를
  ColumnTransformer로 연결하고, sklearn Pipeline 내부에서 train 데이터로만 학습
- 사용 모델: LogisticRegression(class_weight="balanced")
- train/test 분할: test_size=0.2, stratify=income, random_state=42
- Accuracy : 0.8090
- Precision: 0.5684
- Recall   : 0.8610
- F1-score : 0.6848
- ROC-AUC  : 0.9099

## 6. 주요 인사이트
- 아래 내용은 이번 실행에서 나온 데이터 기반 결과이며, 인과관계로 확대 해석하지 않는다.
- income 클래스는 <=50K가 다수를 차지하는 불균형 데이터이므로, Accuracy만으로 모델을 평가하면
  소수 클래스(>50K)를 잘 못 맞혀도 점수가 높게 나올 수 있다.
- hours-per-week는 income 그룹 간 통계적으로 유의한 차이를 보였으나, 이는 상관관계 수준의 관찰이며
  근무시간이 소득을 결정한다는 인과관계로 해석해서는 안 된다.

### 윤리 및 편향 주의
- 이 모델은 수업용 데이터 분석 실습 결과물이며, 교육 목적으로만 사용한다.
- 실제 채용, 대출, 보험, 복지 대상 선정 등 사람에게 실질적 영향을 주는 의사결정에 그대로 사용하면 안 된다.
- Adult 데이터는 과거 시점의 인구조사 데이터이므로, 그 시절 사회적 편향이 모델에 그대로 재현될 수 있다.
- Accuracy/F1 같은 지표가 높다고 해서 모델이 공정하다는 뜻은 아니다.
- race, sex, native-country 같은 민감 속성을 feature에서 빼더라도 education, occupation 등
  다른 변수가 민감 속성의 대리 변수(proxy) 역할을 해서 간접적으로 편향이 남을 수 있다.
- 이 모델은 인과관계를 설명하거나 개인의 능력·가치를 판단하는 모델이 아니라,
  주어진 데이터에서 나타난 통계적 패턴을 학습한 결과일 뿐이다.


## 7. 생성 파일
- 정제 데이터 CSV: data/processed/adult_cleaned.csv
- 정적 차트 PNG: output/figures/income_distribution.png, output/figures/numeric_eda.png, output/figures/categorical_eda.png, output/figures/correlation_heatmap.png, output/figures/confusion_matrix.png
- Plotly 인터랙티브 차트 HTML: output/interactive/adult_income_analysis.html
- 학습된 모델 Pipeline: output/models/adult_income_pipeline.pkl
