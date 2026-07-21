# Adult Census Income End-to-End 분석 보고서

- 작성일: 2026-07-21
- 작성자: 광주 3반 한형준
- 데이터: adult.data 단일 파일
- 목표: 연 소득 >50K 이진 분류와 설명 가능한 분석 흐름 구성
- 전체 기준: [criteria_checklist.md](criteria_checklist.md)

## 1. 분석 원칙

- 원본 하이픈 컬럼명과 income 문자열을 EDA까지 유지했습니다.
- 완전 중복 24건만 제거하고 원본 파일은 보존했습니다.
- age, hours-per-week, capital-gain, capital-loss, fnlwgt와 희소 범주는 실제 가능한 관측으로 보고 변경하지 않았습니다.
- 결측 행은 삭제하지 않았습니다. 모델 결측 대체 기준은 train에서만 학습됩니다.
- EDA에서는 education을, 모델에서는 education-num을 사용해 중복 정보를 피했습니다.

## 2. Pandas·Polars 비교

| engine | load_median_seconds | memory_mb | raw_rows | raw_columns | missing_total | duplicate_rows | deduplicated_rows | income_<=50K | income_>50K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pandas | 0.0437 | 19.91 | 32561 | 15 | 4262 | 24 | 32537 | 24698 | 7839 |
| Polars | 0.01393 | 3.949 | 32561 | 15 | 4262 | 24 | 32537 | 24698 | 7839 |

### 처리 전 결측치

| column | missing_count |
| --- | --- |
| workclass | 1836 |
| occupation | 1843 |
| native-country | 583 |

## 3. 수치형 EDA

| variable | count | mean | std | min | q1 | median | q3 | max | iqr | skewness | kurtosis | missing_count | missing_rate_pct | unique_count | zero_rate_pct | iqr_outlier_candidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age | 32537 | 38.59 | 13.64 | 17 | 28 | 37 | 48 | 90 | 20 | 0.5577 | -0.1698 | 0 | 0 | 73 | 0 | 142 |
| fnlwgt | 32537 | 1.898e+05 | 1.056e+05 | 12285 | 1.178e+05 | 1.784e+05 | 2.37e+05 | 1484705 | 1.192e+05 | 1.448 | 6.222 | 0 | 0 | 21648 | 0 | 993 |
| education-num | 32537 | 10.08 | 2.572 | 1 | 9 | 10 | 12 | 16 | 3 | -0.3095 | 0.619 | 0 | 0 | 16 | 0 | 1193 |
| capital-gain | 32537 | 1078 | 7388 | 0 | 0 | 0 | 0 | 99999 | 0 | 11.95 | 154.7 | 0 | 0 | 119 | 91.66 | 2712 |
| capital-loss | 32537 | 87.37 | 403.1 | 0 | 0 | 0 | 0 | 4356 | 0 | 4.593 | 20.36 | 0 | 0 | 92 | 95.33 | 1519 |
| hours-per-week | 32537 | 40.44 | 12.35 | 1 | 40 | 40 | 45 | 99 | 5 | 0.2288 | 2.918 | 0 | 0 | 94 | 0 | 9002 |

IQR 경계 밖 값은 오류 판정이 아니라 후보 수만 집계했습니다. 소득 그룹별 평균·중앙값과 Pearson·Spearman 상세 결과는 `tables/`의 CSV에 저장했습니다. 상관은 인과가 아닙니다.

## 4. 범주형 EDA

| variable | unique_count | mode | mode_count | missing_count | missing_rate | rare_category_count | rare_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| workclass | 8 | Private | 22673 | 1836 | 0.05643 | 2 | 100 |
| education | 16 | HS-grad | 10494 | 0 | 0 | 1 | 100 |
| marital-status | 7 | Married-civ-spouse | 14970 | 0 | 0 | 1 | 100 |
| occupation | 14 | Prof-specialty | 4136 | 1843 | 0.05664 | 1 | 100 |
| relationship | 6 | Husband | 13187 | 0 | 0 | 0 | 100 |
| race | 5 | White | 27795 | 0 | 0 | 0 | 100 |
| sex | 2 | Male | 21775 | 0 | 0 | 0 | 100 |
| native-country | 41 | United-States | 29153 | 582 | 0.01789 | 33 | 100 |

표본 100건 미만 범주는 희소로 표시하지만 합치거나 삭제하지 않았습니다. [범주별 상세 표](tables/categorical_group_detail.csv)에는 전체 인원, >50K 인원, 그룹 내 >50K 비율, 전체 >50K 중 점유율이 포함됩니다.

## 5. 시각화

### 정적 차트

- [01_static_overview](figures/01_static_overview.png)
- [02_static_group_comparison](figures/02_static_group_comparison.png)
- [03_static_numeric_distributions](figures/03_static_numeric_distributions.png)
- [04_static_age_hours_scatter](figures/04_static_age_hours_scatter.png)
- [05_model_confusion_roc](figures/05_model_confusion_roc.png)

### Plotly 인터랙티브 차트

- [11_plotly_education_income_rate](figures/11_plotly_education_income_rate.html)
- [12_plotly_occupation_income_rate](figures/12_plotly_occupation_income_rate.html)
- [13_plotly_age_hours_scatter](figures/13_plotly_age_hours_scatter.html)
- [14_plotly_occupation_sex_income_facet](figures/14_plotly_occupation_sex_income_facet.html)
- [15_plotly_treemap](figures/15_plotly_treemap.html)
- [16_plotly_parallel_categories](figures/16_plotly_parallel_categories.html)

## 6. 독립표본 검정

모든 가설은 결과 확인 전에 양측으로 정의했습니다. 유의수준 0.05를 주 기준으로 사용하고 0.01·0.10 결과도 CSV에 기록했습니다.

| topic | test | mean_difference | ci_95_low | ci_95_high | p_value_display | cohens_d | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 소득 그룹별 hours-per-week | Welch t-test | 6.631 | 6.342 | 6.919 | <1e-300 | 0.5518 | 귀무가설 기각: 두 그룹 평균에 통계적으로 유의한 차이가 있음 |
| 소득 그룹별 age | Welch t-test | 7.464 | 7.172 | 7.755 | <1e-300 | 0.5629 | 귀무가설 기각: 두 그룹 평균에 통계적으로 유의한 차이가 있음 |
| 성별 hours-per-week | Student t-test | 6.014 | 5.737 | 6.292 | <1e-300 | 0.5004 | 귀무가설 기각: 두 그룹 평균에 통계적으로 유의한 차이가 있음 |
| workclass별 hours-per-week | Welch t-test | -4.153 | -4.818 | -3.487 | 1.344e-33 | -0.3486 | 귀무가설 기각: 두 그룹 평균에 통계적으로 유의한 차이가 있음 |
| 소득 그룹별 education-num | Welch t-test | 2.016 | 1.955 | 2.077 | <1e-300 | 0.8321 | 귀무가설 기각: 두 그룹 평균에 통계적으로 유의한 차이가 있음 |

정규성·Levene 등분산성 결과를 확인해 Student 또는 Welch 검정을 선택하고, 비정규성에 대한 보조 확인으로 Mann–Whitney U 결과도 저장했습니다. 통계적 유의성과 실제 효과 크기는 구분해 해석해야 합니다.

## 7. 카이제곱 검정

| variable | p_value_display | cramers_v | minimum_expected | interpretation |
| --- | --- | --- | --- | --- |
| marital-status | <1e-300 | 0.4473 | 5.541 | 귀무가설 기각: income과 통계적으로 유의한 연관성이 있음 |
| education | <1e-300 | 0.3689 | 12.05 | 귀무가설 기각: income과 통계적으로 유의한 연관성이 있음 |
| occupation | <1e-300 | 0.3519 | 2.168 | 귀무가설 기각: income과 통계적으로 유의한 연관성이 있음 |
| sex | <1e-300 | 0.2159 | 2593 | 귀무가설 기각: income과 통계적으로 유의한 연관성이 있음 |
| workclass | 3.352e-220 | 0.1792 | 1.686 | 귀무가설 기각: income과 통계적으로 유의한 연관성이 있음 |
| race | 2.280e-70 | 0.1009 | 65.29 | 귀무가설 기각: income과 통계적으로 유의한 연관성이 있음 |
| native-country | 4.833e-44 | 0.09846 | 0.2409 | 귀무가설 기각: income과 통계적으로 유의한 연관성이 있음 |

카이제곱 p-value와 함께 연관성 크기인 Cramer's V, 기대빈도 진단을 기록했습니다. 연관성은 인과관계를 뜻하지 않습니다.

## 8. Logistic Regression Pipeline

- 분할: train 26,029행 / test 6,508행
- test_size=0.2, random_state=42, stratify=y
- 수치형: train 중앙값 대체 + StandardScaler
- 범주형: train 최빈값 대체 + OneHotEncoder
- 분류기: LogisticRegression, class_weight=balanced
- 최종 모델: without_fnlwgt

### train에서 학습된 결측 대체값

아래 값은 80% train 데이터에 `fit`할 때만 계산됐으며 test 데이터에는 동일 값을 `transform`만 했습니다. 현재 데이터에는 수치형 결측이 없지만 실제 입력을 위한 중앙값 안전망도 Pipeline에 포함했습니다.

| data_type | column | train_fitted_value |
| --- | --- | --- |
| numeric_median | age | 37.0 |
| numeric_median | education-num | 10.0 |
| numeric_median | capital-gain | 0.0 |
| numeric_median | capital-loss | 0.0 |
| numeric_median | hours-per-week | 40.0 |
| categorical_most_frequent | workclass | Private |
| categorical_most_frequent | marital-status | Married-civ-spouse |
| categorical_most_frequent | occupation | Prof-specialty |
| categorical_most_frequent | relationship | Husband |
| categorical_most_frequent | race | White |
| categorical_most_frequent | sex | Male |
| categorical_most_frequent | native-country | United-States |

### 모델 비교

| model | purpose | raw_feature_count | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| without_fnlwgt | 표본 가중치 제외 민감도 비교 | 12 | 0.809 | 0.5685 | 0.8603 | 0.6846 | 0.9094 |
| all_features | 합의 기준에 따라 fnlwgt·민감 변수 포함 | 13 | 0.8081 | 0.567 | 0.8603 | 0.6836 | 0.9096 |
| without_sensitive | race·sex 제외 안정성 비교 | 11 | 0.8079 | 0.5669 | 0.8597 | 0.6832 | 0.9093 |

| 최종 평가 지표 | 값 |
| --- | ---: |
| Accuracy | 0.8090 |
| Precision | 0.5685 |
| Recall | 0.8603 |
| F1 | 0.6846 |
| ROC-AUC | 0.9094 |

- Confusion Matrix: [[3916, 1024], [219, 1349]]
- [Pipeline과 컬럼 bundle](adult_income_pipeline.joblib)
- [모델 메타데이터](adult_income_model_metadata.json)

fnlwgt 포함·제외 모델을 F1과 설명 가능성으로 비교했습니다. race·sex 제외 모델은 민감 변수 제거 시 성능 안정성을 확인하기 위한 감사용 비교입니다.

## 9. 윤리적 한계

Adult 데이터는 과거 사회·노동 구조의 편향을 포함할 수 있습니다. race와 sex를 포함한 모델의 예측을 실제 의사결정에 바로 사용해서는 안 되며, 집단별 오류율과 공정성 검토가 추가로 필요합니다. fnlwgt는 개인 특성이라기보다 표본 가중치이므로 최종 선택 시 성능과 설명 가능성을 함께 고려했습니다.

## 10. 발표 요약

1. adult.data를 두 엔진으로 읽어 로딩 결과·결측·중복·groupby·시간·메모리를 검증했습니다.
2. 극단값과 희소 범주를 보존한 채 수치·범주 EDA와 정적·인터랙티브 시각화를 수행했습니다.
3. 다중 평균 검정과 카이제곱 검정에 효과 크기·신뢰구간·연관성 크기를 함께 제시했습니다.
4. split 이후 train에서만 전처리 기준을 학습하고 세 Logistic Pipeline을 동일 test에서 비교했습니다.
5. 최종 Pipeline, 원본 컬럼 목록, 타깃 매핑과 메타데이터를 함께 저장했습니다.
