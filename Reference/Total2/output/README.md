# output/ 폴더 설명

`python src/main.py`를 실행할 때마다 아래 파일이 새로 생성/갱신된다. 실제 수치는 매 실행 결과에 따라
달라질 수 있으며, 최신 실제 값은 `report.md`에서 확인한다.

## figures/ (정적 차트 PNG, Seaborn/Matplotlib)
| 파일 | 내용 |
|---|---|
| `income_distribution.png` | income(`<=50K`/`>50K`) 분포와 비율 |
| `numeric_eda.png` | age 히스토그램, hours-per-week 박스플롯, capital-gain/loss(log1p), fnlwgt 분포 2×2 |
| `hours_distribution.png` | hours-per-week 전체 분포(히스토그램) |
| `categorical_eda.png` | education/occupation/sex/workclass별 income 비율 2×2 |
| `correlation_heatmap.png` | 수치형 컬럼 Pearson 상관관계 히트맵 |
| `confusion_matrix.png` | 최종 선택 모델의 confusion matrix |
| `roc_curve.png` | 최종 선택 모델의 ROC curve (양성 클래스: `>50K`) |

## interactive/ (Plotly 인터랙티브 차트 HTML)
| 파일 | 내용 |
|---|---|
| `adult_income_analysis.html` | age × hours-per-week 산점도 (color=income) |
| `education_income_ratio.html` | 교육 수준별 고소득(`>50K`) 비율 막대그래프 |
| `occupation_income_ratio.html` | 직업별 고소득(`>50K`) 비율 막대그래프 |

## models/
| 파일 | 내용 |
|---|---|
| `adult_income_pipeline.pkl` | 전처리(`ColumnTransformer`)+분류기가 포함된 학습된 `sklearn.pipeline.Pipeline` 전체(`joblib.dump`로 저장) |
| `adult_income_pipeline.json` | 위 모델의 메타데이터: 입력 feature 목록(수치형/범주형), 제외된 컬럼(`fnlwgt`, `education`), 타깃 매핑(`{"<=50K":0, ">50K":1}`), 모델 하이퍼파라미터, 최종 모델 선택 이유, 저장 시각 |

`src/main.py`는 저장 직후 `joblib.load()`로 다시 불러와 저장 전/후 예측이 동일한지 확인한다.

## report.md
- 실행 결과(context)를 바탕으로 `src/report.py`가 자동 생성하는 최종 분석 보고서.
- 하드코딩된 숫자 없이, 그 실행에서 실제로 계산된 값만 채워 넣는다.
- 더 서술적인 분석·해석은 루트의 `보고서.md`에 있다. `report.md`는 실행할 때마다 갱신되는 "실제 실행
  결과 스냅샷", `보고서.md`는 그 결과를 사람이 다듬어 정리한 최종 보고서라는 차이가 있다.
