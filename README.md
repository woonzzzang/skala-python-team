# Day2 종합 실습 - End2End 데이터 분석 프로젝트 수행 계획

## 1. 실습 목표

`data.py`에 제공된 UCI Adult Income 데이터를 사용해 데이터 로딩부터 EDA, 시각화, 통계 분석, ML Pipeline, 결과 리포트 자동 생성까지 한 번에 수행하는 End2End 분석 프로젝트입니다.

팀별 회의 전에 각자 전체 흐름을 먼저 수행하고, 결과와 개선점을 바탕으로 발표 내용을 정리하는 것이 목표입니다.

## 2. 사용 데이터

| 항목 | 내용 |
| --- | --- |
| 데이터 | UCI Adult Income |
| 제공 파일 | `data.py` |
| 원본 URL | `https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data` |
| 목적 변수 | `income` |
| 분석 목표 | 소득 구간(`<=50K`, `>50K`)과 인구통계/근로 특성의 관계 분석 및 예측 모델 구성 |

## 3. 산출 파일

| 파일/폴더 | 내용 |
| --- | --- |
| `main.py` | 전체 분석 실행 코드 |
| `report.md` | 실행 결과 자동 생성 리포트 |
| `data/adult_raw.data` | 원본 데이터 캐시 |
| `outputs/cleaned_adult.csv` | 정제 후 데이터 |
| `outputs/eda_summary.json` | EDA 요약 수치 |
| `assets/seaborn_eda.png` | Seaborn 정적 EDA 차트 |
| `assets/plotly_income_by_education.html` | Plotly 인터랙티브 차트 |
| `models/income_pipeline.joblib` | 저장된 sklearn Pipeline 모델 |

## 4. 수행 단계

### 4.1 데이터 준비

해야 할 일:

- `data.py`에 제공된 URL과 컬럼 구조를 기준으로 데이터 로딩
- 원본 데이터를 로컬에 캐시해 반복 실행 안정성 확보
- Pandas와 Polars 양쪽으로 로딩
- 두 도구의 행/열 개수 비교
- 결측치 개수, 중복 개수, 기본 EDA 출력

지켜야 할 점:

- Pandas만 사용하지 않고 Polars 로딩 결과도 함께 비교
- 결측치 처리 전/후 개수 출력
- 중복 제거 전/후 행 수 출력
- 범주형 결측치는 임의 추정하지 않고 `Unknown`으로 대체

### 4.2 시각화

해야 할 일:

- Seaborn 정적 차트 1개 이상 생성
- Plotly 인터랙티브 차트 1개 이상 생성
- 제목, 축 라벨 포함
- 이미지/HTML 파일 저장

선택한 시각화:

- Seaborn 2x2 EDA 차트
  - 소득 구간 분포
  - 나이 분포
  - 주당 근무시간 박스플롯
  - 수치형 변수 상관 히트맵
- Plotly 인터랙티브 차트
  - 교육 수준별 고소득 비율

지켜야 할 점:

- `plt.show()`만 하지 않고 이미지 파일 저장
- Plotly는 `.write_html()`로 HTML 저장
- 제목과 축 라벨 작성

### 4.3 통계 분석

해야 할 일:

- 기술통계 산출
  - 평균
  - 표준편차
  - 분위수
- 수치형 변수 간 상관계수 계산
- `scipy.stats.ttest_ind`로 t-test 수행
- p-value 기준 해석 출력

선택한 t-test:

- `income <=50K` 그룹과 `income >50K` 그룹의 `hours-per-week` 평균 차이 검정

지켜야 할 점:

- t통계량과 p-value 모두 출력
- `p < 0.05` 기준으로 유의미 여부 해석
- 수치만 출력하고 해석을 빠뜨리지 않기

### 4.4 ML Pipeline

해야 할 일:

- `sklearn.pipeline.Pipeline`으로 전처리와 모델을 하나로 구성
- `ColumnTransformer`로 수치형/범주형 전처리 분리
- 모델 학습, 예측, 평가
- 정확도와 F1-score 출력
- `joblib`으로 모델 저장
- 저장된 모델 재로딩 확인

모델 구성:

- 수치형 컬럼
  - 결측치: median
  - 스케일링: `StandardScaler`
- 범주형 컬럼
  - 결측치: most_frequent
  - 인코딩: `OneHotEncoder`
- 모델
  - `LogisticRegression`

지켜야 할 점:

- 전처리와 모델을 따로 실행하지 않고 Pipeline 객체로 묶기
- 평가 지표는 정확도와 F1-score 포함
- 모델 파일 저장 누락 금지

### 4.5 자동화 및 report 자료

해야 할 일:

- 분석 결과를 `report.md`로 자동 생성
- 실행 결과 요약 포함
- 생성된 차트 파일 경로 포함
- 모델 평가 결과 포함
- 핵심 해석 정리

지켜야 할 점:

- 수동으로 결과를 옮기지 않고 코드 실행으로 리포트 생성
- 발표에서 말할 수 있는 해석 문장 포함
- 개선점과 추가 분석 아이디어 포함

## 5. 평가 기준 매핑

| 평가 항목 | 대응 구현 |
| --- | --- |
| 데이터 준비 + 시각화 | Pandas/Polars 로딩 비교, 결측/중복 처리, Seaborn/Plotly 저장 |
| 통계 분석 + ML Pipeline | 기술통계, 상관계수, t-test, sklearn Pipeline, accuracy/F1 출력 |
| 자동화 + 발표 | `report.md` 자동 생성, 발표용 요약 포함 |
| 완성도 | 함수 분리, 주석 작성, 파일 저장, 재실행 가능 구조 |

## 6. 실행 명령

```bash
uv run python main.py
```

## 7. 팀 회의 때 확인할 질문

- 결측치를 `Unknown`으로 처리한 것이 적절한가?
- 예측 목표를 소득 구간으로 둔 것이 발표에 가장 적합한가?
- t-test 변수로 `hours-per-week`를 선택한 근거가 충분한가?
- 모델 지표 중 accuracy와 F1 중 어떤 지표를 더 강조할 것인가?
- 추가로 연령대, 교육 수준, 직업군별 해석을 넣을 것인가?
