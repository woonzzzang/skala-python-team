# Adult Census Income 기준 반영 체크리스트

| 번호 | 기준 | 구현 상태 |
| ---: | --- | --- |
| 1 | End-to-End 분석 | ✅ 준비→EDA→통계→모델→저장→보고서 연결 |
| 2 | adult.data만 사용 | ✅ 단일 파일을 80:20 train/test로 분할 |
| 3 | 원본 컬럼명 | ✅ education-num 등 하이픈 이름 유지 |
| 4 | 물음표 결측 인식 | ✅ 로딩 시 null 처리 |
| 5 | 수치형 결측 | ✅ 존재 여부 출력, Pipeline 중앙값 안전망 |
| 6 | 범주형 결측 | ✅ Pipeline 최빈값 대체 |
| 7 | 결측 처리 시점 | ✅ 대체기는 split 후 train에서만 fit |
| 8 | 결측 행 미삭제 | ✅ 중복 외 행 삭제 없음 |
| 9 | 결측률 | ✅ 개수와 비율을 EDA 표에 기록 |
| 10 | 완전 중복 제거 | ✅ 24건 제거, 원본 파일 보존 |
| 11 | 이상치 탐지 | ✅ IQR 후보 수만 진단 |
| 12 | 이상치 유지 | ✅ clip·삭제·로그 대체 없음 |
| 13 | age 유지 | ✅ 실제 관측값 보존 |
| 14 | hours-per-week 유지 | ✅ 실제 관측값 보존 |
| 15 | capital-gain/loss 유지 | ✅ 원본 보존, 시각화만 log1p |
| 16 | fnlwgt | ✅ 포함·제외 모델 성능 비교 |
| 17 | education 역할 | ✅ EDA는 education, 모델은 education-num |
| 18 | income 문자열 | ✅ EDA 원본 유지, 모델링 복사본만 0·1 매핑 |
| 19 | 민감 변수 | ✅ 기본·선택 대상 모델에 포함, 제외 모델은 감사용 |
| 20 | 희소 범주 | ✅ 표시만 하고 모든 범주 유지 |
| 21 | Pandas·Polars | ✅ 로드·결측·중복·groupby·시간·메모리 비교 |
| 22 | 수치형 EDA | ✅ 기술통계·분위수·왜도·첨도·결측·0·IQR·소득별 비교 |
| 23 | 범주형 EDA | ✅ 빈도·비율·결측·희소·고소득 수와 비율 |
| 24 | 시각화 유형 | ✅ 히스토그램·KDE·박스·count·비율·heatmap·scatter·CM·ROC |
| 25 | 정적 2×2 | ✅ 요약·그룹 비교·수치 분포 3개 세트 |
| 26 | Plotly | ✅ 막대 2·산점도·facet·treemap·parallel 6개 |
| 27 | 범주 계산 기준 | ✅ 인원·고소득 수·그룹 비율·전체 고소득 점유율 |
| 28 | 상관분석 | ✅ 수치+income 0·1 Pearson·Spearman, 인과 주의 |
| 29 | t-test 주제 | ✅ 제시된 5개 주제 모두 실행 |
| 30 | t-test 선택 | ✅ 정규성·Levene 후 Student/Welch, MW 보조 |
| 31 | 검정 방향 | ✅ 사전 정의한 양측 검정 |
| 32 | 유의수준 | ✅ 0.05 중심, 0.01·0.10 플래그 병기 |
| 33 | p-value 표현 | ✅ 기각 또는 기각 근거 부족으로 표현 |
| 34 | 효과 크기 | ✅ 평균차·95% CI·Cohen's d와 p-value 병기 |
| 35 | 카이제곱 | ✅ 제시된 7개 범주와 income, Cramer's V 포함 |
| 36 | 데이터 분할 | ✅ 80:20, random_state=42, stratify |
| 37 | 기본 모델 | ✅ Logistic Regression |
| 38 | 불균형 처리 | ✅ class_weight=balanced |
| 39 | 수치 스케일 | ✅ StandardScaler |
| 40 | 인코딩 | ✅ feature OneHot, target 복사본 0·1 |
| 41 | 평가 지표 | ✅ Accuracy·Precision·Recall·F1·ROC-AUC·CM |
| 42 | 최종 모델 선택 | ✅ F1·설명 가능성 기준: without_fnlwgt |
| 43 | 모델 저장 | ✅ Pipeline·컬럼·타깃 매핑·메타데이터 bundle |
