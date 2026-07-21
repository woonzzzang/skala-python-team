# src/ 모듈 설명

`python src/main.py`로 실행하면 아래 모듈이 이 순서대로 호출된다. 각 함수의 자세한 동작은
파일 안의 docstring과 주석을 참고한다. 프로젝트 전체 설명은 루트 `README.md`를 참고한다.

| 파일 | 역할 |
|---|---|
| `main.py` | 전체 파이프라인 실행 진입점. 컬럼/feature 목록 정의, 각 모듈 호출 순서 관리, report.md용 context 조립 |
| `data_loader.py` | `adult.data` 다운로드/로드, Pandas·Polars 결과 비교, 로드 시간·메모리 사용량 비교 |
| `preprocessing.py` | 기본 점검(shape/결측치/중복), 결측치 처리 방법론 설명, EDA·모델 공통 정제(`clean_for_eda`) |
| `eda.py` | 수치형(`perform_numeric_eda`, `summarize_numeric_by_income`)/범주형(`perform_categorical_eda`)/타깃(`analyze_income_distribution`) EDA |
| `stats_analysis.py` | Pearson/Spearman 상관분석, Welch t-test(`perform_t_test`), 카이제곱 검정 + Cramér's V(`perform_chi_square_test`) |
| `visualization.py` | Seaborn/Matplotlib 정적 차트, Plotly 인터랙티브 차트 생성 함수 모음 |
| `modeling.py` | 전처리 Pipeline 구성(`build_ml_pipeline`), LogisticRegression vs RandomForest 비교(`compare_classifiers`), 그룹별 성능 참고 분석(`analyze_group_performance`), 모델 저장/재로딩(`save_and_verify_model`) |
| `report.py` | 실행 결과(context)를 받아 `output/report.md`를 자동 생성 |

## 이름 관련 참고

통계 모듈 이름은 원래 계획이었던 `statistics.py`가 아니라 `stats_analysis.py`다. `python src/main.py`처럼
스크립트를 직접 실행하면 `src/` 폴더가 `sys.path` 맨 앞에 놓이는데, 이때 `statistics.py`라는 이름을 쓰면
파이썬 표준 라이브러리 `statistics` 모듈을 가려버려 seaborn 같은 라이브러리의 import가 깨지는 문제가
실제로 발생했다. 이름 충돌을 피하기 위해 `stats_analysis.py`로 정했다.

## 실행 방법

루트(`Total2/`) 기준으로 실행한다.

```bash
python src/main.py
```
