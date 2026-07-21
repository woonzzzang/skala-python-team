# data/ 폴더 설명

## raw/adult.data
- UCI Adult Census Income 원본 데이터. 헤더 없음, 결측값은 `?`로 표시됨.
- 로컬에 이미 있으면 그대로 사용하고, 없으면 `src/data_loader.py`의 `download_or_load_data()`가
  UCI 저장소 URL에서 새로 내려받는다.
- 원본 그대로 보존하며, 이 프로젝트의 어떤 단계에서도 직접 수정하지 않는다.

## processed/adult_cleaned.csv
- `src/preprocessing.py`의 `clean_for_eda()`가 만드는 정제 데이터(`src/main.py` 실행 시 자동 생성/갱신).
- 여기서 하는 정제는 문자열 공백 제거, income 표기 통일(`<=50K`/`>50K`), 완전 중복 행 제거뿐이다.
- **결측치 대체(imputation)는 이 단계에서 하지 않는다.** 결측치 대체는 `train_test_split` 이후
  `sklearn Pipeline` 내부에서 train 데이터 기준으로만 계산해, 데이터 누수를 막는다(`src/modeling.py` 참고).

두 파일 모두 `.gitignore`에 포함되어 있어 기본적으로 Git에는 올라가지 않는다(원본은 URL에서, 정제본은
코드 실행으로 언제든 다시 만들 수 있기 때문). 자세한 이유는 루트 `README.md`의 "GitHub 업로드 시
주의사항"을 참고한다.
