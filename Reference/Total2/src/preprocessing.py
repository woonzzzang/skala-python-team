"""데이터 기본 점검, 정제, 결측치 처리 방법론을 다루는 모듈."""

import pandas as pd


def inspect_data(df: pd.DataFrame, key_missing_columns: list[str]) -> None:
    """shape, head, info, dtypes, 결측치, 중복 행 등 기본 정보를 출력한다."""
    print("\n[shape]")
    print(df.shape)

    print("\n[head]")
    print(df.head())

    print("\n[info]")
    df.info()

    print("\n[전체 결측치 개수]")
    print(df.isna().sum())

    print("\n[주요 결측 가능 컬럼의 결측률]")
    for col in key_missing_columns:
        missing_count = int(df[col].isna().sum())
        missing_rate = missing_count / len(df) * 100
        print(f"{col}: {missing_count}건 ({missing_rate:.2f}%)")

    duplicate_count = int(df.duplicated().sum())
    print(f"\n완전 중복 행 개수: {duplicate_count}건")


def explain_missing_value_strategy() -> str:
    """
    결측치 처리 방법론을 설명한다.
    이 텍스트는 콘솔 출력과 report.md 생성에 동일하게 재사용한다.
    """
    explanation = (
        "이 데이터셋의 결측치는 workclass, occupation, native-country "
        "세 범주형 컬럼에서만 발견된다. 결측이 발생한 정확한 이유(MCAR/MAR/MNAR 여부)는 "
        "데이터만으로 단정할 수 없으므로 원인을 임의로 확정하지 않는다.\n"
        "- 수치형 컬럼: 이상치의 영향을 덜 받는 중앙값(median)으로 대체\n"
        "- 범주형 컬럼: 최빈값(most_frequent)으로 대체하되, 결측 자체가 의미 있다면 "
        "'Unknown' 범주로 별도 처리하는 방법도 대안이 될 수 있음\n"
        "- 결측 행을 무조건 dropna()로 삭제하지 않고, 학습 데이터 보존을 위해 대체를 기본으로 선택\n"
        "- 단, 통계 검정처럼 특정 컬럼만 필요한 분석에서는 해당 컬럼의 결측 행만 일시적으로 제거\n"
        "- 전체 데이터에 미리 대체값을 채우면 train/test 분리 후에도 test 정보가 섞여 "
        "데이터 누수가 생길 수 있으므로, 실제 대체는 sklearn Pipeline 내부에서 "
        "train 데이터 기준으로만 수행한다."
    )
    print("\n[결측치 처리 방법론]")
    print(explanation)
    return explanation


def clean_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA와 모델 입력에 공통으로 사용할 정제 DataFrame을 만든다.
    여기서는 형식 정리(공백/중복/income 표기)만 수행하고
    결측치 대체는 하지 않는다 (대체는 sklearn Pipeline 내부에서 수행).
    """
    cleaned_df = df.copy()  # 원본 DataFrame은 그대로 보존하고 복사본으로 작업

    # 문자열 컬럼 앞뒤 공백 제거 (skipinitialspace로 대부분 처리되지만 방어적으로 재확인)
    string_columns = cleaned_df.select_dtypes(include="object").columns
    for col in string_columns:
        cleaned_df[col] = cleaned_df[col].str.strip()

    # income에 마침표가 붙어 있을 가능성까지 고려해 제거하고 두 범주로 통일
    cleaned_df["income"] = cleaned_df["income"].str.rstrip(".").str.strip()
    unexpected_income = set(cleaned_df["income"].dropna().unique()) - {"<=50K", ">50K"}
    if unexpected_income:
        print(f"경고: 예상하지 못한 income 값이 있습니다: {unexpected_income}")

    before_row_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
    after_row_count = len(cleaned_df)
    print(
        f"\n[중복 제거] 제거 전 {before_row_count}행 -> 제거 후 {after_row_count}행 "
        f"(제거된 행 {before_row_count - after_row_count}건)"
    )

    # 숫자형으로 기대되는 컬럼이 실제로 문자열로 잘못 읽히지 않았는지 확인
    numeric_columns = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
            raise TypeError(f"{col} 컬럼이 숫자형이 아닙니다: {cleaned_df[col].dtype}")

    print("\n[수치형 컬럼 최소/최대값] - 비정상적으로 작거나 큰 값이 있는지 점검용")
    for col in numeric_columns:
        print(f"{col}: min={cleaned_df[col].min()}, max={cleaned_df[col].max()}")

    print("\n[정제 후 결측치 개수] (아직 imputation은 적용하지 않은 상태)")
    print(cleaned_df.isna().sum())

    return cleaned_df
