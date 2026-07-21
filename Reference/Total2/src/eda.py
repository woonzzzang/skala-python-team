"""수치형/범주형/타깃 변수에 대한 자료형별 EDA를 담당하는 모듈."""

import pandas as pd


def perform_numeric_eda(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """수치형 컬럼별 기술통계(사분위수, 왜도, 첨도 포함)를 계산하고 해석 힌트를 출력한다."""
    summary_rows = []
    for col in numeric_columns:
        valid_values = df[col].dropna()
        summary_rows.append(
            {
                "column": col,
                "count": valid_values.count(),
                "mean": valid_values.mean(),
                "std": valid_values.std(),
                "min": valid_values.min(),
                "Q1": valid_values.quantile(0.25),
                "median": valid_values.median(),
                "Q3": valid_values.quantile(0.75),
                "max": valid_values.max(),
                "missing": df[col].isna().sum(),
                "skewness": valid_values.skew(),
                "kurtosis": valid_values.kurtosis(),
            }
        )
    summary_df = pd.DataFrame(summary_rows).set_index("column").round(2)

    print("\n[수치형 컬럼 기술통계]")
    print(summary_df)

    print("\n[해석 힌트]")
    for col in numeric_columns:
        row = summary_df.loc[col]
        mean_median_gap = row["mean"] - row["median"]
        if row["skewness"] > 0.5:
            skew_desc = "오른쪽으로 치우침(긴 꼬리)"
        elif row["skewness"] < -0.5:
            skew_desc = "왼쪽으로 치우침"
        else:
            skew_desc = "치우침이 크지 않음"
        zero_ratio = (df[col] == 0).mean() * 100
        print(
            f"{col}: 평균-중앙값 차이={mean_median_gap:.2f}, "
            f"왜도={row['skewness']:.2f}({skew_desc}), 0값 비율={zero_ratio:.1f}%"
        )

    return summary_df


def top_n_with_other(series: pd.Series, n: int = 10, other_label: str = "Other") -> pd.Series:
    """고유값이 많은 범주형 컬럼에서 상위 n개만 남기고 나머지는 other_label로 묶는다(시각화/검정용)."""
    top_categories = series.value_counts().head(n).index
    return series.where(series.isin(top_categories), other_label)


def perform_categorical_eda(df: pd.DataFrame, categorical_columns: list[str], top_n: int = 10) -> None:
    """범주형 컬럼별 고유값 개수, 최빈값, 상위 범주 비율, 결측치, 희소 범주 여부를 출력한다."""
    print("\n[범주형 컬럼 EDA]")
    for col in categorical_columns:
        valid_values = df[col].dropna()
        unique_count = valid_values.nunique()
        mode_value = valid_values.mode().iloc[0] if not valid_values.empty else None
        value_counts = valid_values.value_counts().head(top_n)
        value_ratios = (value_counts / len(valid_values) * 100).round(2)
        missing_count = int(df[col].isna().sum())

        all_ratios = valid_values.value_counts() / len(valid_values)
        sparse_category_count = int((all_ratios < 0.01).sum())

        print(f"\n- {col}")
        print(f"  고유값 개수: {unique_count} (상위 {min(top_n, unique_count)}개만 출력)")
        print(f"  최빈값: {mode_value}")
        print(f"  결측치 개수: {missing_count}")
        for category, count in value_counts.items():
            print(f"    {category}: {count}건 ({value_ratios[category]}%)")
        if unique_count > top_n:
            print(f"  참고: 고유 범주가 {top_n}개보다 많아 시각화에서는 나머지를 'Other'로 묶어서 표현한다.")
        if sparse_category_count > 0:
            print(f"  희소 범주(비율 1% 미만) 개수: {sparse_category_count}개")


def analyze_income_distribution(df: pd.DataFrame, income_column: str = "income") -> dict:
    """타깃(income) 분포와 클래스 불균형 여부를 확인한다."""
    counts = df[income_column].value_counts()
    ratios = (counts / len(df) * 100).round(2)

    print("\n[income 분포]")
    for category in counts.index:
        print(f"{category}: {counts[category]}건 ({ratios[category]}%)")

    imbalance_ratio = counts.max() / counts.min()
    print(f"클래스 비율(다수 클래스 / 소수 클래스): {imbalance_ratio:.2f}배")
    print(
        "클래스 불균형이 있으므로 Accuracy만 보면 >50K 클래스를 거의 맞히지 못해도 "
        "숫자가 높게 나올 수 있다. 따라서 Precision, Recall, F1-score를 함께 확인해야 한다."
    )

    return {
        "counts": counts.to_dict(),
        "ratios": ratios.to_dict(),
        "imbalance_ratio": float(imbalance_ratio),
    }
