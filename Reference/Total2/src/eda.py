"""수치형/범주형/타깃 변수에 대한 자료형별 EDA를 담당하는 모듈."""

import pandas as pd


def perform_numeric_eda(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """수치형 컬럼별 기술통계(사분위수, IQR, 왜도, 첨도, 이상치 후보 수 포함)를 계산하고 해석 힌트를 출력한다."""
    summary_rows = []
    for col in numeric_columns:
        valid_values = df[col].dropna()
        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_candidate_count = int(((valid_values < lower_bound) | (valid_values > upper_bound)).sum())
        missing_count = int(df[col].isna().sum())

        summary_rows.append(
            {
                "column": col,
                "count": valid_values.count(),
                "mean": valid_values.mean(),
                "std": valid_values.std(),
                "min": valid_values.min(),
                "Q1": q1,
                "median": valid_values.median(),
                "Q3": q3,
                "max": valid_values.max(),
                "IQR": iqr,
                "missing": missing_count,
                "missing_pct": missing_count / len(df) * 100,
                "unique": valid_values.nunique(),
                "skewness": valid_values.skew(),
                "kurtosis": valid_values.kurtosis(),
                "outlier_candidates": outlier_candidate_count,
            }
        )
    summary_df = pd.DataFrame(summary_rows).set_index("column").round(2)

    print("\n[수치형 컬럼 기술통계]")
    print(summary_df)
    print(
        "참고: outlier_candidates는 IQR*1.5 기준 통계적 이상치 '후보' 개수일 뿐, "
        "실제 데이터 오류를 의미하지 않으므로 자동으로 삭제하지 않는다."
    )

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
            f"왜도={row['skewness']:.2f}({skew_desc}), 0값 비율={zero_ratio:.1f}%, "
            f"IQR 이상치 후보={int(row['outlier_candidates'])}건"
        )

    return summary_df


def summarize_numeric_by_income(
    df: pd.DataFrame, numeric_columns: list[str], income_column: str = "income"
) -> pd.DataFrame:
    """수치형 컬럼별로 income 그룹(<=50K, >50K)의 평균/중앙값을 비교한다."""
    grouped = df.groupby(income_column)[numeric_columns].agg(["mean", "median"]).round(2)

    print("\n[income 그룹별 수치형 컬럼 평균/중앙값]")
    print(grouped)

    return grouped


def top_n_with_other(series: pd.Series, n: int = 10, other_label: str = "Other") -> pd.Series:
    """고유값이 많은 범주형 컬럼에서 상위 n개만 남기고 나머지는 other_label로 묶는다(시각화/검정용)."""
    top_categories = series.value_counts().head(n).index
    return series.where(series.isin(top_categories), other_label)


def perform_categorical_eda(
    df: pd.DataFrame, categorical_columns: list[str], top_n: int = 10, income_column: str = "income"
) -> None:
    """
    범주형 컬럼별 고유값 개수, 최빈값, 상위 범주 비율, 결측치, 희소 범주 여부를 출력한다.
    income_column과 다른 컬럼이면, 범주별 >50K 인원 수와 비율도 함께 보여준다
    (인원 수와 그룹 내 고소득 비율을 구분해서 표시).
    """
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

        if col != income_column and income_column in df.columns:
            high_income_count = df.loc[df[col].isin(value_counts.index), [col, income_column]]
            high_income_count = (
                high_income_count.groupby(col)[income_column].apply(lambda s: (s == ">50K").sum())
            )
            print(f"    {'범주':<20}{'인원 수':>10}{'비율':>8}{'>50K 인원':>12}{'>50K 비율':>12}")
            for category, count in value_counts.items():
                high_count = int(high_income_count.get(category, 0))
                high_ratio = high_count / count * 100 if count else 0.0
                print(
                    f"    {str(category):<20}{count:>10}{value_ratios[category]:>7.2f}%"
                    f"{high_count:>12}{high_ratio:>11.2f}%"
                )
        else:
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
