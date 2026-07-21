"""
기술통계·상관분석·t-test·카이제곱 검정을 담당하는 모듈.

주의: 이 파일의 이름은 원래 계획인 statistics.py가 아니라 stats_analysis.py다.
src/ 디렉터리에서 스크립트를 직접 실행하면 해당 디렉터리가 sys.path 맨 앞에 추가되는데,
이때 statistics.py라는 이름을 쓰면 파이썬 표준 라이브러리 statistics 모듈을 가려버려서
seaborn 등 표준 라이브러리 statistics를 사용하는 패키지의 import가 깨진다.
그래서 이름 충돌을 피하기 위해 stats_analysis.py로 정했다.
"""

import numpy as np
import pandas as pd
from scipy import stats

from eda import top_n_with_other


def get_top_correlation_pairs(correlation_matrix: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """상관계수 절대값 기준 상위 top_n개 변수 조합을 표로 반환한다(콘솔 출력과 report.md에서 재사용)."""
    columns = correlation_matrix.columns
    pairs = []
    for i, col_i in enumerate(columns):
        for col_j in columns[i + 1 :]:
            pairs.append((col_i, col_j, correlation_matrix.loc[col_i, col_j]))
    pairs_df = pd.DataFrame(pairs, columns=["변수1", "변수2", "상관계수"])
    pairs_df["절대값"] = pairs_df["상관계수"].abs()
    return pairs_df.sort_values("절대값", ascending=False).head(top_n)[["변수1", "변수2", "상관계수"]].round(3)


def perform_correlation_analysis(
    df: pd.DataFrame, numeric_columns: list[str], income_column: str = "income"
) -> pd.DataFrame:
    """숫자형 컬럼 간 Pearson 상관계수를 계산한다. income은 범주형이라 기본 상관행렬에서 제외한다."""
    correlation_matrix = df[numeric_columns].corr(method="pearson")

    print("\n[숫자형 컬럼 상관관계 (Pearson)]")
    print(correlation_matrix.round(2))
    print("\n주의: 상관관계가 있다고 해서 인과관계가 있는 것은 아니다.")

    top_pairs = get_top_correlation_pairs(correlation_matrix, top_n=3)
    print("\n[상관계수 절대값 기준 상위 3개 변수 조합]")
    print(top_pairs.to_string(index=False))

    # income을 0/1로 임시 인코딩한 참고용 상관관계 (공식 분석이 아닌 단순 선형 상관 참고값)
    income_encoded = (df[income_column] == ">50K").astype(int)
    income_correlation = df[numeric_columns].apply(lambda col: col.corr(income_encoded)).round(3)
    print("\n[참고] income을 <=50K=0, >50K=1로 임시 인코딩했을 때의 단순 선형 상관 참고값")
    print(income_correlation)

    print(
        "\n참고: education과 education-num은 같은 학력 정보를 다른 형태로 표현한 값이라 "
        "서로 상관이 매우 높을 수 있으므로, 모델 feature로 함께 쓸 때는 정보 중복을 인지하고 있어야 한다."
    )

    return correlation_matrix


def perform_t_test(
    df: pd.DataFrame,
    group_column: str = "income",
    value_column: str = "hours-per-week",
    alpha: float = 0.05,
) -> dict:
    """income 그룹(>50K vs <=50K)별 hours-per-week 평균 차이를 Welch t-test로 검정한다."""
    high_income_group = df.loc[df[group_column] == ">50K", value_column].dropna()
    low_income_group = df.loc[df[group_column] == "<=50K", value_column].dropna()

    print(f"\n[t-test] {group_column} 그룹별 {value_column} 평균 비교")
    print("H0: 두 그룹의 평균 hours-per-week에는 차이가 없다.")
    print("H1: 두 그룹의 평균 hours-per-week에는 차이가 있다.")
    print(
        f">50K  : n={len(high_income_group)}, 평균={high_income_group.mean():.2f}, "
        f"표준편차={high_income_group.std():.2f}"
    )
    print(
        f"<=50K : n={len(low_income_group)}, 평균={low_income_group.mean():.2f}, "
        f"표준편차={low_income_group.std():.2f}"
    )

    # Welch t-test: 두 그룹의 분산이 같다고 가정하지 않는 양측 검정
    t_statistic, p_value = stats.ttest_ind(high_income_group, low_income_group, equal_var=False)

    # Cohen's d: 두 그룹 표준편차를 합친 값 기준의 효과 크기(참고용, 통계적 유의성과는 별개)
    pooled_std = np.sqrt((high_income_group.std() ** 2 + low_income_group.std() ** 2) / 2)
    mean_difference = high_income_group.mean() - low_income_group.mean()
    cohens_d = mean_difference / pooled_std

    is_significant = p_value < alpha

    print(f"t 통계량: {t_statistic:.4f}, p-value: {p_value:.6f}")
    print(f"평균 차이: {mean_difference:.2f}, Cohen's d: {cohens_d:.3f}")

    if is_significant:
        print(f"해석: p < {alpha} 이므로 귀무가설을 기각한다. 두 그룹의 평균 주당 근무시간은 통계적으로 유의하게 다르다.")
    else:
        print(f"해석: p >= {alpha} 이므로 귀무가설을 기각할 충분한 근거가 없다. 통계적으로 유의한 차이를 확인하지 못했다.")

    print(
        "참고: 표본 수가 매우 크면 실제로는 작은 차이도 아주 작은 p-value로 나올 수 있어, "
        "통계적 유의성만 볼 것이 아니라 평균 차이와 Cohen's d 같은 효과 크기도 함께 봐야 한다."
    )

    return {
        "group_high_n": int(len(high_income_group)),
        "group_low_n": int(len(low_income_group)),
        "group_high_mean": float(high_income_group.mean()),
        "group_low_mean": float(low_income_group.mean()),
        "mean_difference": float(mean_difference),
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "alpha": alpha,
        "is_significant": bool(is_significant),
    }


def perform_chi_square_test(df: pd.DataFrame, row_column: str, col_column: str, top_n: int = 10) -> dict:
    """
    두 범주형 컬럼의 독립성을 카이제곱 검정으로 확인한다(t-test 외 선택적 추가 분석).
    범주가 top_n개보다 많으면 나머지를 'Other'로 묶어 기대빈도가 지나치게 작아지는 것을 완화한다.
    """
    working_df = df[[row_column, col_column]].dropna()

    row_series = working_df[row_column]
    if row_series.nunique() > top_n:
        row_series = top_n_with_other(row_series, n=top_n)

    crosstab = pd.crosstab(row_series, working_df[col_column])
    chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = stats.chi2_contingency(crosstab)

    small_expected_cell_count = int((expected_frequencies < 5).sum())
    alpha = 0.05
    is_significant = p_value < alpha

    print(f"\n[카이제곱 독립성 검정] {row_column} x {col_column}")
    print(f"chi2={chi2_statistic:.4f}, dof={degrees_of_freedom}, p-value={p_value:.6f}")
    print(f"기대빈도 5 미만 셀 개수: {small_expected_cell_count} / 전체 {expected_frequencies.size}")
    if small_expected_cell_count > 0:
        print("주의: 기대빈도가 낮은 셀이 있어 검정 결과의 신뢰도가 다소 낮아질 수 있다.")

    if is_significant:
        print(f"해석: p < {alpha} 이므로 {row_column}와 {col_column}는 통계적으로 유의한 연관성이 있다.")
    else:
        print(f"해석: p >= {alpha} 이므로 {row_column}와 {col_column} 사이의 유의한 연관성을 확인하지 못했다.")

    return {
        "row_column": row_column,
        "col_column": col_column,
        "chi2": float(chi2_statistic),
        "dof": int(degrees_of_freedom),
        "p_value": float(p_value),
        "is_significant": bool(is_significant),
    }
