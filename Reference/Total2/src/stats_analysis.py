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
    """
    숫자형 컬럼 간 상관계수를 계산한다. 수업 기준에 맞춰 Pearson을 공식 분석 결과로 사용하고,
    Spearman/Kendall은 참고용으로 함께 계산해 Pearson과 얼마나 차이나는지 비교한다.
    income은 범주형이라 기본 상관행렬에서 제외한다.
    """
    correlation_matrix = df[numeric_columns].corr(method="pearson")

    print("\n[숫자형 컬럼 상관관계 (Pearson, 공식 분석 기준)]")
    print(correlation_matrix.round(2))
    print("\n주의: 상관관계가 있다고 해서 인과관계가 있는 것은 아니다.")

    top_pairs = get_top_correlation_pairs(correlation_matrix, top_n=3)
    print("\n[상관계수 절대값 기준 상위 3개 변수 조합 (Pearson)]")
    print(top_pairs.to_string(index=False))

    # 참고용: Spearman(순위 기반 단조 관계), Kendall(순위 기반, 표본이 작을 때 유리)
    spearman_matrix = df[numeric_columns].corr(method="spearman")
    kendall_matrix = df[numeric_columns].corr(method="kendall")

    pearson_vs_spearman_diff = (correlation_matrix - spearman_matrix).abs()
    max_diff_value = pearson_vs_spearman_diff.where(
        ~np.eye(len(numeric_columns), dtype=bool)
    ).max().max()

    print("\n[참고] Spearman 상관관계 (순위 기반 단조 관계)")
    print(spearman_matrix.round(2))
    print("\n[참고] Kendall 상관관계 (순위 기반, 표본이 작을 때도 안정적)")
    print(kendall_matrix.round(2))
    print(
        f"\n[참고] Pearson과 Spearman의 상관계수 절대값 차이 중 최대값: {max_diff_value:.3f} "
        "(차이가 크면 선형이 아닌 비선형 단조 관계가 있을 수 있다는 신호)"
    )

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


def check_group_test_assumptions(
    group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05, shapiro_sample_size: int = 5000
) -> dict:
    """
    두 그룹 비교 검정 전에 정규성(Shapiro-Wilk)과 등분산성(Levene)을 확인한다.
    Shapiro-Wilk는 표본이 매우 크면(수만 건) 미세한 차이에도 정규성을 기각하기 쉽고
    scipy 구현상 안정적으로 계산하기 어려워, 그룹 크기가 shapiro_sample_size를 넘으면
    해당 크기만큼 무작위 샘플링해서 검정한다(참고용 판단이지 엄밀한 전수 검정은 아님).
    """

    def sample_for_shapiro(values: pd.Series) -> pd.Series:
        if len(values) > shapiro_sample_size:
            return values.sample(n=shapiro_sample_size, random_state=42)
        return values

    _, p_normal_a = stats.shapiro(sample_for_shapiro(group_a))
    _, p_normal_b = stats.shapiro(sample_for_shapiro(group_b))
    _, p_levene = stats.levene(group_a, group_b)

    return {
        "p_normal_a": float(p_normal_a),
        "p_normal_b": float(p_normal_b),
        "is_normal": bool(p_normal_a >= alpha and p_normal_b >= alpha),
        "p_levene": float(p_levene),
        "is_equal_variance": bool(p_levene >= alpha),
    }


def perform_two_group_test(
    df: pd.DataFrame,
    group_column: str,
    group_a_label: str,
    group_b_label: str,
    value_column: str,
    alpha: float = 0.05,
) -> dict:
    """
    두 그룹의 value_column 차이를 검정한다(양측 검정).
    정규성/등분산성을 먼저 확인해서 검정 방법을 자동으로 고른다:
      - 정규성을 만족하지 않으면 -> Mann-Whitney U(비모수)
      - 정규성은 만족하지만 분산이 다르면 -> Welch t-test
      - 정규성도 만족하고 분산도 같으면 -> Student t-test
    참고: 대응표본(paired) t-test는 같은 대상의 사전/사후 측정치가 있어야 하는데,
    이 데이터셋은 개인별 반복 측정이 없는 횡단면(cross-sectional) 데이터라 적용 대상이 아니다.
    """
    group_a = df.loc[df[group_column] == group_a_label, value_column].dropna()
    group_b = df.loc[df[group_column] == group_b_label, value_column].dropna()

    assumptions = check_group_test_assumptions(group_a, group_b, alpha=alpha)

    print(f"\n[t-test] {group_column}: {group_a_label} vs {group_b_label} ({value_column} 비교)")
    print(f"H0: 두 그룹의 평균(또는 분포)에는 차이가 없다. / H1: 차이가 있다.")
    print(
        f"정규성(Shapiro) p-value: {group_a_label}={assumptions['p_normal_a']:.4f}, "
        f"{group_b_label}={assumptions['p_normal_b']:.4f} / 등분산성(Levene) p-value: {assumptions['p_levene']:.4f}"
    )

    if not assumptions["is_normal"]:
        test_type = "Mann-Whitney U"
        statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    elif assumptions["is_equal_variance"]:
        test_type = "Student t-test"
        statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=True)
    else:
        test_type = "Welch t-test"
        statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    mean_difference = group_a.mean() - group_b.mean()
    pooled_std = np.sqrt((group_a.std() ** 2 + group_b.std() ** 2) / 2)
    cohens_d = mean_difference / pooled_std if pooled_std else 0.0
    is_significant = p_value < alpha

    print(f"선택된 검정: {test_type}")
    print(f"{group_a_label}: n={len(group_a)}, 평균={group_a.mean():.2f}, 표준편차={group_a.std():.2f}")
    print(f"{group_b_label}: n={len(group_b)}, 평균={group_b.mean():.2f}, 표준편차={group_b.std():.2f}")
    print(f"통계량: {statistic:.4f}, p-value: {p_value:.6f}")
    print(f"평균 차이: {mean_difference:.2f}, Cohen's d: {cohens_d:.3f}")

    if is_significant:
        print(f"해석: p < {alpha} 이므로 귀무가설을 기각한다. 두 그룹의 {value_column}에는 통계적으로 유의한 차이가 있다.")
    else:
        print(f"해석: p >= {alpha} 이므로 귀무가설을 기각할 충분한 근거가 없다.")

    return {
        "group_column": group_column,
        "group_a_label": group_a_label,
        "group_b_label": group_b_label,
        "value_column": value_column,
        "test_type": test_type,
        "group_a_n": int(len(group_a)),
        "group_b_n": int(len(group_b)),
        "group_a_mean": float(group_a.mean()),
        "group_b_mean": float(group_b.mean()),
        "mean_difference": float(mean_difference),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "alpha": alpha,
        "is_significant": bool(is_significant),
        "assumptions": assumptions,
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
