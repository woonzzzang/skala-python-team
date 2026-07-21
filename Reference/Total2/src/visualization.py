"""정적(Seaborn/Matplotlib) 시각화와 인터랙티브(Plotly) 시각화를 담당하는 모듈."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from eda import top_n_with_other


def configure_korean_font() -> None:
    """Mac 환경에서 한글이 깨지지 않도록 AppleGothic을 설정한다. 없으면 기본 폰트로 진행한다."""
    try:
        available_fonts = {font.name for font in fm.fontManager.ttflist}
        if "AppleGothic" in available_fonts:
            plt.rcParams["font.family"] = "AppleGothic"
        else:
            print("AppleGothic 폰트를 찾을 수 없어 기본 폰트로 진행합니다.")
    except Exception as error:
        print("한글 폰트 설정 중 문제가 발생했습니다:", error)

    plt.rcParams["axes.unicode_minus"] = False  # 음수 기호(-) 깨짐 방지


def create_income_distribution_plot(df: pd.DataFrame, output_path: Path) -> Path:
    """income 분포를 countplot으로 그리고 비율을 함께 표시한다."""
    fig, ax = plt.subplots(figsize=(6, 5))

    order = df["income"].value_counts().index
    sns.countplot(data=df, x="income", order=order, hue="income", legend=False, ax=ax)

    total = len(df)
    for patch in ax.patches:
        height = patch.get_height()
        ratio = height / total * 100
        ax.annotate(
            f"{int(height)}건 ({ratio:.1f}%)",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
        )

    ax.set_title("income 분포 (클래스 불균형 확인)")
    ax.set_xlabel("income")
    ax.set_ylabel("건수")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {output_path}")
    return output_path


def create_numeric_eda_plot(df: pd.DataFrame, output_path: Path) -> Path:
    """
    age, hours-per-week, capital-gain/loss, fnlwgt를 2x2 서브플롯 하나로 살펴본다.
    capital-gain/loss는 0이 매우 많고 오른쪽 꼬리가 길어 log1p로 눌러서 분포를 확인한다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1) age 분포 (income별 구분)
    sns.histplot(data=df, x="age", hue="income", kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("나이(age) 분포 - income별 구분")
    axes[0, 0].set_xlabel("age")
    axes[0, 0].set_ylabel("빈도")

    # 2) hours-per-week 박스플롯 (income별 비교)
    sns.boxplot(data=df, x="income", y="hours-per-week", hue="income", legend=False, ax=axes[0, 1])
    axes[0, 1].set_title("주당 근무시간 - income별 비교")
    axes[0, 1].set_xlabel("income")
    axes[0, 1].set_ylabel("hours-per-week")

    # 3) capital-gain / capital-loss: 0이 90% 이상이라 log1p로 압축해서 함께 비교
    capital_long = df[["capital-gain", "capital-loss"]].apply(np.log1p).melt(
        var_name="변수", value_name="log1p(값)"
    )
    sns.boxplot(data=capital_long, x="변수", y="log1p(값)", hue="변수", legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("capital-gain / capital-loss (log1p 변환)")
    axes[1, 0].set_xlabel("변수")
    axes[1, 0].set_ylabel("log1p(값)")

    # 4) fnlwgt: 인구총조사 가중치라서 큰 값이 나와도 측정 오류가 아님
    sns.histplot(df["fnlwgt"], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("fnlwgt 분포 (census 가중치, 큰 값이 오류는 아님)")
    axes[1, 1].set_xlabel("fnlwgt")
    axes[1, 1].set_ylabel("빈도")

    fig.suptitle("수치형 컬럼 EDA", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {output_path}")
    return output_path


def create_additional_distribution_plot(df: pd.DataFrame, output_path: Path) -> Path:
    """capital-gain, capital-loss, hours-per-week 각각의 원분포를 개별 히스토그램으로 살펴본다."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.histplot(df["capital-gain"], kde=True, ax=axes[0])
    axes[0].set_title("capital-gain 분포")
    axes[0].set_xlabel("capital-gain")
    axes[0].set_ylabel("빈도")

    sns.histplot(df["capital-loss"], kde=True, ax=axes[1])
    axes[1].set_title("capital-loss 분포")
    axes[1].set_xlabel("capital-loss")
    axes[1].set_ylabel("빈도")

    sns.histplot(df["hours-per-week"], kde=True, ax=axes[2])
    axes[2].set_title("hours-per-week 분포")
    axes[2].set_xlabel("hours-per-week")
    axes[2].set_ylabel("빈도")

    fig.suptitle("capital-gain / capital-loss / hours-per-week 원분포", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {output_path}")
    return output_path


def create_categorical_eda_plot(
    df: pd.DataFrame, output_path: Path, top_n: int = 10, min_group_size: int = 30
) -> Path:
    """
    education/occupation/workclass/sex별 income 비율을 2x2 서브플롯으로 비교한다.
    계산 기준: 그룹별 전체 인원 수, >50K 인원 수, 그룹 내 >50K 비율, 전체 >50K 중 그룹 비율을
    함께 계산해서 콘솔에 표로 출력하고, 표본 수가 min_group_size 미만인 그룹은 차트에서 제외한다.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    total_high_income = int((df["income"] == ">50K").sum())

    def plot_income_ratio(column: str, ax, cap_categories: bool) -> None:
        series = df[column]
        if cap_categories and series.nunique() > top_n:
            series = top_n_with_other(series, n=top_n)  # 범주가 많으면 상위 top_n + Other로 정리

        group_sizes = series.value_counts()
        small_groups = group_sizes[group_sizes < min_group_size].index.tolist()
        if small_groups:
            print(f"{column}: 표본 수 {min_group_size}건 미만이라 차트에서 제외한 그룹 -> {small_groups}")
        kept_categories = group_sizes[group_sizes >= min_group_size].index

        count_table = pd.crosstab(series, df["income"])
        count_table = count_table.loc[count_table.index.isin(kept_categories)]
        ratio_table = count_table.div(count_table.sum(axis=1), axis=0)
        sorted_order = group_sizes.loc[ratio_table.index].sort_values(ascending=False).index
        ratio_table = ratio_table.loc[sorted_order]
        count_table = count_table.loc[ratio_table.index]

        print(f"\n[{column}별 income 계산 기준]")
        for category in ratio_table.index:
            group_total = int(count_table.loc[category].sum())
            group_high_count = int(count_table.loc[category].get(">50K", 0))
            group_high_ratio = group_high_count / group_total * 100 if group_total else 0.0
            share_of_total_high = group_high_count / total_high_income * 100 if total_high_income else 0.0
            print(
                f"  {category}: 전체 {group_total}명, >50K {group_high_count}명 "
                f"(그룹 내 비율 {group_high_ratio:.2f}%, 전체 >50K 중 비율 {share_of_total_high:.2f}%)"
            )

        ratio_table.plot(kind="bar", stacked=True, ax=ax, legend=True)

        # 각 막대 구간에 인원 수와 비율을 함께 표시
        for bar_index, category in enumerate(ratio_table.index):
            cumulative_height = 0.0
            for income_label in ratio_table.columns:
                segment_ratio = ratio_table.loc[category, income_label]
                segment_count = int(count_table.loc[category, income_label])
                if segment_ratio > 0:
                    ax.annotate(
                        f"{segment_count}건\n({segment_ratio * 100:.1f}%)",
                        (bar_index, cumulative_height + segment_ratio / 2),
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
                cumulative_height += segment_ratio

        ax.set_title(f"{column}별 income 비율 (표본 {min_group_size}건 미만 그룹 제외)")
        ax.set_xlabel(column)
        ax.set_ylabel("비율")
        ax.legend(title="income", fontsize=8)
        ax.tick_params(axis="x", rotation=45)

    plot_income_ratio("education", axes[0, 0], cap_categories=True)
    plot_income_ratio("occupation", axes[0, 1], cap_categories=True)
    plot_income_ratio("sex", axes[1, 0], cap_categories=False)
    plot_income_ratio("workclass", axes[1, 1], cap_categories=True)

    fig.suptitle("범주형 컬럼별 income 비율", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {output_path}")
    return output_path


def create_correlation_heatmap(correlation_matrix: pd.DataFrame, output_path: Path) -> Path:
    """숫자형 컬럼 상관관계 행렬을 히트맵으로 저장한다."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("숫자형 컬럼 상관관계 (Pearson)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {output_path}")
    return output_path


def create_confusion_matrix_plot(
    confusion_matrix_values: np.ndarray, class_labels: list[str], output_path: Path
) -> Path:
    """분류 모델의 혼동 행렬(confusion matrix)을 히트맵으로 저장한다."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (income 분류)")
    ax.set_xlabel("예측값 (Predicted)")
    ax.set_ylabel("실제값 (Actual)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {output_path}")
    return output_path


def create_roc_curve_plot(
    y_true: pd.Series, y_proba: np.ndarray, output_path: Path, positive_label: int = 1
) -> Path:
    """분류 임계값별 성능(TPR/FPR)을 보여주는 ROC curve를 그려 저장한다. y_true는 0/1로 매핑된 값을 받는다."""
    fpr, tpr, _ = roc_curve((y_true == positive_label).astype(int), y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="무작위 분류 기준선")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (income 분류)")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"저장 완료: {output_path}")
    return output_path


def show_static_figures_note() -> None:
    """정적 그래프는 파일마다 plt.show()를 반복 호출하지 않고 저장만 한다는 것을 알리는 안내 출력."""
    print(
        "\n정적 차트는 화면에 반복해서 띄우지 않고 PNG로 저장한다. "
        "필요하면 저장된 파일을 열어 확인한다."
    )


def create_plotly_visualization(df: pd.DataFrame, output_path: Path, sample_size: int = 5000) -> Path:
    """age-hours_per_week 산점도를 income으로 색칠한 인터랙티브 차트를 만들고 HTML로 저장한다."""
    plot_df = df
    if len(df) > sample_size:
        # 브라우저 렌더링 성능을 위해 일부만 무작위 샘플링 (분석 결론에는 영향 없는 표시용 축소)
        plot_df = df.sample(n=sample_size, random_state=42)
        print(f"Plotly 산점도는 시각화 성능을 위해 {sample_size}건을 무작위 샘플링해서 표시합니다.")

    fig = px.scatter(
        plot_df,
        x="age",
        y="hours-per-week",
        color="income",
        hover_data=["education", "occupation", "workclass"],
        title="나이 · 주당 근무시간 · income 관계",
        labels={"age": "나이", "hours-per-week": "주당 근무시간", "income": "소득 구간"},
        opacity=0.6,
    )
    fig.update_layout(xaxis_title="나이(age)", yaxis_title="주당 근무시간(hours-per-week)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Plotly 인터랙티브 차트 HTML 저장 완료: {output_path}")

    try:
        fig.show()
    except Exception as error:
        print("Plotly 화면 출력 중 문제가 발생했습니다(HTML 파일 저장은 완료됨):", error)

    return output_path


def _income_ratio_by_category(df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
    """column별 >50K 비율(%)과 전체 인원 수를 계산한다 (Plotly 막대그래프용 공통 헬퍼)."""
    series = df[column]
    if series.nunique() > top_n:
        series = top_n_with_other(series, n=top_n)

    grouped = df.assign(**{column: series}).groupby(column)["income"]
    high_income_ratio = grouped.apply(lambda s: (s == ">50K").mean() * 100)
    group_count = grouped.size()

    result = pd.DataFrame({"high_income_ratio": high_income_ratio, "count": group_count}).reset_index()
    return result.sort_values("high_income_ratio", ascending=False)


def create_plotly_education_income_bar(df: pd.DataFrame, output_path: Path) -> Path:
    """교육 수준별 고소득(>50K) 비율을 Plotly 막대그래프로 만들어 HTML로 저장한다."""
    ratio_df = _income_ratio_by_category(df, "education")

    fig = px.bar(
        ratio_df,
        x="education",
        y="high_income_ratio",
        hover_data=["count"],
        title="교육 수준별 고소득(>50K) 비율",
        labels={"education": "교육 수준", "high_income_ratio": ">50K 비율(%)", "count": "인원 수"},
    )
    fig.update_layout(xaxis_title="교육 수준", yaxis_title=">50K 비율(%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Plotly 교육 수준별 고소득 비율 차트 저장 완료: {output_path}")
    return output_path


def create_plotly_occupation_income_bar(df: pd.DataFrame, output_path: Path) -> Path:
    """직업별 고소득(>50K) 비율을 Plotly 막대그래프로 만들어 HTML로 저장한다."""
    ratio_df = _income_ratio_by_category(df, "occupation")

    fig = px.bar(
        ratio_df,
        x="occupation",
        y="high_income_ratio",
        hover_data=["count"],
        title="직업별 고소득(>50K) 비율",
        labels={"occupation": "직업", "high_income_ratio": ">50K 비율(%)", "count": "인원 수"},
    )
    fig.update_layout(xaxis_title="직업", yaxis_title=">50K 비율(%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Plotly 직업별 고소득 비율 차트 저장 완료: {output_path}")
    return output_path


def create_plotly_facet_scatter(df: pd.DataFrame, output_path: Path, sample_size: int = 5000) -> Path:
    """직업(occupation)을 facet으로, 성별(sex)과 income을 색/기호로 구분한 산점도를 만든다."""
    plot_df = df
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=42)

    top_occupations = df["occupation"].value_counts().head(4).index
    plot_df = plot_df[plot_df["occupation"].isin(top_occupations)]

    fig = px.scatter(
        plot_df,
        x="age",
        y="hours-per-week",
        color="income",
        symbol="sex",
        facet_col="occupation",
        facet_col_wrap=2,
        opacity=0.5,
        title="직업(상위 4개)별 나이·근무시간 관계 (성별·소득 구분)",
        labels={"age": "나이", "hours-per-week": "주당 근무시간", "income": "소득 구간"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Plotly facet 산점도 저장 완료: {output_path}")
    return output_path


def create_plotly_treemap(df: pd.DataFrame, output_path: Path) -> Path:
    """직업 -> 교육 수준 -> income 구성을 Treemap으로 표현한다."""
    top_occupations = df["occupation"].value_counts().head(8).index
    treemap_df = df[df["occupation"].isin(top_occupations)]

    fig = px.treemap(
        treemap_df,
        path=["occupation", "education", "income"],
        title="직업 · 교육 수준 · income 구성 (상위 8개 직업)",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Plotly Treemap 저장 완료: {output_path}")
    return output_path


def create_plotly_parallel_categories(df: pd.DataFrame, output_path: Path) -> Path:
    """sex, workclass, income 등 여러 범주형 변수 간의 관계를 Parallel Categories로 표현한다."""
    plot_df = df.copy()
    plot_df["occupation_top"] = top_n_with_other(plot_df["occupation"], n=6)
    plot_df["income_code"] = (plot_df["income"] == ">50K").astype(int)

    fig = px.parallel_categories(
        plot_df,
        dimensions=["sex", "workclass", "occupation_top", "income"],
        color="income_code",
        color_continuous_scale=px.colors.sequential.Blues,
        title="성별 · workclass · 직업(상위) · income 관계 (Parallel Categories)",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Plotly Parallel Categories 저장 완료: {output_path}")
    return output_path
