"""정적(Seaborn/Matplotlib) 시각화와 인터랙티브(Plotly) 시각화를 담당하는 모듈."""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

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


def create_categorical_eda_plot(df: pd.DataFrame, output_path: Path, top_n: int = 10) -> Path:
    """education/occupation/workclass/sex별 income 비율을 2x2 서브플롯으로 비교한다."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    def plot_income_ratio(column: str, ax, cap_categories: bool) -> None:
        series = df[column]
        if cap_categories and series.nunique() > top_n:
            series = top_n_with_other(series, n=top_n)  # 범주가 많으면 상위 top_n + Other로 정리

        ratio_table = pd.crosstab(series, df["income"], normalize="index")
        ratio_table = ratio_table.loc[ratio_table.sum(axis=1).sort_values(ascending=False).index]
        ratio_table.plot(kind="bar", stacked=True, ax=ax, legend=True)

        ax.set_title(f"{column}별 income 비율")
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
