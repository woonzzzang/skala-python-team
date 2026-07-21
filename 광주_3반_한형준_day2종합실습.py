"""
작성일: 2026-07-21
작성자: 광주 3반 한형준
작성 목적: Adult Census Income 데이터로 데이터 준비부터 EDA·통계 검정·
          머신러닝·모델 저장·자동 보고서까지 End-to-End 분석을 수행하기 위함

프로그램 전체 설명:
adult.data만 Pandas와 Polars로 로딩해 시간·메모리·결측·중복·소득 분포를
비교한다. 원본 컬럼명과 실제 극단값·희소 범주는 보존하고 완전 중복만 제거한다.
수치형·범주형 EDA, 정적·인터랙티브 시각화, 다중 t-test와 효과 크기,
카이제곱 검정을 수행한다. 모델 단계에서는 먼저 80:20 계층 분할하고 train에서만
학습되는 결측 대체·표준화·원-핫 인코딩과 균형 Logistic Regression을 하나의
Pipeline으로 구성한다. fnlwgt·민감 변수 포함 여부를 비교하고 최종 Pipeline,
컬럼 목록, 메타데이터를 joblib과 JSON으로 저장한 뒤 Markdown 보고서를 생성한다.

변경 내역:
- 2026-07-21: Pandas·Polars 로딩 시간/메모리/결측/중복/groupby 비교 추가
- 2026-07-21: 수치형·범주형 확장 EDA와 이상치·희소 범주 진단 추가
- 2026-07-21: Seaborn 2×2 차트 3종, 산점도, Plotly 차트 6종 추가
- 2026-07-21: 다중 t-test·효과 크기·신뢰구간·카이제곱 검정 추가
- 2026-07-21: train-only 전처리, 모델 3종 비교, ROC-AUC·ROC Curve 추가
- 2026-07-21: Pipeline·컬럼·메타데이터 저장과 기준 체크리스트 자동화
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
MODEL_CATEGORICAL_COLUMNS = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
TARGET_COLUMN = "income"
LOW_INCOME = "<=50K"
HIGH_INCOME = ">50K"
TARGET_MAPPING = {LOW_INCOME: 0, HIGH_INCOME: 1}
SENSITIVE_COLUMNS = ["race", "sex"]
MISSING_LABEL = "(Missing)"
RANDOM_STATE = 42
ALPHA = 0.05
MIN_GROUP_SIZE = 100

SCRIPT_DIR = Path(__file__).resolve().parent
MATPLOTLIB_CACHE = SCRIPT_DIR / ".matplotlib_cache"
MATPLOTLIB_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import plotly.express as px  # noqa: E402
import seaborn as sns  # noqa: E402


@dataclass(frozen=True)
class DataPreparationResult:
    """원본 보존·완전 중복 제거 및 두 엔진 비교 결과."""

    pandas_raw: pd.DataFrame
    polars_raw: pl.DataFrame
    deduplicated: pd.DataFrame
    polars_deduplicated: pl.DataFrame
    comparison: pd.DataFrame
    missing_counts: pd.Series
    duplicate_count: int


@dataclass(frozen=True)
class EDAResult:
    """수치형·범주형 EDA와 상관분석 결과."""

    numeric_summary: pd.DataFrame
    numeric_by_income: pd.DataFrame
    categorical_summary: pd.DataFrame
    categorical_detail: pd.DataFrame
    pearson: pd.DataFrame
    spearman: pd.DataFrame


@dataclass(frozen=True)
class StatisticalResult:
    """독립표본 검정과 범주형 독립성 검정 결과."""

    t_tests: pd.DataFrame
    chi_square_tests: pd.DataFrame


@dataclass(frozen=True)
class ModelEvaluation:
    """모델 한 개의 Pipeline과 예측·평가 결과."""

    name: str
    pipeline: Pipeline
    raw_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    metrics: dict[str, float]
    confusion: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    classification: pd.DataFrame


@dataclass(frozen=True)
class ModelingResult:
    """모델 비교, 최종 모델과 재현 메타데이터."""

    final: ModelEvaluation
    comparison: pd.DataFrame
    model_path: Path
    metadata_path: Path
    y_test: pd.Series
    train_rows: int
    test_rows: int
    imputation_statistics: dict[str, dict[str, object]]


def configure_plotting() -> None:
    """macOS에서 한글이 깨지지 않도록 글꼴과 공통 스타일을 설정한다."""

    font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])


def prepare_output_dirs(output_argument: str | None) -> tuple[Path, Path, Path]:
    """보고서·표·차트를 저장할 폴더를 준비한다."""

    output_dir = (
        Path(output_argument).expanduser().resolve()
        if output_argument
        else SCRIPT_DIR / "day2_outputs"
    )
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, figure_dir, table_dir


def download_dataset(data_argument: str | None, force_download: bool = False) -> Path:
    """adult.data 하나만 사용하며 로컬 파일이 없을 때만 UCI에서 내려받는다."""

    if data_argument:
        data_path = Path(data_argument).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"지정한 Adult 데이터가 없습니다: {data_path}")
        return data_path

    data_path = SCRIPT_DIR / "adult.data"
    if data_path.is_file() and not force_download:
        print(f"로컬 adult.data 사용: {data_path}")
        return data_path

    print(f"UCI Adult 데이터 다운로드: {DATA_URL}")
    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    temporary_path = data_path.with_suffix(".data.tmp")
    try:
        with (
            urlopen(request, timeout=60) as response,
            temporary_path.open("wb") as file,
        ):
            shutil.copyfileobj(response, file)
        temporary_path.replace(data_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    if data_path.stat().st_size == 0:
        raise RuntimeError("다운로드한 adult.data가 비어 있습니다.")
    return data_path


def load_with_pandas(data_path: Path) -> pd.DataFrame:
    """원본 하이픈 컬럼명을 유지하고 물음표를 결측값으로 읽는다."""

    frame = pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values=["?"],
    )
    for column in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace("?", pd.NA)
    return frame


def load_with_polars(data_path: Path) -> pl.DataFrame:
    """Pandas와 같은 원본·결측 규칙으로 Adult 데이터를 Polars에 읽는다."""

    frame = pl.read_csv(
        data_path,
        has_header=False,
        new_columns=COLUMNS,
        schema_overrides={column: pl.String for column in COLUMNS},
        infer_schema_length=10_000,
    )
    frame = frame.with_columns(
        [pl.col(column).str.strip_chars().alias(column) for column in COLUMNS]
    )
    frame = frame.filter(pl.col("age").is_not_null())
    frame = frame.with_columns(
        [
            pl.when(pl.col(column) == "?")
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
            for column in CATEGORICAL_COLUMNS + [TARGET_COLUMN]
        ]
    )
    return frame.with_columns(
        [pl.col(column).cast(pl.Int64).alias(column) for column in NUMERIC_COLUMNS]
    )


def timed_load(loader, data_path: Path, repeats: int) -> tuple[object, float]:
    """동일 로더를 반복 실행해 중앙 로딩 시간과 마지막 결과를 반환한다."""

    if repeats < 1:
        raise ValueError("benchmark repeats는 1 이상이어야 합니다.")
    durations = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = loader(data_path)
        durations.append(time.perf_counter() - started)
    return result, float(np.median(durations))


def polars_missing_counts(frame: pl.DataFrame) -> dict[str, int]:
    """Polars null_count 결과를 일반 딕셔너리로 변환한다."""

    return {
        column: int(value)
        for column, value in frame.null_count().row(0, named=True).items()
    }


def prepare_data(data_path: Path, repeats: int = 3) -> DataPreparationResult:
    """두 엔진의 로딩·결측·중복·groupby·시간·메모리를 비교한다."""

    pandas_loaded, pandas_seconds = timed_load(load_with_pandas, data_path, repeats)
    polars_loaded, polars_seconds = timed_load(load_with_polars, data_path, repeats)
    pandas_raw = pandas_loaded
    polars_raw = polars_loaded
    if not isinstance(pandas_raw, pd.DataFrame) or not isinstance(
        polars_raw, pl.DataFrame
    ):
        raise TypeError("데이터 로더가 예상한 DataFrame을 반환하지 않았습니다.")
    if pandas_raw.shape != polars_raw.shape:
        raise AssertionError("Pandas와 Polars 로딩 크기가 다릅니다.")
    if list(pandas_raw.columns) != polars_raw.columns:
        raise AssertionError("Pandas와 Polars 원본 컬럼명이 다릅니다.")

    pandas_missing = pandas_raw.isna().sum()
    polars_missing = polars_missing_counts(polars_raw)
    if pandas_missing.to_dict() != polars_missing:
        raise AssertionError("Pandas와 Polars 결측치 개수가 다릅니다.")

    deduplicated = pandas_raw.drop_duplicates().reset_index(drop=True)
    polars_deduplicated = polars_raw.unique(maintain_order=True)
    duplicate_count = len(pandas_raw) - len(deduplicated)
    polars_duplicate_count = polars_raw.height - polars_deduplicated.height
    if duplicate_count != polars_duplicate_count:
        raise AssertionError("Pandas와 Polars 중복 행 수가 다릅니다.")

    pandas_income = (
        deduplicated[TARGET_COLUMN].value_counts().sort_index().astype(int).to_dict()
    )
    polars_income = {
        str(row[TARGET_COLUMN]): int(row["count"])
        for row in polars_deduplicated.group_by(TARGET_COLUMN)
        .len(name="count")
        .sort(TARGET_COLUMN)
        .to_dicts()
    }
    if pandas_income != polars_income:
        raise AssertionError("Pandas와 Polars income별 개수가 다릅니다.")

    pandas_memory_mb = pandas_raw.memory_usage(index=True, deep=True).sum() / 1024**2
    polars_memory_mb = polars_raw.estimated_size("mb")
    rows = [
        {
            "engine": "Pandas",
            "load_median_seconds": pandas_seconds,
            "memory_mb": pandas_memory_mb,
            "raw_rows": len(pandas_raw),
            "raw_columns": pandas_raw.shape[1],
            "missing_total": int(pandas_missing.sum()),
            "duplicate_rows": duplicate_count,
            "deduplicated_rows": len(deduplicated),
            "income_<=50K": pandas_income[LOW_INCOME],
            "income_>50K": pandas_income[HIGH_INCOME],
        },
        {
            "engine": "Polars",
            "load_median_seconds": polars_seconds,
            "memory_mb": polars_memory_mb,
            "raw_rows": polars_raw.height,
            "raw_columns": polars_raw.width,
            "missing_total": sum(polars_missing.values()),
            "duplicate_rows": polars_duplicate_count,
            "deduplicated_rows": polars_deduplicated.height,
            "income_<=50K": polars_income[LOW_INCOME],
            "income_>50K": polars_income[HIGH_INCOME],
        },
    ]
    comparison = pd.DataFrame(rows)

    print("\n=== 1. 데이터 준비 · Pandas vs Polars ===")
    print(comparison.round(4).to_string(index=False))
    print("\n[처리 전 결측치]")
    print(pandas_missing[pandas_missing > 0].to_string())
    print(
        f"완전 중복 {duplicate_count:,}건만 제거: "
        f"{len(pandas_raw):,}행 → {len(deduplicated):,}행"
    )
    print(
        "EDA에서는 결측을 그대로 표시하고, ML 대체 기준은 "
        "Pipeline이 train에서만 학습합니다."
    )
    return DataPreparationResult(
        pandas_raw=pandas_raw,
        polars_raw=polars_raw,
        deduplicated=deduplicated,
        polars_deduplicated=polars_deduplicated,
        comparison=comparison,
        missing_counts=pandas_missing,
        duplicate_count=duplicate_count,
    )


def numeric_eda(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """기술통계·분포 모양·결측·0 비율·IQR 후보와 소득별 요약을 계산한다."""

    rows = []
    for column in NUMERIC_COLUMNS:
        values = frame[column].dropna()
        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append(
            {
                "variable": column,
                "count": int(values.count()),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": values.max(),
                "iqr": iqr,
                "skewness": values.skew(),
                "kurtosis": values.kurtosis(),
                "missing_count": int(frame[column].isna().sum()),
                "missing_rate": frame[column].isna().mean(),
                "unique_count": int(values.nunique()),
                "zero_rate": (values == 0).mean(),
                "iqr_outlier_candidates": int(
                    ((values < lower) | (values > upper)).sum()
                ),
            }
        )
    summary = pd.DataFrame(rows)

    grouped_rows = []
    for column in NUMERIC_COLUMNS:
        grouped = frame.groupby(TARGET_COLUMN, observed=True)[column].agg(
            ["count", "mean", "median", "std"]
        )
        for income, row in grouped.iterrows():
            grouped_rows.append(
                {
                    "variable": column,
                    "income": income,
                    "count": int(row["count"]),
                    "mean": row["mean"],
                    "median": row["median"],
                    "std": row["std"],
                }
            )
    by_income = pd.DataFrame(grouped_rows)

    print("\n=== 2. 수치형 EDA ===")
    print(
        summary[
            [
                "variable",
                "count",
                "mean",
                "std",
                "q1",
                "median",
                "q3",
                "skewness",
                "iqr_outlier_candidates",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("IQR 후보는 진단만 하며 실제 가능한 극단값은 삭제·변환하지 않습니다.")
    return summary, by_income


def categorical_eda(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """범주 빈도·비율·결측·희소 범주와 그룹별 고소득 비율을 계산한다."""

    summary_rows = []
    detail_rows = []
    total_high_income = int((frame[TARGET_COLUMN] == HIGH_INCOME).sum())
    for column in CATEGORICAL_COLUMNS:
        display_values = frame[column].astype("string").fillna(MISSING_LABEL)
        counts = display_values.value_counts(dropna=False)
        mode = counts.index[0]
        rare_count = int((counts < MIN_GROUP_SIZE).sum())
        summary_rows.append(
            {
                "variable": column,
                "unique_count": int(frame[column].nunique(dropna=True)),
                "mode": mode,
                "mode_count": int(counts.iloc[0]),
                "missing_count": int(frame[column].isna().sum()),
                "missing_rate": frame[column].isna().mean(),
                "rare_category_count": rare_count,
                "rare_threshold": MIN_GROUP_SIZE,
            }
        )

        temporary = pd.DataFrame(
            {
                "category_value": display_values,
                "high_income": (frame[TARGET_COLUMN] == HIGH_INCOME).astype(int),
            }
        )
        grouped = temporary.groupby("category_value", observed=True)["high_income"].agg(
            ["size", "sum"]
        )
        grouped = grouped.rename(
            columns={"size": "total_count", "sum": "high_income_count"}
        ).reset_index()
        grouped["variable"] = column
        grouped["category_rate"] = grouped["total_count"] / len(frame)
        grouped["high_income_rate_within_group"] = (
            grouped["high_income_count"] / grouped["total_count"]
        )
        grouped["share_of_all_high_income"] = (
            grouped["high_income_count"] / total_high_income
        )
        grouped["is_rare"] = grouped["total_count"] < MIN_GROUP_SIZE
        detail_rows.extend(grouped.to_dict("records"))

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    detail = detail[
        [
            "variable",
            "category_value",
            "total_count",
            "category_rate",
            "high_income_count",
            "high_income_rate_within_group",
            "share_of_all_high_income",
            "is_rare",
        ]
    ]
    print("\n=== 3. 범주형 EDA ===")
    print(summary.round(4).to_string(index=False))
    print(
        f"희소 범주는 표본 {MIN_GROUP_SIZE}건 미만으로 표시만 하며 "
        "원본과 모델에서 합치거나 삭제하지 않습니다."
    )
    return summary, detail


def correlation_analysis(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """수치형 변수와 참고용 income 0·1의 Pearson·Spearman 상관을 비교한다."""

    correlation_frame = frame[NUMERIC_COLUMNS].copy()
    correlation_frame["income_binary"] = frame[TARGET_COLUMN].map(TARGET_MAPPING)
    pearson = correlation_frame.corr(method="pearson")
    spearman = correlation_frame.corr(method="spearman")
    print("\n=== 4. 상관분석 ===")
    print("[income_binary와 Pearson 상관]")
    print(pearson["income_binary"].sort_values(ascending=False).round(4).to_string())
    print("\n[income_binary와 Spearman 상관]")
    print(spearman["income_binary"].sort_values(ascending=False).round(4).to_string())
    print("상관관계는 인과관계를 의미하지 않습니다.")
    return pearson, spearman


def run_eda(frame: pd.DataFrame) -> EDAResult:
    """수치형·범주형·상관 EDA 전체를 실행한다."""

    numeric_summary, numeric_by_income = numeric_eda(frame)
    categorical_summary, categorical_detail = categorical_eda(frame)
    pearson, spearman = correlation_analysis(frame)
    return EDAResult(
        numeric_summary=numeric_summary,
        numeric_by_income=numeric_by_income,
        categorical_summary=categorical_summary,
        categorical_detail=categorical_detail,
        pearson=pearson,
        spearman=spearman,
    )


def category_rates(
    detail: pd.DataFrame, variable: str, minimum_count: int = MIN_GROUP_SIZE
) -> pd.DataFrame:
    """시각화를 위해 충분한 표본이 있는 범주별 고소득 통계를 반환한다."""

    return (
        detail[
            (detail["variable"] == variable) & (detail["total_count"] >= minimum_count)
        ]
        .sort_values("high_income_rate_within_group", ascending=False)
        .copy()
    )


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """공통 해상도로 정적 차트를 저장하고 메모리를 해제한다."""

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def create_static_charts(
    frame: pd.DataFrame, eda: EDAResult, figure_dir: Path
) -> list[Path]:
    """후보에 포함된 정적 EDA 차트를 주제별 2×2와 산점도로 저장한다."""

    paths = []
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    sns.countplot(
        data=frame,
        x=TARGET_COLUMN,
        order=[LOW_INCOME, HIGH_INCOME],
        hue=TARGET_COLUMN,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("소득 그룹 인원 분포")
    axes[0, 0].set_xlabel("연 소득")
    axes[0, 0].set_ylabel("인원 수")
    sns.histplot(
        data=frame,
        x="age",
        hue=TARGET_COLUMN,
        bins=30,
        kde=True,
        element="step",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("연령 히스토그램과 KDE")
    axes[0, 1].set_xlabel("연령")
    sns.boxplot(
        data=frame,
        x=TARGET_COLUMN,
        y="hours-per-week",
        order=[LOW_INCOME, HIGH_INCOME],
        hue=TARGET_COLUMN,
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("소득 그룹별 주당 근무시간 박스플롯")
    axes[1, 0].set_xlabel("연 소득")
    axes[1, 0].set_ylabel("주당 근무시간")
    sns.heatmap(
        eda.pearson, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=axes[1, 1]
    )
    axes[1, 1].set_title("수치형 Pearson 상관관계")
    path = figure_dir / "01_static_overview.png"
    paths.append(save_figure(fig, path))

    education_rates = category_rates(eda.categorical_detail, "education")
    occupation_rates = category_rates(eda.categorical_detail, "occupation")
    fig, axes = plt.subplots(2, 2, figsize=(19, 14))
    sns.kdeplot(
        data=frame,
        x="age",
        hue=TARGET_COLUMN,
        common_norm=False,
        fill=True,
        alpha=0.35,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("소득 그룹별 연령 KDE")
    axes[0, 0].set_xlabel("연령")
    sns.barplot(
        data=education_rates,
        y="category_value",
        x="high_income_rate_within_group",
        color="#4c78a8",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("교육 수준별 그룹 내 >50K 비율")
    axes[0, 1].set_xlabel("고소득 비율")
    axes[0, 1].set_ylabel("교육 수준")
    sns.barplot(
        data=occupation_rates,
        y="category_value",
        x="high_income_rate_within_group",
        color="#f58518",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("직업별 그룹 내 >50K 비율")
    axes[1, 0].set_xlabel("고소득 비율")
    axes[1, 0].set_ylabel("직업")
    sns.countplot(
        data=frame,
        x="sex",
        hue=TARGET_COLUMN,
        hue_order=[LOW_INCOME, HIGH_INCOME],
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("성별 소득 그룹 인원")
    axes[1, 1].set_xlabel("성별")
    axes[1, 1].set_ylabel("인원 수")
    path = figure_dir / "02_static_group_comparison.png"
    paths.append(save_figure(fig, path))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    distribution_specs = [
        ("capital-gain", "자본 이득 분포: log1p", True),
        ("capital-loss", "자본 손실 분포: log1p", True),
        ("fnlwgt", "표본 가중치 fnlwgt 분포", False),
        ("hours-per-week", "주당 근무시간 분포", False),
    ]
    for axis, (column, title, log_transform) in zip(axes.flat, distribution_specs):
        values = np.log1p(frame[column]) if log_transform else frame[column]
        sns.histplot(values, bins=40, kde=True, ax=axis)
        axis.set_title(title)
        axis.set_xlabel(f"log1p({column})" if log_transform else column)
        axis.set_ylabel("빈도")
    path = figure_dir / "03_static_numeric_distributions.png"
    paths.append(save_figure(fig, path))

    sample = frame.sample(n=min(6_000, len(frame)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=sample,
        x="age",
        y="hours-per-week",
        hue=TARGET_COLUMN,
        alpha=0.45,
        s=32,
        ax=ax,
    )
    ax.set_title("연령과 주당 근무시간의 관계")
    ax.set_xlabel("연령")
    ax.set_ylabel("주당 근무시간")
    path = figure_dir / "04_static_age_hours_scatter.png"
    paths.append(save_figure(fig, path))
    print(f"\nSeaborn/Matplotlib 정적 차트 {len(paths)}개 저장")
    return paths


def write_plotly(figure, output_path: Path) -> Path:
    """Plotly JavaScript를 폴더에서 공유하는 인터랙티브 HTML을 저장한다."""

    figure.write_html(output_path, include_plotlyjs="directory", full_html=True)
    page_title = escape(str(figure.layout.title.text or "Adult Income Plotly Chart"))
    html = output_path.read_text(encoding="utf-8")
    output_path.write_text(
        html.replace("<head>", f"<head><title>{page_title}</title>", 1),
        encoding="utf-8",
    )
    return output_path


def create_plotly_charts(
    frame: pd.DataFrame, eda: EDAResult, figure_dir: Path
) -> list[Path]:
    """교육·직업·산점도·facet·treemap·parallel categories 차트를 만든다."""

    paths = []
    for variable, label, filename in [
        ("education", "교육 수준", "11_plotly_education_income_rate.html"),
        ("occupation", "직업", "12_plotly_occupation_income_rate.html"),
    ]:
        rates = category_rates(eda.categorical_detail, variable)
        figure = px.bar(
            rates,
            x="category_value",
            y="high_income_rate_within_group",
            color="high_income_rate_within_group",
            hover_data={
                "total_count": ":,",
                "high_income_count": ":,",
                "share_of_all_high_income": ":.2%",
                "category_rate": ":.2%",
                "high_income_rate_within_group": ":.2%",
            },
            labels={
                "category_value": label,
                "high_income_rate_within_group": "그룹 내 >50K 비율",
                "total_count": "전체 인원",
                "high_income_count": ">50K 인원",
                "share_of_all_high_income": "전체 >50K 중 비율",
                "category_rate": "전체 표본 중 비율",
            },
            title=f"{label}별 고소득 비율과 표본 수",
        )
        figure.update_layout(template="plotly_white", xaxis_tickangle=-35)
        paths.append(write_plotly(figure, figure_dir / filename))

    sample = frame.sample(n=min(6_000, len(frame)), random_state=RANDOM_STATE).copy()
    figure = px.scatter(
        sample,
        x="age",
        y="hours-per-week",
        color=TARGET_COLUMN,
        symbol="sex",
        opacity=0.55,
        hover_data=["education", "occupation", "workclass"],
        title="연령과 주당 근무시간: 소득·성별 구분",
        labels={"age": "연령", "hours-per-week": "주당 근무시간"},
    )
    figure.update_layout(template="plotly_white")
    paths.append(write_plotly(figure, figure_dir / "13_plotly_age_hours_scatter.html"))

    occupation_counts = (
        frame.assign(occupation=frame["occupation"].fillna(MISSING_LABEL))
        .groupby(["occupation", "sex", TARGET_COLUMN], observed=True)
        .size()
        .reset_index(name="people")
    )
    occupation_totals = occupation_counts.groupby("occupation")["people"].transform(
        "sum"
    )
    occupation_counts = occupation_counts[occupation_totals >= MIN_GROUP_SIZE]
    figure = px.bar(
        occupation_counts,
        x="occupation",
        y="people",
        color=TARGET_COLUMN,
        facet_col="sex",
        barmode="group",
        title="직업·성별·소득 그룹별 인원",
        labels={"occupation": "직업", "people": "인원 수", "sex": "성별"},
    )
    figure.update_layout(template="plotly_white", xaxis_tickangle=-45)
    paths.append(
        write_plotly(figure, figure_dir / "14_plotly_occupation_sex_income_facet.html")
    )

    treemap_data = (
        frame.assign(
            occupation=frame["occupation"].fillna(MISSING_LABEL),
            education=frame["education"].fillna(MISSING_LABEL),
        )
        .groupby(["occupation", "education", TARGET_COLUMN], observed=True)
        .size()
        .reset_index(name="people")
    )
    figure = px.treemap(
        treemap_data,
        path=["occupation", "education", TARGET_COLUMN],
        values="people",
        color=TARGET_COLUMN,
        title="직업·교육·소득 구성 Treemap",
    )
    paths.append(write_plotly(figure, figure_dir / "15_plotly_treemap.html"))

    parallel_sample = (
        frame[
            [
                "workclass",
                "education",
                "occupation",
                "sex",
                TARGET_COLUMN,
            ]
        ]
        .fillna(MISSING_LABEL)
        .sample(n=min(5_000, len(frame)), random_state=RANDOM_STATE)
        .copy()
    )
    parallel_sample["income_binary"] = parallel_sample[TARGET_COLUMN].map(
        TARGET_MAPPING
    )
    figure = px.parallel_categories(
        parallel_sample,
        dimensions=["workclass", "education", "occupation", "sex", TARGET_COLUMN],
        color="income_binary",
        color_continuous_scale=px.colors.sequential.Blues,
        title="근로형태·교육·직업·성별·소득 Parallel Categories",
    )
    paths.append(
        write_plotly(figure, figure_dir / "16_plotly_parallel_categories.html")
    )
    print(f"Plotly 인터랙티브 차트 {len(paths)}개 저장")
    return paths


def cohens_d(first: pd.Series, second: pd.Series) -> float:
    """두 독립 집단의 평균 차이를 pooled 표준편차 단위로 계산한다."""

    pooled_variance = (
        (len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)
    ) / (len(first) + len(second) - 2)
    return float((first.mean() - second.mean()) / np.sqrt(pooled_variance))


def normality_p_value(values: pd.Series) -> float:
    """표본이 충분하면 D’Agostino 정규성 검정 p-value를 반환한다."""

    if len(values) < 8:
        return float("nan")
    return float(stats.normaltest(values).pvalue)


def format_p_value(p_value: float) -> str:
    """부동소수점 언더플로를 고려해 매우 작은 p-value를 표현한다."""

    return "<1e-300" if p_value == 0 else f"{p_value:.3e}"


def t_test_row(
    topic: str,
    first_label: str,
    first: pd.Series,
    second_label: str,
    second: pd.Series,
) -> dict[str, object]:
    """정규성·등분산성 확인 후 양측 t-test와 보조 비모수 검정을 수행한다."""

    first = first.dropna().astype(float)
    second = second.dropna().astype(float)
    if min(len(first), len(second)) < 2:
        raise ValueError(f"{topic}: 두 그룹 표본이 부족합니다.")
    normal_first = normality_p_value(first)
    normal_second = normality_p_value(second)
    levene_p = float(stats.levene(first, second, center="median").pvalue)
    equal_var = levene_p >= ALPHA
    result = stats.ttest_ind(
        first,
        second,
        equal_var=equal_var,
        alternative="two-sided",
    )
    confidence = result.confidence_interval(confidence_level=0.95)
    mann_whitney = stats.mannwhitneyu(
        first, second, alternative="two-sided", method="asymptotic"
    )
    p_value = float(result.pvalue)
    interpretation = (
        "귀무가설 기각: 두 그룹 평균에 통계적으로 유의한 차이가 있음"
        if p_value < ALPHA
        else "귀무가설을 기각할 충분한 근거가 없음"
    )
    return {
        "topic": topic,
        "group_1": first_label,
        "group_2": second_label,
        "n_1": len(first),
        "n_2": len(second),
        "mean_1": first.mean(),
        "mean_2": second.mean(),
        "mean_difference": first.mean() - second.mean(),
        "ci_95_low": float(confidence.low),
        "ci_95_high": float(confidence.high),
        "normality_p_1": normal_first,
        "normality_p_2": normal_second,
        "levene_p": levene_p,
        "test": "Student t-test" if equal_var else "Welch t-test",
        "t_statistic": float(result.statistic),
        "p_value": p_value,
        "p_value_display": format_p_value(p_value),
        "significant_0.01": p_value < 0.01,
        "significant_0.05": p_value < 0.05,
        "significant_0.10": p_value < 0.10,
        "cohens_d": cohens_d(first, second),
        "mann_whitney_p": float(mann_whitney.pvalue),
        "interpretation": interpretation,
    }


def perform_t_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """후보에 제시된 다섯 가지 독립 두 그룹 비교를 모두 수행한다."""

    tests = [
        t_test_row(
            "소득 그룹별 hours-per-week",
            HIGH_INCOME,
            frame.loc[frame[TARGET_COLUMN] == HIGH_INCOME, "hours-per-week"],
            LOW_INCOME,
            frame.loc[frame[TARGET_COLUMN] == LOW_INCOME, "hours-per-week"],
        ),
        t_test_row(
            "소득 그룹별 age",
            HIGH_INCOME,
            frame.loc[frame[TARGET_COLUMN] == HIGH_INCOME, "age"],
            LOW_INCOME,
            frame.loc[frame[TARGET_COLUMN] == LOW_INCOME, "age"],
        ),
        t_test_row(
            "성별 hours-per-week",
            "Male",
            frame.loc[frame["sex"] == "Male", "hours-per-week"],
            "Female",
            frame.loc[frame["sex"] == "Female", "hours-per-week"],
        ),
        t_test_row(
            "workclass별 hours-per-week",
            "Private",
            frame.loc[frame["workclass"] == "Private", "hours-per-week"],
            "Self-emp-not-inc",
            frame.loc[frame["workclass"] == "Self-emp-not-inc", "hours-per-week"],
        ),
        t_test_row(
            "소득 그룹별 education-num",
            HIGH_INCOME,
            frame.loc[frame[TARGET_COLUMN] == HIGH_INCOME, "education-num"],
            LOW_INCOME,
            frame.loc[frame[TARGET_COLUMN] == LOW_INCOME, "education-num"],
        ),
    ]
    result = pd.DataFrame(tests)
    print("\n=== 5. 독립표본 양측 t-test · 효과 크기 ===")
    print(
        result[
            [
                "topic",
                "test",
                "mean_difference",
                "p_value_display",
                "cohens_d",
                "mann_whitney_p",
                "interpretation",
            ]
        ]
        .round(5)
        .to_string(index=False)
    )
    print(
        "정규성 결과는 보조 판단에 사용하고, 등분산이면 Student, "
        "아니면 Welch를 사용했습니다. 비정규성 민감도 확인용 "
        "Mann–Whitney p-value도 함께 제시합니다."
    )
    return result


def perform_chi_square_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """후보에 제시된 범주형 변수와 income의 독립성 검정을 수행한다."""

    rows = []
    for column in [
        "education",
        "occupation",
        "workclass",
        "sex",
        "marital-status",
        "race",
        "native-country",
    ]:
        values = frame[column].astype("string").fillna(MISSING_LABEL)
        table = pd.crosstab(values, frame[TARGET_COLUMN])
        chi2, p_value, dof, expected = stats.chi2_contingency(table)
        denominator = len(frame) * min(table.shape[0] - 1, table.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / denominator) if denominator else 0.0
        low_expected_rate = float((expected < 5).mean())
        rows.append(
            {
                "variable": column,
                "chi2": float(chi2),
                "degrees_of_freedom": int(dof),
                "p_value": float(p_value),
                "p_value_display": format_p_value(float(p_value)),
                "cramers_v": float(cramers_v),
                "minimum_expected": float(expected.min()),
                "expected_below_5_rate": low_expected_rate,
                "interpretation": (
                    "귀무가설 기각: income과 통계적으로 유의한 연관성이 있음"
                    if p_value < ALPHA
                    else "귀무가설을 기각할 충분한 근거가 없음"
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
    print("\n=== 6. 카이제곱 검정 · Cramer's V ===")
    print(
        result[
            [
                "variable",
                "chi2",
                "p_value_display",
                "cramers_v",
                "minimum_expected",
            ]
        ]
        .round(5)
        .to_string(index=False)
    )
    print("통계적 연관성은 인과관계를 의미하지 않습니다.")
    return result


def run_statistical_analysis(frame: pd.DataFrame) -> StatisticalResult:
    """수치형 평균 검정과 범주형 독립성 검정을 실행한다."""

    return StatisticalResult(
        t_tests=perform_t_tests(frame),
        chi_square_tests=perform_chi_square_tests(frame),
    )


def build_pipeline(
    numeric_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
    """train에서만 fit되는 대체·표준화·인코딩과 분류기를 구성한다."""

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def model_frame_for_columns(
    frame: pd.DataFrame, columns: list[str], categorical_columns: list[str]
) -> pd.DataFrame:
    """원본 결측을 유지하되 sklearn이 인식하는 np.nan 형태로만 통일한다."""

    result = frame[columns].copy()
    for column in categorical_columns:
        result[column] = (
            result[column].astype(object).where(result[column].notna(), np.nan)
        )
    return result


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    raw_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ModelEvaluation:
    """하나의 Pipeline을 학습하고 모든 합의 평가 지표를 계산한다."""

    pipeline.fit(x_train[raw_columns], y_train)
    predictions = pipeline.predict(x_test[raw_columns])
    probabilities = pipeline.predict_proba(x_test[raw_columns])[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    confusion = confusion_matrix(y_test, predictions)
    classification = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            target_names=[LOW_INCOME, HIGH_INCOME],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    return ModelEvaluation(
        name=name,
        pipeline=pipeline,
        raw_columns=raw_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        metrics=metrics,
        confusion=confusion,
        predictions=predictions,
        probabilities=probabilities,
        classification=classification,
    )


def train_models(frame: pd.DataFrame, output_dir: Path) -> ModelingResult:
    """동일 split에서 전체·fnlwgt 제외·민감 변수 제외 모델을 비교한다."""

    model_frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    y = model_frame[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)
    all_feature_columns = NUMERIC_COLUMNS + MODEL_CATEGORICAL_COLUMNS
    x = model_frame_for_columns(
        model_frame, all_feature_columns, MODEL_CATEGORICAL_COLUMNS
    )
    train_index, test_index = train_test_split(
        model_frame.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_train, x_test = x.loc[train_index], x.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    variants = [
        (
            "all_features",
            NUMERIC_COLUMNS,
            MODEL_CATEGORICAL_COLUMNS,
            "합의 기준에 따라 fnlwgt·민감 변수 포함",
        ),
        (
            "without_fnlwgt",
            [column for column in NUMERIC_COLUMNS if column != "fnlwgt"],
            MODEL_CATEGORICAL_COLUMNS,
            "표본 가중치 제외 민감도 비교",
        ),
        (
            "without_sensitive",
            NUMERIC_COLUMNS,
            [
                column
                for column in MODEL_CATEGORICAL_COLUMNS
                if column not in SENSITIVE_COLUMNS
            ],
            "race·sex 제외 안정성 비교",
        ),
    ]
    evaluations = []
    comparison_rows = []
    for name, numeric_columns, categorical_columns, purpose in variants:
        raw_columns = numeric_columns + categorical_columns
        pipeline = build_pipeline(numeric_columns, categorical_columns)
        evaluation = evaluate_model(
            name,
            pipeline,
            raw_columns,
            numeric_columns,
            categorical_columns,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        evaluations.append(evaluation)
        comparison_rows.append(
            {
                "model": name,
                "purpose": purpose,
                "raw_feature_count": len(raw_columns),
                **evaluation.metrics,
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["f1", "roc_auc"], ascending=False
    )

    eligible = {
        evaluation.name: evaluation
        for evaluation in evaluations
        if evaluation.name in {"all_features", "without_fnlwgt"}
    }
    all_model = eligible["all_features"]
    no_weight_model = eligible["without_fnlwgt"]
    # F1 차이가 0.005 이내면 설명이 쉬운 fnlwgt 제외 모델을 선택한다.
    final = (
        no_weight_model
        if no_weight_model.metrics["f1"] >= all_model.metrics["f1"] - 0.005
        else all_model
    )

    fitted_preprocessor = final.pipeline.named_steps["preprocessor"]
    numeric_imputer = fitted_preprocessor.named_transformers_["numeric"].named_steps[
        "imputer"
    ]
    categorical_imputer = fitted_preprocessor.named_transformers_[
        "categorical"
    ].named_steps["imputer"]
    imputation_statistics: dict[str, dict[str, object]] = {
        "numeric_median": {
            column: float(value)
            for column, value in zip(
                final.numeric_columns, numeric_imputer.statistics_, strict=True
            )
        },
        "categorical_most_frequent": {
            column: str(value)
            for column, value in zip(
                final.categorical_columns,
                categorical_imputer.statistics_,
                strict=True,
            )
        },
    }

    model_path = output_dir / "adult_income_pipeline.joblib"
    metadata_path = output_dir / "adult_income_model_metadata.json"
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "adult.data",
        "selected_model": final.name,
        "feature_columns": final.raw_columns,
        "numeric_columns": final.numeric_columns,
        "categorical_columns": final.categorical_columns,
        "excluded_from_model": [
            column
            for column in COLUMNS
            if column not in final.raw_columns + [TARGET_COLUMN]
        ],
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratify": TARGET_COLUMN,
        },
        "missing_strategy": {
            "numeric": "median learned from train only",
            "categorical": "most_frequent learned from train only",
            "fitted_values": imputation_statistics,
        },
        "class_weight": "balanced",
        "metrics": final.metrics,
        "model_comparison": comparison.to_dict("records"),
    }
    bundle = {
        "pipeline": final.pipeline,
        "feature_columns": final.raw_columns,
        "numeric_columns": final.numeric_columns,
        "categorical_columns": final.categorical_columns,
        "target_mapping": TARGET_MAPPING,
        "metadata": metadata,
    }
    joblib.dump(bundle, model_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    loaded_bundle = joblib.load(model_path)
    reloaded_predictions = loaded_bundle["pipeline"].predict(
        x_test[loaded_bundle["feature_columns"]].head(20)
    )
    if not np.array_equal(final.predictions[:20], reloaded_predictions):
        raise RuntimeError("저장 전후 Pipeline 예측이 다릅니다.")

    print("\n=== 7. Logistic Regression Pipeline 비교 ===")
    print(comparison.round(4).to_string(index=False))
    print(f"최종 모델: {final.name}")
    print(f"원본 feature 컬럼: {final.raw_columns}")
    print("결측 대체·스케일링·인코딩 기준은 train에서만 학습되었습니다.")
    print(
        "train에서 학습된 범주형 최빈값: "
        f"{imputation_statistics['categorical_most_frequent']}"
    )
    print(f"Confusion Matrix [[TN, FP], [FN, TP]]:\n{final.confusion}")
    print(f"Pipeline+컬럼 bundle 저장: {model_path}")
    return ModelingResult(
        final=final,
        comparison=comparison,
        model_path=model_path,
        metadata_path=metadata_path,
        y_test=y_test,
        train_rows=len(train_index),
        test_rows=len(test_index),
        imputation_statistics=imputation_statistics,
    )


def create_model_evaluation_chart(modeling: ModelingResult, figure_dir: Path) -> Path:
    """최종 모델의 Confusion Matrix와 ROC Curve를 한 이미지로 저장한다."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.heatmap(
        modeling.final.confusion,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=[LOW_INCOME, HIGH_INCOME],
        yticklabels=[LOW_INCOME, HIGH_INCOME],
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("예측")
    axes[0].set_ylabel("실제")
    false_positive, true_positive, _ = roc_curve(
        modeling.y_test, modeling.final.probabilities
    )
    axes[1].plot(
        false_positive,
        true_positive,
        linewidth=2,
        label=f"ROC-AUC = {modeling.final.metrics['roc_auc']:.3f}",
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="무작위")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    path = figure_dir / "05_model_confusion_roc.png"
    return save_figure(fig, path)


def save_tables(
    preparation: DataPreparationResult,
    eda: EDAResult,
    statistical: StatisticalResult,
    modeling: ModelingResult,
    table_dir: Path,
) -> list[Path]:
    """재검토 가능한 분석 표를 CSV로 저장한다."""

    tables = {
        "engine_comparison.csv": preparation.comparison,
        "numeric_eda.csv": eda.numeric_summary,
        "numeric_by_income.csv": eda.numeric_by_income,
        "categorical_eda.csv": eda.categorical_summary,
        "categorical_group_detail.csv": eda.categorical_detail,
        "correlation_pearson.csv": eda.pearson.reset_index(names="variable"),
        "correlation_spearman.csv": eda.spearman.reset_index(names="variable"),
        "t_tests.csv": statistical.t_tests,
        "chi_square_tests.csv": statistical.chi_square_tests,
        "model_comparison.csv": modeling.comparison,
        "classification_report.csv": modeling.final.classification.reset_index(
            names="class"
        ),
    }
    paths = []
    for filename, table in tables.items():
        path = table_dir / filename
        table.to_csv(path, index=False)
        paths.append(path)
    return paths


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    """추가 패키지 없이 DataFrame을 Markdown 표로 변환한다."""

    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}g}"
            )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def create_criteria_checklist(output_dir: Path, selected_model: str) -> Path:
    """43개 팀 합의 후보의 코드 반영 상태를 Markdown으로 저장한다."""

    rows = [
        (1, "End-to-End 분석", "준비→EDA→통계→모델→저장→보고서 연결"),
        (2, "adult.data만 사용", "단일 파일을 80:20 train/test로 분할"),
        (3, "원본 컬럼명", "education-num 등 하이픈 이름 유지"),
        (4, "물음표 결측 인식", "로딩 시 null 처리"),
        (5, "수치형 결측", "존재 여부 출력, Pipeline 중앙값 안전망"),
        (6, "범주형 결측", "Pipeline 최빈값 대체"),
        (7, "결측 처리 시점", "대체기는 split 후 train에서만 fit"),
        (8, "결측 행 미삭제", "중복 외 행 삭제 없음"),
        (9, "결측률", "개수와 비율을 EDA 표에 기록"),
        (10, "완전 중복 제거", "24건 제거, 원본 파일 보존"),
        (11, "이상치 탐지", "IQR 후보 수만 진단"),
        (12, "이상치 유지", "clip·삭제·로그 대체 없음"),
        (13, "age 유지", "실제 관측값 보존"),
        (14, "hours-per-week 유지", "실제 관측값 보존"),
        (15, "capital-gain/loss 유지", "원본 보존, 시각화만 log1p"),
        (16, "fnlwgt", "포함·제외 모델 성능 비교"),
        (17, "education 역할", "EDA는 education, 모델은 education-num"),
        (18, "income 문자열", "EDA 원본 유지, 모델링 복사본만 0·1 매핑"),
        (19, "민감 변수", "기본·선택 대상 모델에 포함, 제외 모델은 감사용"),
        (20, "희소 범주", "표시만 하고 모든 범주 유지"),
        (21, "Pandas·Polars", "로드·결측·중복·groupby·시간·메모리 비교"),
        (22, "수치형 EDA", "기술통계·분위수·왜도·첨도·결측·0·IQR·소득별 비교"),
        (23, "범주형 EDA", "빈도·비율·결측·희소·고소득 수와 비율"),
        (24, "시각화 유형", "히스토그램·KDE·박스·count·비율·heatmap·scatter·CM·ROC"),
        (25, "정적 2×2", "요약·그룹 비교·수치 분포 3개 세트"),
        (26, "Plotly", "막대 2·산점도·facet·treemap·parallel 6개"),
        (27, "범주 계산 기준", "인원·고소득 수·그룹 비율·전체 고소득 점유율"),
        (28, "상관분석", "수치+income 0·1 Pearson·Spearman, 인과 주의"),
        (29, "t-test 주제", "제시된 5개 주제 모두 실행"),
        (30, "t-test 선택", "정규성·Levene 후 Student/Welch, MW 보조"),
        (31, "검정 방향", "사전 정의한 양측 검정"),
        (32, "유의수준", "0.05 중심, 0.01·0.10 플래그 병기"),
        (33, "p-value 표현", "기각 또는 기각 근거 부족으로 표현"),
        (34, "효과 크기", "평균차·95% CI·Cohen's d와 p-value 병기"),
        (35, "카이제곱", "제시된 7개 범주와 income, Cramer's V 포함"),
        (36, "데이터 분할", "80:20, random_state=42, stratify"),
        (37, "기본 모델", "Logistic Regression"),
        (38, "불균형 처리", "class_weight=balanced"),
        (39, "수치 스케일", "StandardScaler"),
        (40, "인코딩", "feature OneHot, target 복사본 0·1"),
        (41, "평가 지표", "Accuracy·Precision·Recall·F1·ROC-AUC·CM"),
        (42, "최종 모델 선택", f"F1·설명 가능성 기준: {selected_model}"),
        (43, "모델 저장", "Pipeline·컬럼·타깃 매핑·메타데이터 bundle"),
    ]
    lines = [
        "# Adult Census Income 기준 반영 체크리스트",
        "",
        "| 번호 | 기준 | 구현 상태 |",
        "| ---: | --- | --- |",
    ]
    lines.extend(
        f"| {number} | {name} | ✅ {implementation} |"
        for number, name, implementation in rows
    )
    path = output_dir / "criteria_checklist.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_report(
    preparation: DataPreparationResult,
    eda: EDAResult,
    statistical: StatisticalResult,
    modeling: ModelingResult,
    static_paths: list[Path],
    plotly_paths: list[Path],
    checklist_path: Path,
    output_dir: Path,
) -> Path:
    """실제 분석 숫자·조건부 해석·산출물 링크를 report.md에 기록한다."""

    missing = (
        preparation.missing_counts[preparation.missing_counts > 0]
        .rename_axis("column")
        .reset_index(name="missing_count")
    )
    numeric_view = eda.numeric_summary.copy()
    numeric_view["missing_rate_pct"] = numeric_view["missing_rate"] * 100
    numeric_view["zero_rate_pct"] = numeric_view["zero_rate"] * 100
    numeric_view = numeric_view[
        [
            "variable",
            "count",
            "mean",
            "std",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "iqr",
            "skewness",
            "kurtosis",
            "missing_count",
            "missing_rate_pct",
            "unique_count",
            "zero_rate_pct",
            "iqr_outlier_candidates",
        ]
    ]
    t_view = statistical.t_tests[
        [
            "topic",
            "test",
            "mean_difference",
            "ci_95_low",
            "ci_95_high",
            "p_value_display",
            "cohens_d",
            "interpretation",
        ]
    ]
    chi_view = statistical.chi_square_tests[
        [
            "variable",
            "p_value_display",
            "cramers_v",
            "minimum_expected",
            "interpretation",
        ]
    ]
    confusion = modeling.final.confusion
    imputation_rows = [
        {
            "data_type": data_type,
            "column": column,
            "train_fitted_value": value,
        }
        for data_type, values in modeling.imputation_statistics.items()
        for column, value in values.items()
    ]
    imputation_view = pd.DataFrame(imputation_rows)
    static_links = "\n".join(
        f"- [{path.stem}](figures/{path.name})" for path in static_paths
    )
    plotly_links = "\n".join(
        f"- [{path.stem}](figures/{path.name})" for path in plotly_paths
    )
    report = f"""# Adult Census Income End-to-End 분석 보고서

- 작성일: {datetime.now().strftime("%Y-%m-%d")}
- 작성자: 광주 3반 한형준
- 데이터: adult.data 단일 파일
- 목표: 연 소득 >50K 이진 분류와 설명 가능한 분석 흐름 구성
- 전체 기준: [{checklist_path.name}]({checklist_path.name})

## 1. 분석 원칙

- 원본 하이픈 컬럼명과 income 문자열을 EDA까지 유지했습니다.
- 완전 중복 {preparation.duplicate_count:,}건만 제거하고 원본 파일은 보존했습니다.
- age, hours-per-week, capital-gain, capital-loss, fnlwgt와 희소 범주는 실제 가능한 관측으로 보고 변경하지 않았습니다.
- 결측 행은 삭제하지 않았습니다. 모델 결측 대체 기준은 train에서만 학습됩니다.
- EDA에서는 education을, 모델에서는 education-num을 사용해 중복 정보를 피했습니다.

## 2. Pandas·Polars 비교

{markdown_table(preparation.comparison)}

### 처리 전 결측치

{markdown_table(missing, digits=0)}

## 3. 수치형 EDA

{markdown_table(numeric_view)}

IQR 경계 밖 값은 오류 판정이 아니라 후보 수만 집계했습니다. 소득 그룹별 평균·중앙값과 Pearson·Spearman 상세 결과는 `tables/`의 CSV에 저장했습니다. 상관은 인과가 아닙니다.

## 4. 범주형 EDA

{markdown_table(eda.categorical_summary)}

표본 {MIN_GROUP_SIZE}건 미만 범주는 희소로 표시하지만 합치거나 삭제하지 않았습니다. [범주별 상세 표](tables/categorical_group_detail.csv)에는 전체 인원, >50K 인원, 그룹 내 >50K 비율, 전체 >50K 중 점유율이 포함됩니다.

## 5. 시각화

### 정적 차트

{static_links}

### Plotly 인터랙티브 차트

{plotly_links}

## 6. 독립표본 검정

모든 가설은 결과 확인 전에 양측으로 정의했습니다. 유의수준 0.05를 주 기준으로 사용하고 0.01·0.10 결과도 CSV에 기록했습니다.

{markdown_table(t_view)}

정규성·Levene 등분산성 결과를 확인해 Student 또는 Welch 검정을 선택하고, 비정규성에 대한 보조 확인으로 Mann–Whitney U 결과도 저장했습니다. 통계적 유의성과 실제 효과 크기는 구분해 해석해야 합니다.

## 7. 카이제곱 검정

{markdown_table(chi_view)}

카이제곱 p-value와 함께 연관성 크기인 Cramer's V, 기대빈도 진단을 기록했습니다. 연관성은 인과관계를 뜻하지 않습니다.

## 8. Logistic Regression Pipeline

- 분할: train {modeling.train_rows:,}행 / test {modeling.test_rows:,}행
- test_size=0.2, random_state=42, stratify=y
- 수치형: train 중앙값 대체 + StandardScaler
- 범주형: train 최빈값 대체 + OneHotEncoder
- 분류기: LogisticRegression, class_weight=balanced
- 최종 모델: {modeling.final.name}

### train에서 학습된 결측 대체값

아래 값은 80% train 데이터에 `fit`할 때만 계산됐으며 test 데이터에는 동일 값을 `transform`만 했습니다. 현재 데이터에는 수치형 결측이 없지만 실제 입력을 위한 중앙값 안전망도 Pipeline에 포함했습니다.

{markdown_table(imputation_view)}

### 모델 비교

{markdown_table(modeling.comparison)}

| 최종 평가 지표 | 값 |
| --- | ---: |
| Accuracy | {modeling.final.metrics["accuracy"]:.4f} |
| Precision | {modeling.final.metrics["precision"]:.4f} |
| Recall | {modeling.final.metrics["recall"]:.4f} |
| F1 | {modeling.final.metrics["f1"]:.4f} |
| ROC-AUC | {modeling.final.metrics["roc_auc"]:.4f} |

- Confusion Matrix: [[{confusion[0, 0]}, {confusion[0, 1]}], [{confusion[1, 0]}, {confusion[1, 1]}]]
- [Pipeline과 컬럼 bundle]({modeling.model_path.name})
- [모델 메타데이터]({modeling.metadata_path.name})

fnlwgt 포함·제외 모델을 F1과 설명 가능성으로 비교했습니다. race·sex 제외 모델은 민감 변수 제거 시 성능 안정성을 확인하기 위한 감사용 비교입니다.

## 9. 윤리적 한계

Adult 데이터는 과거 사회·노동 구조의 편향을 포함할 수 있습니다. race와 sex를 포함한 모델의 예측을 실제 의사결정에 바로 사용해서는 안 되며, 집단별 오류율과 공정성 검토가 추가로 필요합니다. fnlwgt는 개인 특성이라기보다 표본 가중치이므로 최종 선택 시 성능과 설명 가능성을 함께 고려했습니다.

## 10. 발표 요약

1. adult.data를 두 엔진으로 읽어 로딩 결과·결측·중복·groupby·시간·메모리를 검증했습니다.
2. 극단값과 희소 범주를 보존한 채 수치·범주 EDA와 정적·인터랙티브 시각화를 수행했습니다.
3. 다중 평균 검정과 카이제곱 검정에 효과 크기·신뢰구간·연관성 크기를 함께 제시했습니다.
4. split 이후 train에서만 전처리 기준을 학습하고 세 Logistic Pipeline을 동일 test에서 비교했습니다.
5. 최종 Pipeline, 원본 컬럼 목록, 타깃 매핑과 메타데이터를 함께 저장했습니다.
"""
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def cleanup_legacy_outputs(output_dir: Path) -> None:
    """새 결과와 혼동되는 기존 단일 차트 두 개만 제거한다."""

    for filename in ["adult_income_seaborn.png", "adult_income_plotly.html"]:
        (output_dir / filename).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    """데이터·산출물·벤치마크 옵션을 처리한다."""

    parser = argparse.ArgumentParser(
        description="Adult Census Income End-to-End 팀 프로젝트 분석"
    )
    parser.add_argument("--data", help="adult.data 경로")
    parser.add_argument("--output-dir", help="산출물 저장 폴더")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """데이터 준비부터 자동 보고서까지 전체 흐름을 실행한다."""

    args = parse_args()
    configure_plotting()
    output_dir, figure_dir, table_dir = prepare_output_dirs(args.output_dir)
    cleanup_legacy_outputs(output_dir)
    data_path = download_dataset(args.data, args.force_download)

    preparation = prepare_data(data_path, args.benchmark_repeats)
    eda = run_eda(preparation.deduplicated)
    static_paths = create_static_charts(preparation.deduplicated, eda, figure_dir)
    plotly_paths = create_plotly_charts(preparation.deduplicated, eda, figure_dir)
    statistical = run_statistical_analysis(preparation.deduplicated)
    modeling = train_models(preparation.deduplicated, output_dir)
    model_chart = create_model_evaluation_chart(modeling, figure_dir)
    static_paths.append(model_chart)
    table_paths = save_tables(preparation, eda, statistical, modeling, table_dir)
    checklist_path = create_criteria_checklist(output_dir, modeling.final.name)
    report_path = generate_report(
        preparation,
        eda,
        statistical,
        modeling,
        static_paths,
        plotly_paths,
        checklist_path,
        output_dir,
    )

    print("\n=== Adult Census Income 팀 프로젝트 완료 ===")
    print(f"정적 차트: {len(static_paths)}개")
    print(f"Plotly 차트: {len(plotly_paths)}개")
    print(f"분석 CSV: {len(table_paths)}개")
    print(f"최종 모델: {modeling.final.name}")
    print(
        f"F1={modeling.final.metrics['f1']:.4f}, ROC-AUC={modeling.final.metrics['roc_auc']:.4f}"
    )
    print(f"체크리스트: {checklist_path}")
    print(f"자동 보고서: {report_path}")


if __name__ == "__main__":
    main()
