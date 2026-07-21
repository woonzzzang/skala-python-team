"""
작성일: 2026-07-21
작성자: 광주 3반 한형준
작성 목적: Adult Census Income 데이터로 Pandas·Polars 데이터 준비부터
          시각화, 통계 분석, ML Pipeline, 분석 보고서 자동화까지 익히기 위함

프로그램 전체 설명:
UCI Adult Census Income 원본을 내려받아 Pandas와 Polars로 각각 로딩한다.
두 도구의 크기·결측치·중복 처리 결과를 비교하고 기본 EDA와 기술통계를
출력한다. Seaborn 정적 차트와 Plotly 인터랙티브 차트를 저장하고, 소득 그룹별
평균 연령을 t-test로 검정해 p-value를 해석한다. 수치형·범주형 전처리와
LogisticRegression을 sklearn Pipeline으로 묶어 학습·평가한 뒤 joblib으로
저장한다. 마지막으로 모든 분석 결과와 산출물 링크를 report.md에 자동 기록한다.

변경 내역:
- 2026-07-21: UCI Adult 데이터 자동 다운로드 및 Pandas·Polars 로딩 작성
- 2026-07-21: 결측치·중복 처리, 기본 EDA, 기술통계·상관분석 작성
- 2026-07-21: Seaborn·Plotly 시각화와 t-test·p-value 해석 추가
- 2026-07-21: sklearn Pipeline 평가·저장 및 report.md 자동 생성 추가
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
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
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# UCI Adult Census Income 원본과 열 이름입니다.
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
TARGET_COLUMN = "income"
LOW_INCOME = "<=50K"
HIGH_INCOME = ">50K"
RANDOM_STATE = 42

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
    """Pandas·Polars 데이터 준비 결과와 비교 지표."""

    pandas_raw: pd.DataFrame
    polars_raw: pl.DataFrame
    pandas_deduplicated: pd.DataFrame
    pandas_clean: pd.DataFrame
    polars_clean: pl.DataFrame
    missing_before: pd.Series
    duplicate_count: int


@dataclass(frozen=True)
class TTestResult:
    """독립표본 t-test 결과."""

    statistic: float
    p_value: float
    significant: bool
    interpretation: str


@dataclass(frozen=True)
class ModelResult:
    """분류 Pipeline 평가 결과와 저장 정보."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    model_path: Path
    train_rows: int
    test_rows: int


def configure_plotting() -> None:
    """macOS 환경에서 한글이 깨지지 않도록 글꼴과 스타일을 설정한다."""

    font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])


def prepare_output_dir(output_argument: str | None = None) -> Path:
    """차트·모델·보고서를 저장할 폴더를 준비한다."""

    output_dir = (
        Path(output_argument).expanduser().resolve()
        if output_argument
        else SCRIPT_DIR / "day2_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def download_dataset(
    data_argument: str | None,
    force_download: bool = False,
) -> Path:
    """사용자 지정 파일을 사용하거나 UCI 원본을 로컬에 안전하게 저장한다."""

    if data_argument:
        data_path = Path(data_argument).expanduser().resolve()
        if not data_path.is_file():
            raise FileNotFoundError(f"지정한 Adult 데이터 파일이 없습니다: {data_path}")
        return data_path

    data_path = SCRIPT_DIR / "adult.data"
    if data_path.is_file() and not force_download:
        print(f"로컬 캐시 사용: {data_path}")
        return data_path

    print(f"UCI Adult 데이터 다운로드: {DATA_URL}")
    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    temporary_path = data_path.with_suffix(".data.tmp")
    try:
        with urlopen(request, timeout=60) as response, temporary_path.open("wb") as file:
            shutil.copyfileobj(response, file)
        temporary_path.replace(data_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    if data_path.stat().st_size == 0:
        raise RuntimeError("다운로드한 Adult 데이터 파일이 비어 있습니다.")
    return data_path


def load_with_pandas(data_path: Path) -> pd.DataFrame:
    """Adult 데이터를 Pandas DataFrame으로 읽고 문자열 공백·? 결측치를 정리한다."""

    frame = pd.read_csv(
        data_path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
    )
    string_columns = CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    for column in string_columns:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace("?", pd.NA)
    return frame


def load_with_polars(data_path: Path) -> pl.DataFrame:
    """같은 Adult 데이터를 Polars DataFrame으로 읽고 동일하게 정규화한다."""

    frame = pl.read_csv(
        data_path,
        has_header=False,
        new_columns=COLUMNS,
        schema_overrides={column: pl.String for column in COLUMNS},
        infer_schema_length=10_000,
    )
    string_columns = CATEGORICAL_COLUMNS + [TARGET_COLUMN]
    # UCI 파일은 값 앞에 공백이 있고 마지막에 빈 줄이 있으므로 먼저 정규화합니다.
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
            for column in string_columns
        ]
    )
    frame = frame.with_columns(
        [pl.col(column).cast(pl.Int64).alias(column) for column in NUMERIC_COLUMNS]
    )
    return frame


def prepare_data(data_path: Path) -> DataPreparationResult:
    """두 도구로 로딩·중복 제거·결측치 대체를 수행하고 결과를 검증한다."""

    pandas_raw = load_with_pandas(data_path)
    polars_raw = load_with_polars(data_path)

    if pandas_raw.shape != polars_raw.shape:
        raise AssertionError(
            f"로딩 결과 크기가 다릅니다: Pandas={pandas_raw.shape}, "
            f"Polars={polars_raw.shape}"
        )
    if list(pandas_raw.columns) != polars_raw.columns:
        raise AssertionError("Pandas와 Polars의 열 이름이 다릅니다.")

    missing_before = pandas_raw.isna().sum()
    pandas_deduplicated = pandas_raw.drop_duplicates().reset_index(drop=True)
    polars_deduplicated = polars_raw.unique(maintain_order=True)
    duplicate_count = len(pandas_raw) - len(pandas_deduplicated)
    polars_duplicate_count = polars_raw.height - polars_deduplicated.height
    if duplicate_count != polars_duplicate_count:
        raise AssertionError("Pandas와 Polars의 중복 행 수가 다릅니다.")

    # EDA용 정제본은 범주형 결측치를 해당 열의 최빈값으로 대체합니다.
    modes = {
        column: str(pandas_deduplicated[column].mode(dropna=True).iloc[0])
        for column in CATEGORICAL_COLUMNS
    }
    pandas_clean = pandas_deduplicated.copy()
    pandas_clean[CATEGORICAL_COLUMNS] = pandas_clean[CATEGORICAL_COLUMNS].fillna(modes)

    polars_clean = polars_deduplicated.with_columns(
        [pl.col(column).fill_null(modes[column]).alias(column) for column in CATEGORICAL_COLUMNS]
    )

    pandas_null_after = int(pandas_clean.isna().sum().sum())
    polars_null_after = int(polars_clean.null_count().to_numpy().sum())
    if pandas_null_after != 0 or polars_null_after != 0:
        raise AssertionError("결측치 처리 후에도 null 값이 남아 있습니다.")
    if pandas_clean.shape != polars_clean.shape:
        raise AssertionError("정제 후 Pandas와 Polars 결과 크기가 다릅니다.")

    print("\n=== 데이터 준비: Pandas vs Polars ===")
    print(f"Pandas 로딩 크기: {pandas_raw.shape}")
    print(f"Polars 로딩 크기: {polars_raw.shape}")
    print(f"중복 행: {duplicate_count:,}건 → 제거 후 {len(pandas_clean):,}행")
    print("[처리 전 결측치]")
    print(missing_before[missing_before > 0].to_string())
    print("처리 후 결측치: Pandas=0, Polars=0")

    print("\n[Pandas 상위 3행]")
    print(pandas_clean.head(3).to_string(index=False))
    print("\n[Polars 상위 3행]")
    print(polars_clean.head(3))
    print("\n[Pandas dtype]")
    print(pandas_clean.dtypes.to_string())
    print("\n[Polars schema]")
    print(polars_clean.schema)

    return DataPreparationResult(
        pandas_raw=pandas_raw,
        polars_raw=polars_raw,
        pandas_deduplicated=pandas_deduplicated,
        pandas_clean=pandas_clean,
        polars_clean=polars_clean,
        missing_before=missing_before,
        duplicate_count=duplicate_count,
    )


def basic_eda(clean_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """수치형 기술통계, 소득 분포, 상관계수를 계산하고 출력한다."""

    statistics_table = (
        clean_frame[NUMERIC_COLUMNS]
        .describe()
        .loc[["mean", "std", "25%", "50%", "75%"]]
        .transpose()
        .reset_index(names="variable")
    )
    correlation_table = clean_frame[NUMERIC_COLUMNS].corr()
    income_distribution = clean_frame[TARGET_COLUMN].value_counts().sort_index()

    print("\n=== 기본 EDA·기술통계 ===")
    print("[평균·표준편차·분위수]")
    print(statistics_table.round(3).to_string(index=False))
    print("\n[소득 클래스 분포]")
    print(income_distribution.to_string())
    print("\n[수치형 변수 상관계수]")
    print(correlation_table.round(3).to_string())
    return statistics_table, correlation_table


def create_seaborn_chart(clean_frame: pd.DataFrame, output_dir: Path) -> Path:
    """소득 그룹별 연령 분포를 비교하는 Seaborn 정적 차트를 저장한다."""

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=clean_frame,
        x=TARGET_COLUMN,
        y="age",
        hue=TARGET_COLUMN,
        order=[LOW_INCOME, HIGH_INCOME],
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.set_title("Adult Census Income: 소득 그룹별 연령 분포", fontsize=16)
    ax.set_xlabel("연 소득 그룹")
    ax.set_ylabel("연령(세)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / "adult_income_seaborn.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSeaborn 정적 차트 저장: {output_path}")
    return output_path


def create_plotly_chart(clean_frame: pd.DataFrame, output_dir: Path) -> Path:
    """교육 수준별 고소득 비율을 비교하는 Plotly 인터랙티브 차트를 저장한다."""

    chart_data = clean_frame.copy()
    chart_data["high_income"] = (chart_data[TARGET_COLUMN] == HIGH_INCOME).astype(int)
    education_summary = (
        chart_data.groupby(["education", "education-num"], observed=True)
        .agg(
            high_income_rate=("high_income", "mean"),
            people=("high_income", "size"),
            average_hours=("hours-per-week", "mean"),
        )
        .reset_index()
        .sort_values("education-num")
    )
    education_summary["high_income_percent"] = (
        education_summary["high_income_rate"] * 100
    )

    fig = px.bar(
        education_summary,
        x="education",
        y="high_income_percent",
        color="high_income_percent",
        color_continuous_scale="Blues",
        hover_data={
            "people": ":,d",
            "average_hours": ":.1f",
            "education-num": True,
            "high_income_rate": False,
        },
        labels={
            "education": "교육 수준",
            "high_income_percent": "고소득자 비율(%)",
            "people": "표본 수",
            "average_hours": "주당 평균 근무시간",
            "education-num": "교육 수준 수치",
        },
        title="Adult Census Income: 교육 수준별 >50K 소득 비율",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="교육 수준",
        yaxis_title="고소득자 비율(%)",
        xaxis_tickangle=-35,
        coloraxis_colorbar_title="고소득 비율(%)",
    )

    output_path = output_dir / "adult_income_plotly.html"
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    print(f"Plotly 인터랙티브 차트 저장: {output_path}")
    return output_path


def format_p_value(p_value: float) -> str:
    """수치 언더플로로 0이 된 매우 작은 p-value를 의미 있게 표시한다."""

    return "< 1e-300" if p_value == 0 else f"{p_value:.6g}"


def perform_t_test(clean_frame: pd.DataFrame, alpha: float = 0.05) -> TTestResult:
    """저소득·고소득 그룹의 평균 연령 차이를 Welch t-test로 검정한다."""

    low_income_age = clean_frame.loc[
        clean_frame[TARGET_COLUMN] == LOW_INCOME, "age"
    ].dropna()
    high_income_age = clean_frame.loc[
        clean_frame[TARGET_COLUMN] == HIGH_INCOME, "age"
    ].dropna()
    if low_income_age.empty or high_income_age.empty:
        raise ValueError("한 소득 그룹에 표본이 없어 t-test를 수행할 수 없습니다.")

    statistic, p_value = stats.ttest_ind(
        low_income_age,
        high_income_age,
        equal_var=False,
        nan_policy="omit",
    )
    significant = bool(p_value < alpha)
    if significant:
        interpretation = (
            "p-value < 0.05이므로 귀무가설을 기각합니다. "
            "두 소득 그룹의 평균 연령에는 통계적으로 유의한 차이가 있습니다."
        )
    else:
        interpretation = (
            "p-value >= 0.05이므로 귀무가설을 기각할 수 없습니다. "
            "두 소득 그룹의 평균 연령 차이가 통계적으로 유의하지 않습니다."
        )

    print("\n=== 독립표본 t-test: 소득 그룹별 평균 연령 ===")
    print(f"<=50K 평균 연령: {low_income_age.mean():.3f}")
    print(f">50K 평균 연령: {high_income_age.mean():.3f}")
    print(f"t={statistic:.4f}, p={format_p_value(float(p_value))}")
    print(f"해석: {interpretation}")
    return TTestResult(float(statistic), float(p_value), significant, interpretation)


def train_pipeline(
    deduplicated_frame: pd.DataFrame,
    output_dir: Path,
) -> ModelResult:
    """전처리와 LogisticRegression을 Pipeline으로 묶어 평가·저장한다."""

    model_frame = deduplicated_frame.dropna(subset=[TARGET_COLUMN]).copy()
    X = model_frame.drop(columns=TARGET_COLUMN)
    # Pandas 3의 pd.NA를 sklearn SimpleImputer가 처리하는 표준 np.nan으로 통일합니다.
    X[CATEGORICAL_COLUMNS] = X[CATEGORICAL_COLUMNS].astype(object).where(
        X[CATEGORICAL_COLUMNS].notna(),
        np.nan,
    )
    y = (model_frame[TARGET_COLUMN] == HIGH_INCOME).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Pipeline 내부에서 결측치 대체·스케일링·원-핫 인코딩·분류를 순서대로 실행합니다.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    confusion = confusion_matrix(y_test, predictions)

    model_path = output_dir / "adult_income_pipeline.joblib"
    joblib.dump(model, model_path)
    loaded_model: Pipeline = joblib.load(model_path)
    if not np.array_equal(
        predictions[:20],
        loaded_model.predict(X_test.head(20)),
    ):
        raise RuntimeError("저장 전 모델과 재로딩 모델의 예측 결과가 다릅니다.")

    print("\n=== sklearn 분류 Pipeline ===")
    print(f"학습 데이터: {len(X_train):,}행 / 평가 데이터: {len(X_test):,}행")
    print(f"정확도(Accuracy): {accuracy:.4f}")
    print(f"정밀도(Precision): {precision:.4f}")
    print(f"재현율(Recall): {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print("[Confusion Matrix]")
    print(confusion)
    print("[Classification Report]")
    print(classification_report(y_test, predictions, target_names=[LOW_INCOME, HIGH_INCOME]))
    print(f"Pipeline 저장·재로딩 확인: {model_path}")

    return ModelResult(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion=confusion,
        model_path=model_path,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )


def markdown_table(frame: pd.DataFrame, float_digits: int = 3) -> str:
    """추가 패키지 없이 작은 DataFrame을 Markdown 표 문자열로 변환한다."""

    display_frame = frame.copy()
    for column in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].map(
                lambda value: f"{value:.{float_digits}f}"
            )

    headers = [str(column).replace("|", "\\|") for column in display_frame.columns]
    rows = [
        [str(value).replace("|", "\\|") for value in row]
        for row in display_frame.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def generate_report(
    preparation: DataPreparationResult,
    statistics_table: pd.DataFrame,
    correlation_table: pd.DataFrame,
    t_test: TTestResult,
    model: ModelResult,
    seaborn_path: Path,
    plotly_path: Path,
    output_dir: Path,
) -> Path:
    """분석 결과·해석·산출물 링크를 포함한 report.md를 자동 생성한다."""

    missing_table = (
        preparation.missing_before[preparation.missing_before > 0]
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    correlation_for_report = correlation_table.round(3).reset_index(names="variable")
    confusion = model.confusion

    report = f"""# Adult Census Income 종합 분석 보고서

- 작성일: {datetime.now().strftime('%Y-%m-%d')}
- 작성자: 광주 3반 한형준
- 데이터 출처: [UCI Machine Learning Repository - Adult]({DATA_URL})
- 분석 목표: 연 소득이 50K를 초과하는지 예측하는 이진 분류

## 1. 데이터 준비 및 Pandas·Polars 비교

- 원본 데이터: {preparation.pandas_raw.shape[0]:,}행 × {preparation.pandas_raw.shape[1]}열
- Pandas 로딩 결과: {preparation.pandas_raw.shape}
- Polars 로딩 결과: {preparation.polars_raw.shape}
- 발견한 중복 행: {preparation.duplicate_count:,}건
- 중복 제거 후 데이터: {len(preparation.pandas_clean):,}행
- 결측치 처리: `?`를 null로 변환한 후 범주형 열의 최빈값으로 대체
- 처리 후 결측치: Pandas 0건, Polars 0건

### 처리 전 결측치

{markdown_table(missing_table, float_digits=0)}

## 2. 기술통계

평균·표준편차와 25%·50%·75% 분위수를 산출했습니다.

{markdown_table(statistics_table)}

## 3. 수치형 변수 상관계수

{markdown_table(correlation_for_report)}

상관계수는 선형 관계의 방향과 강도를 나타내며, 인과관계를 의미하지 않습니다.

## 4. 시각화

### Seaborn 정적 차트

소득 그룹별 연령 분포를 박스플롯으로 비교했습니다.

![소득 그룹별 연령 분포]({seaborn_path.name})

### Plotly 인터랙티브 차트

교육 수준별 고소득자 비율과 표본 수·주당 평균 근무시간을 확인할 수 있습니다.

[인터랙티브 차트 열기]({plotly_path.name})

## 5. 독립표본 t-test

- 검정 대상: `<=50K`와 `>50K` 소득 그룹의 평균 연령
- t 통계량: {t_test.statistic:.4f}
- p-value: {format_p_value(t_test.p_value)}
- 유의수준: 0.05
- 해석: {t_test.interpretation}

## 6. ML Pipeline 및 평가

수치형 열에는 중앙값 대체와 표준화를, 범주형 열에는 최빈값 대체와 원-핫 인코딩을 적용했습니다.
이 전처리와 LogisticRegression을 하나의 sklearn `Pipeline` 객체로 구성했습니다.

| 평가 지표 | 값 |
| --- | ---: |
| Accuracy | {model.accuracy:.4f} |
| Precision | {model.precision:.4f} |
| Recall | {model.recall:.4f} |
| F1 score | {model.f1:.4f} |

- 학습 데이터: {model.train_rows:,}행
- 평가 데이터: {model.test_rows:,}행
- Confusion Matrix: `[[{confusion[0, 0]}, {confusion[0, 1]}], [{confusion[1, 0]}, {confusion[1, 1]}]]`
- 저장 모델: [{model.model_path.name}]({model.model_path.name})

## 7. 발표 요약

1. Pandas와 Polars는 동일한 32,561행·15열을 로딩했으며 결측치·중복 처리 결과도 일치했습니다.
2. 연령 분포와 교육 수준별 고소득 비율을 정적·인터랙티브 차트로 비교했습니다.
3. t-test로 두 소득 그룹의 평균 연령 차이에 대한 통계적 유의성을 확인했습니다.
4. 전처리와 분류기를 Pipeline으로 묶어 재현 가능한 모델 파일로 저장했습니다.
"""

    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"report.md 자동 생성: {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    """데이터 경로·산출물 경로·재다운로드 옵션을 처리한다."""

    parser = argparse.ArgumentParser(
        description="Adult Census Income Day 2 종합실습 자동 분석"
    )
    parser.add_argument("--data", help="로컬 adult.data 경로(생략 시 UCI에서 다운로드)")
    parser.add_argument("--output-dir", help="차트·모델·report.md 저장 폴더")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="로컬 캐시가 있어도 UCI 원본을 다시 다운로드",
    )
    return parser.parse_args()


def main() -> None:
    """데이터 준비부터 보고서 생성까지 전체 분석 흐름을 실행한다."""

    args = parse_args()
    configure_plotting()
    output_dir = prepare_output_dir(args.output_dir)
    data_path = download_dataset(args.data, args.force_download)

    print(f"분석 파일: {data_path}")
    print(f"산출물 폴더: {output_dir}")

    preparation = prepare_data(data_path)
    statistics_table, correlation_table = basic_eda(preparation.pandas_clean)
    seaborn_path = create_seaborn_chart(preparation.pandas_clean, output_dir)
    plotly_path = create_plotly_chart(preparation.pandas_clean, output_dir)
    t_test = perform_t_test(preparation.pandas_clean)
    model = train_pipeline(preparation.pandas_deduplicated, output_dir)
    report_path = generate_report(
        preparation,
        statistics_table,
        correlation_table,
        t_test,
        model,
        seaborn_path,
        plotly_path,
        output_dir,
    )

    print("\n=== Day 2 종합실습 완료 ===")
    print("1) Pandas·Polars 로딩/정제/EDA 비교 완료")
    print("2) Seaborn PNG·Plotly HTML 생성 완료")
    print("3) 기술통계·상관분석·t-test 해석 완료")
    print("4) sklearn Pipeline 평가·joblib 저장 완료")
    print(f"5) 자동 보고서 생성 완료: {report_path}")


if __name__ == "__main__":
    main()
