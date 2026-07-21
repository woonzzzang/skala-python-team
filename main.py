from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
ASSET_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "models"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
RAW_DATA_PATH = DATA_DIR / "adult_raw.data"
CLEAN_DATA_PATH = OUTPUT_DIR / "cleaned_adult.csv"
EDA_SUMMARY_PATH = OUTPUT_DIR / "eda_summary.json"
SEABORN_CHART_PATH = ASSET_DIR / "seaborn_eda.png"
SEABORN_DISTRIBUTION_PATH = ASSET_DIR / "seaborn_distribution.png"
SEABORN_CORRELATION_PATH = ASSET_DIR / "seaborn_correlation.png"
SEABORN_GROUP_COMPARE_PATH = ASSET_DIR / "seaborn_group_compare.png"
PLOTLY_CHART_PATH = ASSET_DIR / "plotly_income_by_education.html"
PLOTLY_DISTRIBUTION_PATH = ASSET_DIR / "plotly_age_distribution.html"
PLOTLY_CORRELATION_PATH = ASSET_DIR / "plotly_numeric_correlation.html"
MODEL_PATH = MODEL_DIR / "income_pipeline.joblib"
REPORT_PATH = BASE_DIR / "report.md"

COLS = [
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

NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
OUTLIER_TARGET_COLS = [
    "age",
    "fnlwgt",
    "hours-per-week",
]
CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
TARGET_COL = "income"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def log_section(title: str) -> None:
    """실행 결과를 구분해서 보기 위한 로그"""
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def ensure_dirs() -> None:
    """분석 산출물 저장 폴더 생성"""
    for path in [DATA_DIR, OUTPUT_DIR, ASSET_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_raw_data() -> Path:
    """data.py에 제공된 URL 데이터를 로컬에 캐시"""
    ensure_dirs()

    # 네트워크 의존성 최소화
    # 최초 1회만 다운로드, 이후 로컬 캐시 재사용
    if not RAW_DATA_PATH.exists():
        logger.info("원본 데이터 다운로드: %s", DATA_URL)
        urlretrieve(DATA_URL, RAW_DATA_PATH)

    return RAW_DATA_PATH


def load_with_pandas(path: Path) -> pd.DataFrame:
    """Pandas로 Adult Income 데이터 로딩"""
    return pd.read_csv(
        path,
        header=None,
        names=COLS,
        na_values="?",
        skipinitialspace=True,
    )


def load_with_polars(path: Path) -> pl.DataFrame:
    """Polars로 Adult Income 데이터 로딩"""
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=COLS,
        null_values=["?", " ?"],
    )

    string_cols = [
        col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.String
    ]

    # UCI 원본 끝 빈 줄 방어
    # Polars는 빈 줄을 1행으로 읽을 수 있어 age null 행 제거
    return df.filter(pl.col("age").is_not_null()).with_columns(
        pl.col(string_cols).str.strip_chars()
    )


def compare_loaders(
    pandas_df: pd.DataFrame, polars_df: pl.DataFrame
) -> dict[str, object]:
    """Pandas와 Polars 로딩 결과 비교"""
    result = {
        "pandas_shape": pandas_df.shape,
        "polars_shape": polars_df.shape,
        "same_shape": pandas_df.shape == polars_df.shape,
        "pandas_missing": pandas_df.isna().sum().to_dict(),
        "polars_missing": polars_df.null_count().to_dicts()[0],
    }

    log_section("Pandas / Polars 로딩 비교")
    logger.info("Pandas shape: %s", result["pandas_shape"])
    logger.info("Polars shape: %s", result["polars_shape"])
    logger.info("shape 일치 여부: %s", result["same_shape"])

    return result


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """결측치와 중복을 처리한 분석용 데이터 생성"""
    before_rows = len(df)
    missing_before = df.isna().sum().to_dict()
    duplicate_before = int(df.duplicated().sum())

    # 중복은 같은 사람이라는 보장이 아니라 원본 중복 레코드
    # 분석 표본 중복 집계 방지 목적
    clean_df = df.drop_duplicates().copy()
    categorical_missing_cols = [
        col for col in CATEGORICAL_COLS if clean_df[col].isna().sum() > 0
    ]

    # 범주형 결측은 임의 추정 근거 부족
    # 행 삭제 대신 Unknown 그룹으로 보존
    for col in categorical_missing_cols:
        clean_df[col] = clean_df[col].fillna("Unknown")

    outlier_summary = cap_outliers_iqr(clean_df)

    # 모델 학습용 이진 라벨
    # >50K이면 1, 아니면 0
    clean_df[TARGET_COL] = clean_df[TARGET_COL].str.strip()
    clean_df["income_label"] = (clean_df[TARGET_COL] == ">50K").astype(int)

    summary = {
        "before_rows": before_rows,
        "duplicate_before": duplicate_before,
        "after_rows": len(clean_df),
        "removed_duplicates": before_rows - len(clean_df),
        "missing_before": missing_before,
        "missing_after": clean_df.isna().sum().to_dict(),
        "categorical_missing_filled": categorical_missing_cols,
        "outlier_treatment": outlier_summary,
    }

    log_section("결측치 / 중복 처리")
    logger.info("처리 전 행 수: %s", summary["before_rows"])
    logger.info("중복 행 수: %s", summary["duplicate_before"])
    logger.info("처리 후 행 수: %s", summary["after_rows"])
    logger.info("Unknown 대체 컬럼: %s", categorical_missing_cols)
    logger.info("IQR cap 적용 컬럼: %s", OUTLIER_TARGET_COLS)

    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    return clean_df, summary


def cap_outliers_iqr(df: pd.DataFrame) -> dict[str, object]:
    """IQR 기준 이상치를 행 삭제 대신 경계값으로 cap"""
    summary = {}

    # Adult 데이터의 수치형 중 일부는 사실상 범주/사건성 변수
    # education-num: education의 숫자 코드라 이상치 처리 대상 아님
    # capital-gain/loss: 0이 많고 고소득과 관련된 강한 신호라 cap 제외
    # age, fnlwgt, hours-per-week만 연속형에 가까워 IQR cap 적용
    for col in OUTLIER_TARGET_COLS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        mask = ~df[col].between(lo, hi)

        summary[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "lower": round(float(lo), 2),
            "upper": round(float(hi), 2),
            "outlier_count": int(mask.sum()),
            "method": "IQR capping",
        }

        # 행 제거 시 클래스 비율 왜곡 가능
        # 발표용 분석에서는 표본 보존을 위해 경계값으로 제한
        df[col] = df[col].clip(lower=lo, upper=hi)

    return summary


def build_eda_summary(
    df: pd.DataFrame,
    loader_summary: dict[str, object],
    cleaning_summary: dict[str, object],
) -> dict[str, object]:
    """기본 EDA 결과를 dict로 정리"""
    describe = df[NUMERIC_COLS].describe().round(2).to_dict()
    income_counts = df[TARGET_COL].value_counts().to_dict()
    income_rate = (df[TARGET_COL].value_counts(normalize=True) * 100).round(2).to_dict()

    summary = {
        "loader": loader_summary,
        "cleaning": cleaning_summary,
        "numeric_describe": describe,
        "income_counts": income_counts,
        "income_rate_percent": income_rate,
    }

    with EDA_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)

    log_section("기본 EDA")
    logger.info("소득 구간 건수: %s", income_counts)
    logger.info("소득 구간 비율: %s", income_rate)

    return summary


def create_seaborn_chart(df: pd.DataFrame) -> None:
    """Seaborn 정적 EDA 차트 저장"""
    sns.set_theme(style="whitegrid", font="AppleGothic")
    plt.rcParams["axes.unicode_minus"] = False

    # 대용량은 아니지만 차트 렌더링 안정성 목적 표본 사용
    sample_df = df.sample(min(8000, len(df)), random_state=42)
    corr = df[NUMERIC_COLS + ["income_label"]].corr()

    # 한 장 요약
    # 분포, 그룹 비교, 상관관계를 빠르게 훑는 발표용 차트
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.countplot(
        data=df,
        x=TARGET_COL,
        hue=TARGET_COL,
        palette="Set2",
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Income Class Count")
    axes[0, 0].set_xlabel("income")
    axes[0, 0].set_ylabel("count")

    sns.histplot(
        data=sample_df,
        x="age",
        hue=TARGET_COL,
        kde=True,
        bins=30,
        alpha=0.45,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Age Distribution by Income")
    axes[0, 1].set_xlabel("age")
    axes[0, 1].set_ylabel("count")

    sns.boxplot(
        data=sample_df,
        x=TARGET_COL,
        y="hours-per-week",
        hue=TARGET_COL,
        palette="Set3",
        legend=False,
        showfliers=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Hours per Week by Income")
    axes[1, 0].set_xlabel("income")
    axes[1, 0].set_ylabel("hours per week")

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Numeric Correlation")

    fig.tight_layout()
    fig.savefig(SEABORN_CHART_PATH, dpi=150)
    plt.close(fig)

    logger.info("Seaborn 차트 저장: %s", SEABORN_CHART_PATH)

    # 분포 차트
    # 수치형 변수별 모양 확인용
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS, strict=True):
        sns.histplot(
            data=sample_df,
            x=col,
            hue=TARGET_COL,
            kde=True,
            bins=30,
            alpha=0.45,
            ax=ax,
        )
        ax.set_title(f"{col} Distribution")
    fig.tight_layout()
    fig.savefig(SEABORN_DISTRIBUTION_PATH, dpi=150)
    plt.close(fig)

    # 상관관계 차트
    # 수치형 변수와 income_label 간 선형 관계 확인용
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Numeric Variables Correlation")
    fig.tight_layout()
    fig.savefig(SEABORN_CORRELATION_PATH, dpi=150)
    plt.close(fig)

    # 그룹 비교 차트
    # income 그룹별 수치형 변수 차이 확인용
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.ravel(), NUMERIC_COLS, strict=True):
        sns.boxplot(
            data=sample_df,
            x=TARGET_COL,
            y=col,
            hue=TARGET_COL,
            palette="Set3",
            legend=False,
            showfliers=False,
            ax=ax,
        )
        ax.set_title(f"{col} by Income")
    fig.tight_layout()
    fig.savefig(SEABORN_GROUP_COMPARE_PATH, dpi=150)
    plt.close(fig)

    logger.info("Seaborn 분포 차트 저장: %s", SEABORN_DISTRIBUTION_PATH)
    logger.info("Seaborn 상관관계 차트 저장: %s", SEABORN_CORRELATION_PATH)
    logger.info("Seaborn 그룹 비교 차트 저장: %s", SEABORN_GROUP_COMPARE_PATH)


def create_plotly_chart(df: pd.DataFrame) -> pd.DataFrame:
    """교육 수준별 고소득 비율 Plotly 차트 저장"""
    # 발표에서 설명하기 쉬운 축
    # 교육 수준별로 >50K 비율이 어떻게 달라지는지 확인
    education_rate = (
        df.groupby("education", observed=True)
        .agg(
            total=("income_label", "count"),
            high_income_rate=("income_label", "mean"),
        )
        .reset_index()
        .sort_values("high_income_rate", ascending=False)
    )
    education_rate["high_income_rate"] = (
        education_rate["high_income_rate"] * 100
    ).round(2)

    fig = px.bar(
        education_rate,
        x="education",
        y="high_income_rate",
        color="total",
        title="Education Level별 >50K 비율",
        labels={
            "education": "교육 수준",
            "high_income_rate": ">50K 비율(%)",
            "total": "표본 수",
        },
    )
    fig.write_html(PLOTLY_CHART_PATH)

    # 인터랙티브 분포
    # hover로 구간별 건수 확인 가능
    distribution_fig = px.histogram(
        df,
        x="age",
        color=TARGET_COL,
        nbins=40,
        barmode="overlay",
        title="Age Distribution by Income",
        labels={"age": "나이", TARGET_COL: "소득 구간"},
    )
    distribution_fig.write_html(PLOTLY_DISTRIBUTION_PATH)

    # 인터랙티브 상관관계
    # hover로 변수쌍 상관계수 확인 가능
    corr = df[NUMERIC_COLS + ["income_label"]].corr().round(4)
    corr_fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numeric Variables Correlation",
    )
    corr_fig.write_html(PLOTLY_CORRELATION_PATH)

    logger.info("Plotly 차트 저장: %s", PLOTLY_CHART_PATH)
    logger.info("Plotly 분포 차트 저장: %s", PLOTLY_DISTRIBUTION_PATH)
    logger.info("Plotly 상관관계 차트 저장: %s", PLOTLY_CORRELATION_PATH)

    return education_rate


def run_statistical_analysis(df: pd.DataFrame) -> dict[str, object]:
    """기술통계, 상관계수, t-test 수행"""
    numeric_corr = df[NUMERIC_COLS].corr().round(4)
    corr_with_target = (
        df[NUMERIC_COLS + ["income_label"]]
        .corr()["income_label"]
        .drop("income_label")
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .round(4)
        .reset_index()
    )
    corr_with_target.columns = ["variable", "income_label_corr"]

    # 변수쌍별 상관관계
    # 중복 조합 제거 후 절댓값 기준 정렬
    corr_pairs = []
    for i, col in enumerate(NUMERIC_COLS):
        for other_col in NUMERIC_COLS[i + 1 :]:
            corr_value = numeric_corr.loc[col, other_col]
            corr_pairs.append(
                {
                    "variable_a": col,
                    "variable_b": other_col,
                    "correlation": corr_value,
                    "abs_correlation": abs(corr_value),
                }
            )
    top_corr_pairs = (
        pd.DataFrame(corr_pairs)
        .sort_values("abs_correlation", ascending=False)
        .head(10)
        .round(4)
    )

    # 소득 그룹별 주당 근무시간 차이 검정
    # 연속형 변수이며 발표 해석이 쉬운 변수라 선택
    low_income = df.loc[df[TARGET_COL] == "<=50K", "hours-per-week"]
    high_income = df.loc[df[TARGET_COL] == ">50K", "hours-per-week"]
    t_stat, p_value = stats.ttest_ind(
        low_income,
        high_income,
        equal_var=False,
    )

    interpretation = (
        "두 소득 그룹의 주당 근무시간 평균 차이는 통계적으로 유의미함"
        if p_value < 0.05
        else "두 소득 그룹의 주당 근무시간 평균 차이는 통계적으로 유의미하지 않음"
    )

    result = {
        "numeric_describe": df[NUMERIC_COLS].describe().round(2).to_dict(),
        "correlation": numeric_corr.to_dict(),
        "target_correlation": corr_with_target.to_dict("records"),
        "top_correlation_pairs": top_corr_pairs.to_dict("records"),
        "ttest": {
            "target": "hours-per-week",
            "group_a": "<=50K",
            "group_b": ">50K",
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 8),
            "alpha": 0.05,
            "interpretation": interpretation,
        },
    }

    log_section("통계 분석")
    logger.info("t-statistic: %.4f", t_stat)
    logger.info("p-value: %.8f", p_value)
    logger.info("해석: %s", interpretation)

    return result


def build_ml_pipeline() -> Pipeline:
    """전처리와 모델을 포함한 sklearn Pipeline 생성"""
    # Pipeline 사용 이유
    # 전처리와 모델을 한 객체로 묶어 재현성, 저장, 재로딩 용이
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
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def train_evaluate_model(df: pd.DataFrame) -> dict[str, object]:
    """ML Pipeline 학습, 평가, 저장, 재로딩"""
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["income_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_ml_pipeline()
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    pred_proba = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, pred)
    balanced_accuracy = balanced_accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    matrix = confusion_matrix(y_test, pred)
    report = classification_report(
        y_test,
        pred,
        target_names=["<=50K", ">50K"],
        output_dict=True,
    )

    joblib.dump(pipeline, MODEL_PATH)
    loaded_pipeline = joblib.load(MODEL_PATH)
    reloaded_accuracy = loaded_pipeline.score(X_test, y_test)

    result = {
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": matrix.tolist(),
        "reloaded_accuracy": round(float(reloaded_accuracy), 4),
        "classification_report": report,
        "model_path": str(MODEL_PATH),
    }

    log_section("ML Pipeline 평가")
    logger.info("accuracy: %.4f", accuracy)
    logger.info("balanced accuracy: %.4f", balanced_accuracy)
    logger.info("precision: %.4f", precision)
    logger.info("recall: %.4f", recall)
    logger.info("f1-score: %.4f", f1)
    logger.info("roc-auc: %.4f", roc_auc)
    logger.info("confusion matrix: %s", matrix.tolist())
    logger.info("재로딩 후 accuracy: %.4f", reloaded_accuracy)
    logger.info("모델 저장: %s", MODEL_PATH)

    return result


def write_report(
    eda_summary: dict[str, object],
    education_rate: pd.DataFrame,
    stat_result: dict[str, object],
    ml_result: dict[str, object],
) -> None:
    """분석 결과를 report.md로 자동 생성"""
    ttest = stat_result["ttest"]
    income_counts = eda_summary["income_counts"]
    income_rate = eda_summary["income_rate_percent"]
    cleaning = eda_summary["cleaning"]
    outlier_treatment = make_nested_markdown_table(cleaning["outlier_treatment"])
    top_education = make_markdown_table(
        education_rate.head(5)[["education", "high_income_rate", "total"]]
    )
    target_correlation = make_markdown_table(
        pd.DataFrame(stat_result["target_correlation"])
    )
    top_correlation_pairs = make_markdown_table(
        pd.DataFrame(stat_result["top_correlation_pairs"])
    )

    report = f"""# Day2 종합 실습 - End2End 데이터 분석 프로젝트 결과

## 1. 데이터 개요

- 데이터: UCI Adult Income
- 원본 행 수: {cleaning["before_rows"]:,}
- 중복 제거 후 행 수: {cleaning["after_rows"]:,}
- 제거된 중복 행 수: {cleaning["removed_duplicates"]:,}
- 목적 변수: `income`

## 2. 데이터 준비 결과

Pandas와 Polars로 데이터를 모두 로딩해 shape 일치 여부를 확인했습니다.

- Pandas shape: {eda_summary["loader"]["pandas_shape"]}
- Polars shape: {eda_summary["loader"]["polars_shape"]}
- shape 일치 여부: {eda_summary["loader"]["same_shape"]}

결측치는 범주형 컬럼에서 발생했으며, 분석에서 행을 삭제하지 않기 위해 `Unknown`으로 대체했습니다.

대체 컬럼:

```text
{cleaning["categorical_missing_filled"]}
```

이상치는 `age`, `fnlwgt`, `hours-per-week`에 한해 IQR 기준으로 capping 처리했습니다.

`education-num`은 교육 수준 코드이고, `capital-gain`, `capital-loss`는 0이 많은 사건성 변수라 이상치 제거 대상에서 제외했습니다.

이상치 처리 요약:

{outlier_treatment}

## 3. EDA 요약

소득 구간별 건수:

```text
{income_counts}
```

소득 구간별 비율:

```text
{income_rate}
```

Seaborn EDA 차트:

![seaborn eda](./assets/seaborn_eda.png)

Seaborn 분포 차트:

![seaborn distribution](./assets/seaborn_distribution.png)

Seaborn 상관관계 차트:

![seaborn correlation](./assets/seaborn_correlation.png)

Seaborn 그룹 비교 차트:

![seaborn group compare](./assets/seaborn_group_compare.png)

Plotly 인터랙티브 차트:

[교육 수준별 고소득 비율](./assets/plotly_income_by_education.html)

[나이 분포](./assets/plotly_age_distribution.html)

[수치형 변수 상관관계](./assets/plotly_numeric_correlation.html)

교육 수준별 고소득 비율 TOP 5:

{top_education}

## 4. 통계 분석

`income <=50K` 그룹과 `income >50K` 그룹의 `hours-per-week` 평균 차이를 t-test로 검정했습니다.

| 항목 | 값 |
| --- | --- |
| t-statistic | {ttest["t_statistic"]} |
| p-value | {ttest["p_value"]} |
| 기준 | p < {ttest["alpha"]} |
| 해석 | {ttest["interpretation"]} |

수치형 변수와 `income_label`의 상관관계:

{target_correlation}

수치형 변수쌍 상관관계 TOP 10:

{top_correlation_pairs}

## 5. ML Pipeline 결과

`ColumnTransformer`와 `Pipeline`을 사용해 전처리와 모델 학습을 하나의 객체로 구성했습니다.

| 평가 지표 | 값 |
| --- | --- |
| accuracy | {ml_result["accuracy"]} |
| balanced accuracy | {ml_result["balanced_accuracy"]} |
| precision | {ml_result["precision"]} |
| recall | {ml_result["recall"]} |
| F1-score | {ml_result["f1_score"]} |
| ROC-AUC | {ml_result["roc_auc"]} |
| 재로딩 후 accuracy | {ml_result["reloaded_accuracy"]} |
| 모델 파일 | `{MODEL_PATH.name}` |

confusion matrix:

```text
{ml_result["confusion_matrix"]}
```

## 6. 발표용 핵심 해석

- Adult Income 데이터는 `<=50K` 클래스가 더 많은 불균형 데이터입니다.
- 교육 수준이 높을수록 `>50K` 비율이 높아지는 경향이 있습니다.
- t-test 결과, 소득 그룹 간 주당 근무시간 평균 차이는 통계적으로 유의미합니다.
- Pipeline 구조로 전처리와 모델을 묶어 재사용성과 저장 가능성을 확보했습니다.

## 7. 아쉬운 점과 추가 개선

아쉬운 점은 첫 모델을 Logistic Regression 하나로만 구성해 다양한 모델 비교까지는 수행하지 못했다는 점입니다.

추가로 더 해볼 수 있는 것은 RandomForest, XGBoost 등 모델 비교, class imbalance 보정, 교차검증, SHAP 기반 변수 중요도 해석입니다.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("자동 리포트 저장: %s", REPORT_PATH)


def make_markdown_table(df: pd.DataFrame) -> str:
    """tabulate 없이 간단한 Markdown 표 생성"""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def make_nested_markdown_table(data: dict[str, object]) -> str:
    """중첩 dict를 간단한 Markdown 표로 변환"""
    rows = []

    for col, values in data.items():
        if not isinstance(values, dict):
            continue

        rows.append(
            {
                "column": col,
                "lower": values["lower"],
                "upper": values["upper"],
                "outlier_count": values["outlier_count"],
                "method": values["method"],
            }
        )

    return make_markdown_table(pd.DataFrame(rows))


def main() -> int:
    ensure_dirs()
    raw_path = ensure_raw_data()

    pandas_df = load_with_pandas(raw_path)
    polars_df = load_with_polars(raw_path)
    loader_summary = compare_loaders(pandas_df, polars_df)

    clean_df, cleaning_summary = clean_data(pandas_df)
    eda_summary = build_eda_summary(clean_df, loader_summary, cleaning_summary)

    create_seaborn_chart(clean_df)
    education_rate = create_plotly_chart(clean_df)
    stat_result = run_statistical_analysis(clean_df)
    ml_result = train_evaluate_model(clean_df)
    write_report(eda_summary, education_rate, stat_result, ml_result)

    assert loader_summary["same_shape"] is True
    assert CLEAN_DATA_PATH.exists()
    assert SEABORN_CHART_PATH.exists()
    assert SEABORN_DISTRIBUTION_PATH.exists()
    assert SEABORN_CORRELATION_PATH.exists()
    assert SEABORN_GROUP_COMPARE_PATH.exists()
    assert PLOTLY_CHART_PATH.exists()
    assert PLOTLY_DISTRIBUTION_PATH.exists()
    assert PLOTLY_CORRELATION_PATH.exists()
    assert MODEL_PATH.exists()
    assert REPORT_PATH.exists()

    log_section("End2End 분석 프로젝트 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
