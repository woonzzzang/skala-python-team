"""Adult 데이터셋을 다운로드/로드하고 Pandas와 Polars 결과를 비교하는 모듈."""

import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl


def download_or_load_data(raw_path: Path, url: str) -> Path:
    """raw_path에 파일이 이미 있으면 그대로 사용하고, 없으면 url에서 다운로드한다."""
    if raw_path.exists():
        print(f"로컬 원본 파일을 사용합니다: {raw_path}")
        return raw_path

    print(f"로컬 파일이 없어 데이터를 다운로드합니다: {url}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, raw_path)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise RuntimeError(f"데이터 다운로드에 실패했습니다: {error}") from error

    print(f"다운로드 완료: {raw_path}")
    return raw_path


def load_with_pandas(raw_path: Path, columns: list[str]) -> pd.DataFrame:
    """Pandas로 Adult 원본 데이터를 읽는다. 헤더가 없고 '?' 결측 표시를 반영한다."""
    try:
        df = pd.read_csv(
            raw_path,
            names=columns,
            skipinitialspace=True,  # ", State-gov"처럼 쉼표 뒤 공백 제거
            na_values="?",  # 결측은 '?'로 표시되어 있음
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {raw_path}") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"CSV 파일 형식을 읽는 중 오류가 발생했습니다: {error}") from error

    # 파일 끝의 빈 줄 등으로 생긴 완전 결측 행 제거
    df = df.dropna(how="all").reset_index(drop=True)

    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"필수 컬럼이 없습니다: {missing_columns}")

    return df


def load_with_polars(raw_path: Path, columns: list[str], numeric_columns: list[str]) -> pl.DataFrame:
    """
    Polars Lazy API로 같은 데이터를 읽는다.
    scan_csv -> with_columns(정리) -> filter -> collect 흐름을 따른다.
    Polars에는 pandas의 skipinitialspace 옵션이 없어 문자열을 직접 strip 처리한다.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {raw_path}")

    strip_columns = [col for col in columns if col != "age"]  # age는 첫 컬럼이라 공백이 없음

    try:
        lazy_frame = pl.scan_csv(raw_path, has_header=False, new_columns=columns, null_values="?")

        cleaned_frame = (
            lazy_frame
            # pandas skipinitialspace=True와 같은 효과: 문자열 앞뒤 공백 제거
            .with_columns([pl.col(col).str.strip_chars().alias(col) for col in strip_columns])
            # strip 이후에도 남아있는 '?' 문자열을 결측(None)으로 통일
            .with_columns(
                [
                    pl.when(pl.col(col) == "?").then(None).otherwise(pl.col(col)).alias(col)
                    for col in strip_columns
                ]
            )
            # 공백 때문에 문자열로 읽혔던 숫자형 컬럼을 다시 정수로 변환
            .with_columns([pl.col(col).cast(pl.Int64, strict=False).alias(col) for col in numeric_columns])
            # 파일 끝의 빈 줄로 생긴 완전 결측 행 제거
            .filter(pl.col("age").is_not_null())
        )

        return cleaned_frame.collect()

    except pl.exceptions.ComputeError as error:
        raise ValueError(f"Polars로 CSV를 읽는 중 오류가 발생했습니다: {error}") from error


def compare_pandas_polars(pandas_df: pd.DataFrame, polars_df: pl.DataFrame, income_column: str = "income") -> None:
    """Pandas와 Polars로 각각 읽은 결과의 주요 지표가 일치하는지 비교 출력한다."""
    print("\n[Pandas vs Polars 비교]")
    print(f"행 수 - Pandas: {pandas_df.shape[0]} / Polars: {polars_df.shape[0]}")
    print(f"열 수 - Pandas: {pandas_df.shape[1]} / Polars: {polars_df.shape[1]}")

    print("\n[컬럼별 결측치 수 비교]")
    for col in ["workclass", "occupation", "native-country"]:
        pandas_na = int(pandas_df[col].isna().sum())
        polars_na = int(polars_df[col].null_count())
        flag = "일치" if pandas_na == polars_na else "불일치"
        print(f"{col}: Pandas={pandas_na}, Polars={polars_na} ({flag})")

    print("\n[income 범주별 개수 비교]")
    pandas_counts = pandas_df[income_column].value_counts().sort_index()
    polars_counts = (
        polars_df.group_by(income_column)
        .len()
        .sort(income_column)
        .to_pandas()
        .set_index(income_column)["len"]
    )
    for category in pandas_counts.index:
        p_count = int(pandas_counts[category])
        q_count = int(polars_counts.get(category, 0))
        flag = "일치" if p_count == q_count else "불일치"
        print(f"{category}: Pandas={p_count}, Polars={q_count} ({flag})")

    print("\n[중복 제거 후 행 수 비교]")
    pandas_dedup_rows = pandas_df.drop_duplicates().shape[0]
    polars_dedup_rows = polars_df.unique().shape[0]
    flag = "일치" if pandas_dedup_rows == polars_dedup_rows else "불일치"
    print(f"Pandas={pandas_dedup_rows}, Polars={polars_dedup_rows} ({flag})")


def compare_load_performance(raw_path: Path, columns: list[str], numeric_columns: list[str]) -> dict:
    """
    Pandas와 Polars로 같은 파일을 각각 새로 읽어 로드 시간과 메모리 사용량을 1회 측정해 비교한다.
    주의: 같은 프로세스 내 1회 측정값이며(OS 파일 캐시 영향 가능), 이 결과만으로
    두 라이브러리의 절대적 우열을 단정하지 않는다.
    """
    start = time.perf_counter()
    pandas_df = load_with_pandas(raw_path, columns)
    pandas_seconds = time.perf_counter() - start
    pandas_memory_mb = pandas_df.memory_usage(deep=True).sum() / (1024**2)

    start = time.perf_counter()
    polars_df = load_with_polars(raw_path, columns, numeric_columns)
    polars_seconds = time.perf_counter() - start
    polars_memory_mb = polars_df.estimated_size() / (1024**2)

    print("\n[Pandas vs Polars 실행시간·메모리 비교] (1회 측정, 절대적 우열 판단 근거 아님)")
    print(f"로드 시간   - Pandas: {pandas_seconds:.4f}초 / Polars: {polars_seconds:.4f}초")
    print(f"메모리 사용량 - Pandas: {pandas_memory_mb:.2f}MB / Polars: {polars_memory_mb:.2f}MB")

    return {
        "pandas_seconds": float(pandas_seconds),
        "polars_seconds": float(polars_seconds),
        "pandas_memory_mb": float(pandas_memory_mb),
        "polars_memory_mb": float(polars_memory_mb),
    }
