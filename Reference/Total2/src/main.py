"""
Adult Census Income End-to-End 분석 프로젝트 실행 스크립트.

python src/main.py 로 실행하면 다음을 순서대로 수행한다.
데이터 다운로드/로드 -> Pandas/Polars 비교 -> 정제 -> EDA -> 정적/인터랙티브 시각화
-> 통계 분석(상관/검정) -> sklearn Pipeline 학습/평가 -> 모델 저장/재로딩 -> report.md 생성
"""

from pathlib import Path

import data_loader
import eda
import modeling
import preprocessing
import stats_analysis
import visualization
from report import generate_markdown_report

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

NUMERIC_FEATURES = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
KEY_MISSING_COLUMNS = ["workclass", "occupation", "native-country"]
TARGET_COLUMN = "income"
TARGET_MAPPING = {"<=50K": 0, ">50K": 1}

# 모델 feature 목록: EDA에는 fnlwgt/education을 그대로 쓰지만, 모델 feature에서는 제외한다.
# - fnlwgt: 개인 속성이 아니라 표본 가중치(census weight)라서 일반 예측 feature로 부적합해 제외
# - education: education-num과 같은 학력 정보를 중복 제공하므로, 모델에는 education-num만 사용
MODEL_NUMERIC_FEATURES = [col for col in NUMERIC_FEATURES if col != "fnlwgt"]
MODEL_CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col != "education"]
EXCLUDED_MODEL_COLUMNS = ["fnlwgt", "education"]


def configure_paths(project_root: Path) -> dict[str, Path]:
    """프로젝트에서 사용할 주요 경로를 정의하고 출력 폴더를 미리 만든다."""
    paths = {
        "raw_data": project_root / "data" / "raw" / "adult.data",
        "processed_dir": project_root / "data" / "processed",
        "figures_dir": project_root / "output" / "figures",
        "interactive_dir": project_root / "output" / "interactive",
        "models_dir": project_root / "output" / "models",
        "report_path": project_root / "output" / "report.md",
    }
    for key in ("processed_dir", "figures_dir", "interactive_dir", "models_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def schedule_daily_run() -> None:
    """
    (선택 기능) schedule 라이브러리로 매일 같은 시각에 main()을 자동 실행하는 예시.
    기본 실행(main.py 직접 실행) 시에는 이 함수를 호출하지 않는다.
    필요할 때만 별도로 `python -c "from main import schedule_daily_run; schedule_daily_run()"` 형태로 실행한다.
    """
    import time

    import schedule

    schedule.every().day.at("06:00").do(main)
    print("매일 06:00에 main()을 실행하도록 예약했습니다. Ctrl+C로 종료할 수 있습니다.")
    while True:
        schedule.run_pending()
        time.sleep(60)


def main() -> None:
    """전체 분석 파이프라인을 순서대로 실행한다."""
    project_root = Path(__file__).resolve().parent.parent
    paths = configure_paths(project_root)

    try:
        # 1. 데이터 다운로드 또는 로드
        raw_path = data_loader.download_or_load_data(paths["raw_data"], DATA_URL)

        # 2. Pandas / Polars로 각각 로드하고 결과 비교
        pandas_df = data_loader.load_with_pandas(raw_path, COLUMNS)
        polars_df = data_loader.load_with_polars(raw_path, COLUMNS, NUMERIC_FEATURES)
        data_loader.compare_pandas_polars(pandas_df, polars_df, TARGET_COLUMN)
        load_performance = data_loader.compare_load_performance(raw_path, COLUMNS, NUMERIC_FEATURES)

        # 3. 기본 점검, 결측치 처리 방법론 설명, 정제
        preprocessing.inspect_data(pandas_df, KEY_MISSING_COLUMNS)
        missing_strategy_summary = preprocessing.explain_missing_value_strategy()
        cleaned_df = preprocessing.clean_for_eda(pandas_df)

        cleaned_csv_path = paths["processed_dir"] / "adult_cleaned.csv"
        cleaned_df.to_csv(cleaned_csv_path, index=False)
        print(f"\n정제 데이터 CSV 저장 완료: {cleaned_csv_path}")

        # 4. 자료형별 EDA (수치형 / 범주형 / 타깃)
        numeric_summary = eda.perform_numeric_eda(cleaned_df, NUMERIC_FEATURES)
        income_numeric_summary = eda.summarize_numeric_by_income(cleaned_df, NUMERIC_FEATURES, TARGET_COLUMN)
        eda.perform_categorical_eda(cleaned_df, CATEGORICAL_FEATURES + [TARGET_COLUMN], income_column=TARGET_COLUMN)
        income_distribution = eda.analyze_income_distribution(cleaned_df, TARGET_COLUMN)

        # 5. 정적 시각화 (Seaborn/Matplotlib)
        visualization.configure_korean_font()
        income_png = visualization.create_income_distribution_plot(
            cleaned_df, paths["figures_dir"] / "income_distribution.png"
        )
        numeric_png = visualization.create_numeric_eda_plot(cleaned_df, paths["figures_dir"] / "numeric_eda.png")
        hours_dist_png = visualization.create_hours_distribution_plot(
            cleaned_df, paths["figures_dir"] / "hours_distribution.png"
        )
        categorical_png = visualization.create_categorical_eda_plot(
            cleaned_df, paths["figures_dir"] / "categorical_eda.png"
        )
        visualization.show_static_figures_note()

        # 6. 기술통계/상관분석 및 통계 검정
        correlation_matrix = stats_analysis.perform_correlation_analysis(
            cleaned_df, NUMERIC_FEATURES, TARGET_COLUMN
        )
        correlation_png = visualization.create_correlation_heatmap(
            correlation_matrix, paths["figures_dir"] / "correlation_heatmap.png"
        )
        # t-test 필수 3주제: income 그룹별 hours-per-week / age / education-num 평균 차이
        t_test_results = {
            "hours-per-week": stats_analysis.perform_t_test(cleaned_df, value_column="hours-per-week"),
            "age": stats_analysis.perform_t_test(cleaned_df, value_column="age"),
            "education-num": stats_analysis.perform_t_test(cleaned_df, value_column="education-num"),
        }
        t_test_result = t_test_results["hours-per-week"]  # report.py 하위 호환을 위한 대표값

        # 카이제곱 독립성 검정: education/occupation/workclass/sex x income
        chi_square_results = [
            stats_analysis.perform_chi_square_test(cleaned_df, "education", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "occupation", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "workclass", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "sex", TARGET_COLUMN),
        ]

        # 7. Plotly 인터랙티브 시각화
        plotly_html_path = visualization.create_plotly_visualization(
            cleaned_df, paths["interactive_dir"] / "adult_income_analysis.html"
        )
        plotly_education_path = visualization.create_plotly_income_ratio_bar(
            cleaned_df, "education", paths["interactive_dir"] / "education_income_ratio.html"
        )
        plotly_occupation_path = visualization.create_plotly_income_ratio_bar(
            cleaned_df, "occupation", paths["interactive_dir"] / "occupation_income_ratio.html"
        )

        # 8. sklearn Pipeline 학습 및 평가
        # income은 feature에서 제외해 데이터 누수 방지, y는 모델 학습/평가용으로만 0/1 매핑(EDA는 문자열 유지)
        # fnlwgt(표본가중치)와 education(education-num과 정보 중복)은 모델 feature에서 제외
        X = cleaned_df[MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES]
        y = cleaned_df[TARGET_COLUMN].map(TARGET_MAPPING)

        comparison_result = modeling.compare_classifiers(MODEL_NUMERIC_FEATURES, MODEL_CATEGORICAL_FEATURES, X, y)
        model_result = comparison_result["final"]

        confusion_png = visualization.create_confusion_matrix_plot(
            model_result["confusion_matrix"],
            model_result["confusion_matrix_labels"],
            paths["figures_dir"] / "confusion_matrix.png",
        )
        roc_png = visualization.create_roc_curve_plot(
            model_result["y_test"],
            model_result["y_proba"],
            model_result["roc_auc"],
            paths["figures_dir"] / "roc_curve.png",
        )

        # 민감 변수(sex) 그룹별 예측 성능/양성 예측 비율 참고 분석 (공정성 라이브러리 없이 최소 분석)
        sex_group_performance = modeling.analyze_group_performance(
            model_result["pipeline"], model_result["X_test"], model_result["y_test"], group_column="sex"
        )

        # 9. 모델 저장 및 재로딩 검증 (feature 목록/제외 컬럼/타깃 매핑/설정을 메타데이터로 함께 저장)
        classifier = model_result["pipeline"].named_steps["classifier"]
        model_path = paths["models_dir"] / "adult_income_pipeline.pkl"
        model_metadata = {
            "model_name": model_result["model_name"],
            "input_features": {
                "numeric": MODEL_NUMERIC_FEATURES,
                "categorical": MODEL_CATEGORICAL_FEATURES,
            },
            "excluded_columns": EXCLUDED_MODEL_COLUMNS,
            "target_mapping": TARGET_MAPPING,
            "model_params": {
                key: value
                for key, value in classifier.get_params().items()
                if isinstance(value, (str, int, float, bool)) or value is None
            },
            "selection_reason": comparison_result["selection_reason"],
        }
        modeling.save_and_verify_model(
            model_result["pipeline"], model_result["X_test"], model_path, metadata=model_metadata
        )

        # 10. report.md 자동 생성 (아래 context의 모든 값은 위에서 실제로 계산된 결과)
        correlation_top_pairs = stats_analysis.get_top_correlation_pairs(correlation_matrix, top_n=3)
        context = {
            "data": {
                "raw_row_count": len(pandas_df),
                "raw_col_count": pandas_df.shape[1],
                "cleaned_row_count": len(cleaned_df),
                "cleaned_col_count": cleaned_df.shape[1],
                "duplicate_count": int(pandas_df.duplicated().sum()),
                "key_missing_counts": {
                    col: int(pandas_df[col].isna().sum()) for col in KEY_MISSING_COLUMNS
                },
                "missing_strategy_summary": missing_strategy_summary,
            },
            "load_performance": load_performance,
            "income": income_distribution,
            "numeric_summary": numeric_summary,
            "correlation_top_pairs": correlation_top_pairs,
            "t_test": t_test_result,
            "t_test_results": t_test_results,
            "chi_square_tests": chi_square_results,
            "model_comparison": comparison_result["comparison_table"],
            "model_selection_reason": comparison_result["selection_reason"],
            "sex_group_performance": sex_group_performance,
            "model": {
                "numeric_features": MODEL_NUMERIC_FEATURES,
                "categorical_features": MODEL_CATEGORICAL_FEATURES,
                "excluded_columns": EXCLUDED_MODEL_COLUMNS,
                "model_name": model_result["model_name"],
                "accuracy": model_result["accuracy"],
                "precision": model_result["precision"],
                "recall": model_result["recall"],
                "f1_score": model_result["f1_score"],
                "roc_auc": model_result["roc_auc"],
            },
            "output_files": {
                "cleaned_csv": cleaned_csv_path.relative_to(project_root),
                "income_distribution_png": income_png.relative_to(project_root),
                "numeric_eda_png": numeric_png.relative_to(project_root),
                "hours_distribution_png": hours_dist_png.relative_to(project_root),
                "categorical_eda_png": categorical_png.relative_to(project_root),
                "correlation_heatmap_png": correlation_png.relative_to(project_root),
                "confusion_matrix_png": confusion_png.relative_to(project_root),
                "roc_curve_png": roc_png.relative_to(project_root),
                "plotly_html": plotly_html_path.relative_to(project_root),
                "plotly_education_html": plotly_education_path.relative_to(project_root),
                "plotly_occupation_html": plotly_occupation_path.relative_to(project_root),
                "model_pkl": model_path.relative_to(project_root),
                "model_metadata_json": model_path.with_suffix(".json").relative_to(project_root),
            },
        }
        generate_markdown_report(context, paths["report_path"])

        print("\n모든 작업이 완료되었습니다.")

    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"\n분석 파이프라인 실행 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    main()
