"""
Adult Census Income End-to-End 분석 프로젝트 실행 스크립트.

python src/main.py 로 실행하면 다음을 순서대로 수행한다.
데이터 다운로드/로드 -> Pandas/Polars 비교 -> 정제 -> EDA -> 정적/인터랙티브 시각화
-> 통계 분석(상관/검정) -> sklearn Pipeline 학습/평가 -> 모델 저장/재로딩 -> report.md 생성
"""

import time
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
# 모델 입력에서는 education을 제외한다: education-num이 같은 정보를 서열형 숫자로 담고 있어
# 둘 다 넣으면 정보가 중복된다. EDA/시각화에서는 education(범주형 표기)을 그대로 사용한다.
MODEL_CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col != "education"]
KEY_MISSING_COLUMNS = ["workclass", "occupation", "native-country"]
TARGET_COLUMN = "income"


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

        # 2. Pandas / Polars로 각각 로드하고 결과 비교 (실행 시간도 함께 측정)
        pandas_load_start = time.perf_counter()
        pandas_df = data_loader.load_with_pandas(raw_path, COLUMNS)
        pandas_load_seconds = time.perf_counter() - pandas_load_start

        polars_load_start = time.perf_counter()
        polars_df = data_loader.load_with_polars(raw_path, COLUMNS, NUMERIC_FEATURES)
        polars_load_seconds = time.perf_counter() - polars_load_start

        data_loader.compare_pandas_polars(
            pandas_df, polars_df, TARGET_COLUMN, pandas_load_seconds, polars_load_seconds
        )

        # 3. 기본 점검, 결측치 처리 방법론 설명, 정제
        preprocessing.inspect_data(pandas_df, KEY_MISSING_COLUMNS)
        missing_strategy_summary = preprocessing.explain_missing_value_strategy()
        cleaned_df = preprocessing.clean_for_eda(pandas_df)

        cleaned_csv_path = paths["processed_dir"] / "adult_cleaned.csv"
        cleaned_df.to_csv(cleaned_csv_path, index=False)
        print(f"\n정제 데이터 CSV 저장 완료: {cleaned_csv_path}")

        # 4. 자료형별 EDA (수치형 / 범주형 / 타깃)
        numeric_summary = eda.perform_numeric_eda(cleaned_df, NUMERIC_FEATURES, TARGET_COLUMN)
        eda.perform_categorical_eda(cleaned_df, CATEGORICAL_FEATURES + [TARGET_COLUMN], income_column=TARGET_COLUMN)
        income_distribution = eda.analyze_income_distribution(cleaned_df, TARGET_COLUMN)

        # 5. 정적 시각화 (Seaborn/Matplotlib)
        visualization.configure_korean_font()
        income_png = visualization.create_income_distribution_plot(
            cleaned_df, paths["figures_dir"] / "income_distribution.png"
        )
        numeric_png = visualization.create_numeric_eda_plot(cleaned_df, paths["figures_dir"] / "numeric_eda.png")
        additional_distribution_png = visualization.create_additional_distribution_plot(
            cleaned_df, paths["figures_dir"] / "additional_distributions.png"
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
        t_test_result = stats_analysis.perform_t_test(cleaned_df)

        # 추가 t-test 주제: 정규성/등분산성을 확인해 검정 방법을 자동으로 고르는 perform_two_group_test 사용
        top_workclass_groups = cleaned_df["workclass"].value_counts().head(2).index.tolist()
        additional_t_test_results = [
            stats_analysis.perform_two_group_test(cleaned_df, TARGET_COLUMN, ">50K", "<=50K", "age"),
            stats_analysis.perform_two_group_test(cleaned_df, "sex", "Male", "Female", "hours-per-week"),
            stats_analysis.perform_two_group_test(
                cleaned_df, "workclass", top_workclass_groups[0], top_workclass_groups[1], "hours-per-week"
            ),
            stats_analysis.perform_two_group_test(cleaned_df, TARGET_COLUMN, ">50K", "<=50K", "education-num"),
        ]

        # 선택적 추가 분석: 카이제곱 독립성 검정 (t-test는 필수, 이건 완성도를 위한 추가 분석)
        chi_square_results = [
            stats_analysis.perform_chi_square_test(cleaned_df, "education", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "occupation", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "sex", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "workclass", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "marital-status", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "race", TARGET_COLUMN),
            stats_analysis.perform_chi_square_test(cleaned_df, "native-country", TARGET_COLUMN),
        ]

        # 7. Plotly 인터랙티브 시각화
        plotly_html_path = visualization.create_plotly_visualization(
            cleaned_df, paths["interactive_dir"] / "adult_income_analysis.html"
        )
        plotly_education_html_path = visualization.create_plotly_education_income_bar(
            cleaned_df, paths["interactive_dir"] / "education_income_ratio.html"
        )
        plotly_occupation_html_path = visualization.create_plotly_occupation_income_bar(
            cleaned_df, paths["interactive_dir"] / "occupation_income_ratio.html"
        )
        plotly_facet_html_path = visualization.create_plotly_facet_scatter(
            cleaned_df, paths["interactive_dir"] / "occupation_facet_scatter.html"
        )
        plotly_treemap_html_path = visualization.create_plotly_treemap(
            cleaned_df, paths["interactive_dir"] / "occupation_education_income_treemap.html"
        )
        plotly_parallel_html_path = visualization.create_plotly_parallel_categories(
            cleaned_df, paths["interactive_dir"] / "sex_workclass_income_parallel_categories.html"
        )

        # 8. sklearn Pipeline 학습 및 평가 (income은 feature에서 제외해 데이터 누수 방지)
        # EDA에서는 income을 문자열("<=50K"/">50K") 그대로 쓰고, 모델의 y만 0/1로 매핑해서 사용한다.
        feature_columns = NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
        X = cleaned_df[feature_columns]
        y = (cleaned_df[TARGET_COLUMN] == ">50K").astype(int)

        candidate_pipelines = modeling.build_candidate_pipelines(NUMERIC_FEATURES, MODEL_CATEGORICAL_FEATURES)
        model_results = {
            name: modeling.train_and_evaluate_model(pipeline, X, y)
            for name, pipeline in candidate_pipelines.items()
        }
        final_model_name = modeling.select_final_model(model_results)
        model_result = model_results[final_model_name]

        confusion_png = visualization.create_confusion_matrix_plot(
            model_result["confusion_matrix"],
            model_result["confusion_matrix_labels"],
            paths["figures_dir"] / "confusion_matrix.png",
        )
        roc_curve_png = visualization.create_roc_curve_plot(
            model_result["y_test"],
            model_result["y_proba"],
            paths["figures_dir"] / "roc_curve.png",
        )

        # 9. 모델 저장 및 재로딩 검증 (컬럼 목록도 함께 저장)
        model_path = paths["models_dir"] / "adult_income_pipeline.pkl"
        modeling.save_and_verify_model(
            model_result["pipeline"], model_result["X_test"], model_path, feature_columns
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
            "income": income_distribution,
            "numeric_summary": numeric_summary,
            "correlation_top_pairs": correlation_top_pairs,
            "t_test": t_test_result,
            "additional_t_tests": additional_t_test_results,
            "chi_square_tests": chi_square_results,
            "model": {
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": MODEL_CATEGORICAL_FEATURES,
                "final_model_name": final_model_name,
                "candidate_comparison": {
                    name: {
                        "accuracy": result["accuracy"],
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "f1_score": result["f1_score"],
                        "roc_auc": result["roc_auc"],
                    }
                    for name, result in model_results.items()
                },
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
                "additional_distribution_png": additional_distribution_png.relative_to(project_root),
                "categorical_eda_png": categorical_png.relative_to(project_root),
                "correlation_heatmap_png": correlation_png.relative_to(project_root),
                "confusion_matrix_png": confusion_png.relative_to(project_root),
                "roc_curve_png": roc_curve_png.relative_to(project_root),
                "plotly_html": plotly_html_path.relative_to(project_root),
                "plotly_education_html": plotly_education_html_path.relative_to(project_root),
                "plotly_occupation_html": plotly_occupation_html_path.relative_to(project_root),
                "plotly_facet_html": plotly_facet_html_path.relative_to(project_root),
                "plotly_treemap_html": plotly_treemap_html_path.relative_to(project_root),
                "plotly_parallel_html": plotly_parallel_html_path.relative_to(project_root),
                "model_pkl": model_path.relative_to(project_root),
            },
        }
        generate_markdown_report(context, paths["report_path"])

        print("\n모든 작업이 완료되었습니다.")

    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"\n분석 파이프라인 실행 중 오류가 발생했습니다: {error}")


if __name__ == "__main__":
    main()
