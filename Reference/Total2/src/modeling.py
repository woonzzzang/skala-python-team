"""전처리+모델을 하나로 묶은 sklearn Pipeline 구성, 학습/평가, 저장/재로딩을 담당하는 모듈."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_ml_pipeline(
    numeric_features: list[str], categorical_features: list[str], classifier: ClassifierMixin | None = None
) -> Pipeline:
    """
    수치형/범주형 전처리와 분류기를 하나의 Pipeline으로 연결한다.
    결측치 대체(imputation)를 Pipeline 내부에 두는 이유는, train 데이터로만 대체 규칙을
    학습하게 해서 test 정보가 섞여 들어가는 데이터 누수를 막기 위함이다.
    classifier를 지정하지 않으면 기본 모델인 LogisticRegression을 사용한다.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    if classifier is None:
        # 이진 분류에 적합하고 해석이 비교적 쉬운 LogisticRegression을 기본 모델로 사용
        # class_weight="balanced"로 income 클래스 불균형을 보정
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return model_pipeline


def build_candidate_pipelines(numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    """
    비교 대상 모델 후보들의 Pipeline을 만든다.
    - LogisticRegression: 해석이 쉬운 기본 모델
    - RandomForest: 비선형 관계를 잡을 수 있는 비교 모델(해석 가능성은 상대적으로 낮음)
    """
    return {
        "LogisticRegression": build_ml_pipeline(numeric_features, categorical_features),
        "RandomForest": build_ml_pipeline(
            numeric_features,
            categorical_features,
            classifier=RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42
            ),
        ),
    }


def select_final_model(model_results: dict[str, dict[str, Any]], interpretability_tolerance: float = 0.01) -> str:
    """
    F1-score(>50K 기준)를 최우선 기준으로 최종 모델을 고른다.
    다만 F1-score 차이가 interpretability_tolerance 이내로 작으면, 성능 이득이 미미한 것으로 보고
    해석 가능성이 더 좋은 LogisticRegression을 최종 모델로 선택한다.
    """
    print("\n[모델 비교 및 최종 선택]")
    for name, result in model_results.items():
        print(
            f"{name}: Accuracy={result['accuracy']:.4f}, Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}, F1={result['f1_score']:.4f}, ROC-AUC={result['roc_auc']:.4f}"
        )

    best_name = max(model_results, key=lambda name: model_results[name]["f1_score"])
    best_f1 = model_results[best_name]["f1_score"]

    if (
        "LogisticRegression" in model_results
        and best_name != "LogisticRegression"
        and best_f1 - model_results["LogisticRegression"]["f1_score"] <= interpretability_tolerance
    ):
        logistic_f1 = model_results["LogisticRegression"]["f1_score"]
        print(
            f"{best_name}의 F1-score가 근소하게 높지만({best_f1:.4f} vs {logistic_f1:.4f}), "
            f"차이가 {interpretability_tolerance} 이내라 해석 가능성이 더 좋은 LogisticRegression을 "
            "최종 모델로 선택한다."
        )
        best_name = "LogisticRegression"
    else:
        print(f"F1-score 기준 최종 선택 모델: {best_name} (F1={best_f1:.4f})")

    return best_name


def train_and_evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """train/test로 나눈 뒤 Pipeline을 학습하고, 여러 평가 지표를 계산해 반환한다."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # class 1(>50K)에 대한 예측 확률

    # y는 호출부에서 이미 <=50K=0, >50K=1로 매핑된 값이라 pos_label=1이 ">50K" 기준이다.
    positive_label = 1
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=positive_label)
    recall = recall_score(y_test, y_pred, pos_label=positive_label)
    f1 = f1_score(y_test, y_pred, pos_label=positive_label)
    roc_auc = roc_auc_score(y_test, y_proba)
    report_text = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    confusion = confusion_matrix(y_test, y_pred, labels=[positive_label, 0])

    print("\n[모델 평가 결과 - income 분류]")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (>50K 기준)")
    print(f"Recall   : {recall:.4f} (>50K 기준)")
    print(f"F1-score : {f1:.4f} (>50K 기준)")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\n[classification_report]")
    print(report_text)
    print("[confusion_matrix] (행/열 순서: >50K, <=50K)")
    print(confusion)

    print(
        "\n참고: income 클래스가 불균형하기 때문에 Accuracy만 보면 다수 클래스(<=50K)만 "
        "잘 맞혀도 점수가 높게 나올 수 있다. 그래서 Precision, Recall, F1-score를 함께 확인해야 한다."
    )

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "classification_report": report_text,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": [">50K", "<=50K"],
    }


def save_and_verify_model(
    pipeline: Pipeline, X_test: pd.DataFrame, model_path: Path, feature_columns: list[str]
) -> None:
    """
    학습된 전체 Pipeline(전처리+모델)과 학습에 사용한 컬럼 목록(feature_columns)을 함께 저장하고,
    다시 불러와 예측이 동일한지 확인한다. 컬럼 목록을 같이 저장해 두면, 재로딩 후 예측할 때
    입력 DataFrame의 컬럼 순서/구성을 이 목록과 대조해서 확인할 수 있다.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    before_save_predictions = pipeline.predict(X_test.head(5))

    bundle = {"pipeline": pipeline, "feature_columns": feature_columns}
    joblib.dump(bundle, model_path)
    print(f"\nPipeline + 컬럼 목록 저장 완료: {model_path}")

    loaded_bundle = joblib.load(model_path)
    loaded_pipeline = loaded_bundle["pipeline"]
    loaded_feature_columns = loaded_bundle["feature_columns"]
    after_load_predictions = loaded_pipeline.predict(X_test[loaded_feature_columns].head(5))

    predictions_match = np.array_equal(before_save_predictions, after_load_predictions)
    print(f"저장 전 예측: {before_save_predictions}")
    print(f"재로딩 후 예측: {after_load_predictions}")
    print(f"저장 전/후 예측 일치 여부: {predictions_match}")
    print(f"저장된 컬럼 목록: {loaded_feature_columns}")

    if not predictions_match:
        raise ValueError("저장 전과 재로딩 후 예측 결과가 다릅니다. 모델 저장 과정을 확인하세요.")
