"""전처리+모델을 하나로 묶은 sklearn Pipeline 구성, 학습/평가, 저장/재로딩을 담당하는 모듈."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_ml_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """
    수치형/범주형 전처리와 LogisticRegression을 하나의 Pipeline으로 연결한다.
    결측치 대체(imputation)를 Pipeline 내부에 두는 이유는, train 데이터로만 대체 규칙을
    학습하게 해서 test 정보가 섞여 들어가는 데이터 누수를 막기 위함이다.
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

    # 이진 분류에 적합하고 해석이 비교적 쉬운 LogisticRegression을 기본 모델로 사용
    # class_weight="balanced"로 income 클래스 불균형을 보정
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return model_pipeline


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
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # ">50K"에 대한 예측 확률

    positive_label = ">50K"
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=positive_label)
    recall = recall_score(y_test, y_pred, pos_label=positive_label)
    f1 = f1_score(y_test, y_pred, pos_label=positive_label)
    roc_auc = roc_auc_score((y_test == positive_label).astype(int), y_proba)
    report_text = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred, labels=[positive_label, "<=50K"])

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
        "confusion_matrix_labels": [positive_label, "<=50K"],
    }


def save_and_verify_model(pipeline: Pipeline, X_test: pd.DataFrame, model_path: Path) -> None:
    """학습된 전체 Pipeline(전처리+모델)을 저장하고, 다시 불러와 예측이 동일한지 확인한다."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    before_save_predictions = pipeline.predict(X_test.head(5))

    joblib.dump(pipeline, model_path)
    print(f"\nPipeline 저장 완료: {model_path}")

    loaded_pipeline = joblib.load(model_path)
    after_load_predictions = loaded_pipeline.predict(X_test.head(5))

    predictions_match = np.array_equal(before_save_predictions, after_load_predictions)
    print(f"저장 전 예측: {before_save_predictions}")
    print(f"재로딩 후 예측: {after_load_predictions}")
    print(f"저장 전/후 예측 일치 여부: {predictions_match}")

    if not predictions_match:
        raise ValueError("저장 전과 재로딩 후 예측 결과가 다릅니다. 모델 저장 과정을 확인하세요.")
