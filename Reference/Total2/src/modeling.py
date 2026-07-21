"""전처리+모델을 하나로 묶은 sklearn Pipeline 구성, 학습/평가, 저장/재로딩을 담당하는 모듈."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    수치형/범주형 전처리를 하나의 ColumnTransformer로 묶는다.
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def build_ml_pipeline(
    numeric_features: list[str], categorical_features: list[str], classifier: Any = None
) -> Pipeline:
    """
    전처리와 분류 모델을 하나의 Pipeline으로 연결한다.
    classifier를 지정하지 않으면 해석이 비교적 쉬운 LogisticRegression(class_weight="balanced")을 기본값으로 쓴다.
    """
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    if classifier is None:
        # 이진 분류에 적합하고 해석이 쉬운 LogisticRegression을 기본 모델로 사용
        # class_weight="balanced"로 income 클래스 불균형을 보정
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    train/test로 나눈 뒤 Pipeline을 학습하고, 여러 평가 지표를 계산해 반환한다.
    y는 <=50K=0, >50K=1로 매핑된 정수 Series여야 하며, 양성 클래스는 1(>50K)로 고정한다.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # ">50K"(1)에 대한 예측 확률

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)
    report_text = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])
    confusion = confusion_matrix(y_test, y_pred, labels=[1, 0])

    print(f"\n[모델 평가 결과 - income 분류: {model_name}]")
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
        "model_name": model_name,
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


def compare_classifiers(
    numeric_features: list[str],
    categorical_features: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    f1_tie_margin: float = 0.01,
) -> dict[str, Any]:
    """
    LogisticRegression(class_weight="balanced")과 RandomForestClassifier(class_weight="balanced")를
    같은 train/test 분할로 비교한다. >50K 클래스의 F1-score를 기준으로 최종 모델을 선택하되,
    두 모델의 F1 차이가 f1_tie_margin 미만이면 설명 가능성이 더 좋은 LogisticRegression을 선택한다.
    """
    logistic_pipeline = build_ml_pipeline(
        numeric_features,
        categorical_features,
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    random_forest_pipeline = build_ml_pipeline(
        numeric_features,
        categorical_features,
        RandomForestClassifier(class_weight="balanced", random_state=42),
    )

    logistic_result = train_and_evaluate_model(logistic_pipeline, X, y, model_name="LogisticRegression")
    random_forest_result = train_and_evaluate_model(random_forest_pipeline, X, y, model_name="RandomForest")

    comparison_df = pd.DataFrame(
        [
            {
                "model": result["model_name"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "roc_auc": result["roc_auc"],
            }
            for result in (logistic_result, random_forest_result)
        ]
    ).set_index("model")

    print("\n[모델 비교: LogisticRegression vs RandomForest]")
    print(comparison_df.round(4))

    f1_gap = abs(logistic_result["f1_score"] - random_forest_result["f1_score"])
    if f1_gap < f1_tie_margin:
        final_result = logistic_result
        selection_reason = (
            f"F1-score 차이({f1_gap:.4f})가 기준({f1_tie_margin}) 미만이라, "
            "해석 가능성이 더 좋은 LogisticRegression을 최종 모델로 선택했다."
        )
    elif random_forest_result["f1_score"] > logistic_result["f1_score"]:
        final_result = random_forest_result
        selection_reason = (
            f"RandomForest의 F1-score({random_forest_result['f1_score']:.4f})가 "
            f"LogisticRegression({logistic_result['f1_score']:.4f})보다 유의미하게 높아 RandomForest를 선택했다."
        )
    else:
        final_result = logistic_result
        selection_reason = (
            f"LogisticRegression의 F1-score({logistic_result['f1_score']:.4f})가 "
            f"RandomForest({random_forest_result['f1_score']:.4f})보다 높거나 같아 LogisticRegression을 선택했다."
        )

    print(f"\n[최종 모델 선택] {final_result['model_name']}")
    print(selection_reason)

    return {
        "logistic": logistic_result,
        "random_forest": random_forest_result,
        "comparison_table": comparison_df,
        "final": final_result,
        "selection_reason": selection_reason,
    }


def analyze_group_performance(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, group_column: str = "sex"
) -> pd.DataFrame:
    """
    민감 변수(예: sex) 그룹별로 예측 성능과 양성(>50K) 예측 비율을 참고용으로 확인한다.
    공정성 문제를 판단하는 별도 라이브러리 없이, 그룹별 accuracy/F1/양성 예측 비율만 비교한다.
    """
    y_pred = pipeline.predict(X_test)
    rows = []
    for group_value, group_index in X_test.groupby(group_column).groups.items():
        group_y_true = y_test.loc[group_index]
        group_y_pred = pd.Series(y_pred, index=X_test.index).loc[group_index]
        rows.append(
            {
                "group": group_value,
                "n": len(group_index),
                "accuracy": accuracy_score(group_y_true, group_y_pred),
                "f1_score": f1_score(group_y_true, group_y_pred, pos_label=1, zero_division=0),
                "positive_pred_ratio": float(group_y_pred.mean()),
            }
        )
    group_performance_df = pd.DataFrame(rows).set_index("group").round(4)

    print(f"\n[참고 분석] {group_column} 그룹별 예측 성능 및 양성(>50K) 예측 비율")
    print(group_performance_df)
    print(
        "주의: 이 결과는 모델이 그룹별로 어떻게 동작하는지 참고하기 위한 것이며, "
        "성능 차이가 있다고 해서 그 자체로 모델이 차별적이라고 단정할 수는 없다."
    )

    return group_performance_df


def save_and_verify_model(
    pipeline: Pipeline, X_test: pd.DataFrame, model_path: Path, metadata: dict[str, Any] | None = None
) -> None:
    """
    학습된 전체 Pipeline(전처리+모델)을 저장하고, 다시 불러와 예측이 동일한지 확인한다.
    metadata를 주면 입력 feature 목록/제외 컬럼/타깃 매핑/모델 설정/저장 시각을 같은 폴더에 JSON으로 함께 남긴다.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    before_save_predictions = pipeline.predict(X_test.head(5))

    joblib.dump(pipeline, model_path)
    print(f"\nPipeline 저장 완료: {model_path}")

    if metadata is not None:
        metadata_path = model_path.with_suffix(".json")
        metadata_to_save = {**metadata, "saved_at": datetime.now().isoformat(timespec="seconds")}
        metadata_path.write_text(json.dumps(metadata_to_save, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"모델 메타데이터 저장 완료: {metadata_path}")

    loaded_pipeline = joblib.load(model_path)
    after_load_predictions = loaded_pipeline.predict(X_test.head(5))

    predictions_match = np.array_equal(before_save_predictions, after_load_predictions)
    print(f"저장 전 예측: {before_save_predictions}")
    print(f"재로딩 후 예측: {after_load_predictions}")
    print(f"저장 전/후 예측 일치 여부: {predictions_match}")

    if not predictions_match:
        raise ValueError("저장 전과 재로딩 후 예측 결과가 다릅니다. 모델 저장 과정을 확인하세요.")
