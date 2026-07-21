"""실행 결과를 바탕으로 report.md를 자동 생성하는 모듈."""

from pathlib import Path
from typing import Any

ETHICS_NOTE = """\
- 이 모델은 수업용 데이터 분석 실습 결과물이며, 교육 목적으로만 사용한다.
- 실제 채용, 대출, 보험, 복지 대상 선정 등 사람에게 실질적 영향을 주는 의사결정에 그대로 사용하면 안 된다.
- Adult 데이터는 과거 시점의 인구조사 데이터이므로, 그 시절 사회적 편향이 모델에 그대로 재현될 수 있다.
- Accuracy/F1 같은 지표가 높다고 해서 모델이 공정하다는 뜻은 아니다.
- race, sex, native-country 같은 민감 속성을 feature에서 빼더라도 education, occupation 등
  다른 변수가 민감 속성의 대리 변수(proxy) 역할을 해서 간접적으로 편향이 남을 수 있다.
- 이 모델은 인과관계를 설명하거나 개인의 능력·가치를 판단하는 모델이 아니라,
  주어진 데이터에서 나타난 통계적 패턴을 학습한 결과일 뿐이다.
"""


def generate_markdown_report(context: dict[str, Any], report_path: Path) -> Path:
    """context에 담긴 실제 실행 결과 값을 이용해 report.md를 작성한다(하드코딩된 숫자 없음)."""
    data_info = context["data"]
    income_info = context["income"]
    t_test = context["t_test"]
    model_info = context["model"]
    output_files = context["output_files"]

    missing_lines = "\n".join(
        f"  - {col}: {count}건" for col, count in data_info["key_missing_counts"].items()
    )
    chi_square_lines = "\n".join(
        f"  - {test['row_column']} x {test['col_column']}: "
        f"chi2={test['chi2']:.2f}, p-value={test['p_value']:.4f}, "
        f"{'유의함' if test['is_significant'] else '유의하지 않음'}"
        for test in context.get("chi_square_tests", [])
    )

    t_test_conclusion = (
        "귀무가설을 기각하며, 두 income 그룹의 평균 주당 근무시간에 통계적으로 유의한 차이가 있다."
        if t_test["is_significant"]
        else "귀무가설을 기각할 충분한 근거가 없으며, 통계적으로 유의한 차이를 확인하지 못했다."
    )

    numeric_summary_markdown = context["numeric_summary"].to_markdown()
    correlation_pairs_markdown = context["correlation_top_pairs"].to_markdown(index=False)

    report_content = f"""# Adult Census Income 분석 보고서

## 1. 프로젝트 개요
- 분석 목적: UCI Adult Census Income 데이터를 이용해 개인의 income이 <=50K인지 >50K인지 예측하는
  End-to-End 데이터 분석/모델링 파이프라인을 구현하고, 그 과정에서 데이터 품질과 통계적 특성을 확인한다.
- 사용 데이터: UCI Adult Census Income 데이터셋 (adult.data)
- 분석 대상: 인구조사 응답자 개인 단위 레코드, 총 {data_info['raw_row_count']}행

## 2. 데이터 품질
- 원본 데이터: {data_info['raw_row_count']}행 x {data_info['raw_col_count']}열
- 정제 후 데이터: {data_info['cleaned_row_count']}행 x {data_info['cleaned_col_count']}열
- 완전 중복 행: {data_info['duplicate_count']}건 (정제 과정에서 제거)
- 주요 결측 컬럼별 결측치 수:
{missing_lines}
- 처리 방법: {data_info['missing_strategy_summary']}

## 3. EDA 주요 결과

### income 분포
- <=50K: {income_info['counts'].get('<=50K', 0)}건 ({income_info['ratios'].get('<=50K', 0)}%)
- >50K : {income_info['counts'].get('>50K', 0)}건 ({income_info['ratios'].get('>50K', 0)}%)
- 클래스 비율(다수/소수): {income_info['imbalance_ratio']:.2f}배로 불균형이 존재하므로,
  모델 평가 시 Accuracy 외에 Precision/Recall/F1-score를 함께 봐야 한다.

### 주요 수치형 변수 기술통계
{numeric_summary_markdown}

- capital-gain과 capital-loss는 0값 비율이 매우 높고 오른쪽 꼬리가 긴 분포라서,
  큰 값을 단순 오류로 보고 삭제하지 않았다(시각화에서는 log1p 변환으로 확인).
- fnlwgt는 인구조사 가중치(census weight)이므로 값이 크더라도 측정 오류가 아니다.

## 4. 통계 분석

### 상관관계 (숫자형 컬럼, income 제외)
{correlation_pairs_markdown}

- 상관관계는 인과관계를 의미하지 않는다.
- education과 education-num은 같은 학력 정보를 다른 형태로 표현하므로 서로 상관이 높을 수 있고,
  모델 feature로 함께 쓸 때 정보 중복에 유의해야 한다.

### t-test: income 그룹별 hours-per-week
- H0: 두 income 그룹의 평균 hours-per-week에는 차이가 없다.
- H1: 두 income 그룹의 평균 hours-per-week에는 차이가 있다.
- >50K 그룹: n={t_test['group_high_n']}, 평균={t_test['group_high_mean']:.2f}
- <=50K 그룹: n={t_test['group_low_n']}, 평균={t_test['group_low_mean']:.2f}
- t 통계량: {t_test['t_statistic']:.4f}
- p-value: {t_test['p_value']:.6f}
- Cohen's d: {t_test['cohens_d']:.3f}
- 결론: {t_test_conclusion}
- 주의: 표본 수가 크면 작은 차이도 매우 작은 p-value로 나타날 수 있어, 통계적 유의성과
  실질적 효과 크기(Cohen's d)를 함께 고려해야 한다.

### 카이제곱 독립성 검정 (선택적 추가 분석)
{chi_square_lines if chi_square_lines else '  - 수행하지 않음'}

## 5. 모델링
- 사용한 feature: 수치형 {model_info['numeric_features']}, 범주형 {model_info['categorical_features']}
- target: income (feature에서는 제외하여 데이터 누수를 방지)
- 전처리: SimpleImputer + StandardScaler(수치형), SimpleImputer + OneHotEncoder(범주형)를
  ColumnTransformer로 연결하고, sklearn Pipeline 내부에서 train 데이터로만 학습
- 사용 모델: LogisticRegression(class_weight="balanced")
- train/test 분할: test_size=0.2, stratify=income, random_state=42
- Accuracy : {model_info['accuracy']:.4f}
- Precision: {model_info['precision']:.4f}
- Recall   : {model_info['recall']:.4f}
- F1-score : {model_info['f1_score']:.4f}
- ROC-AUC  : {model_info['roc_auc']:.4f}

## 6. 주요 인사이트
- 아래 내용은 이번 실행에서 나온 데이터 기반 결과이며, 인과관계로 확대 해석하지 않는다.
- income 클래스는 <=50K가 다수를 차지하는 불균형 데이터이므로, Accuracy만으로 모델을 평가하면
  소수 클래스(>50K)를 잘 못 맞혀도 점수가 높게 나올 수 있다.
- hours-per-week는 income 그룹 간 통계적으로 유의한 차이를 보였으나, 이는 상관관계 수준의 관찰이며
  근무시간이 소득을 결정한다는 인과관계로 해석해서는 안 된다.

### 윤리 및 편향 주의
{ETHICS_NOTE}

## 7. 생성 파일
- 정제 데이터 CSV: {output_files['cleaned_csv']}
- 정적 차트 PNG: {output_files['income_distribution_png']}, {output_files['numeric_eda_png']}, \
{output_files['categorical_eda_png']}, {output_files['correlation_heatmap_png']}, \
{output_files['confusion_matrix_png']}
- Plotly 인터랙티브 차트 HTML: {output_files['plotly_html']}
- 학습된 모델 Pipeline: {output_files['model_pkl']}
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nreport.md 생성 완료: {report_path}")

    return report_path
