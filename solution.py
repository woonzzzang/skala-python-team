import pandas as pd
import time
import polars as pl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
from pathlib import Path
from scipy import stats
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ["age","workclass","fnlwgt","education","education-num",
"marital-status","occupation","relationship","race","sex",
"capital-gain","capital-loss","hours-per-week","native-country","income"]
df = pd.read_csv(url, header=None, names=cols, na_values=" ?",
skipinitialspace=True)
print(df.shape) # (32561, 15)

# ── Pandas로 로딩 ──
df_pd = pd.read_csv(url, header=None, names=cols, na_values=" ?",
skipinitialspace=True)

print("=== Pandas EDA ===")
print(df_pd.info())
print(df_pd.describe())
print(df_pd.nunique())
print("실제 결측(NaN):\n", df_pd.isna().sum())
print("'?' 로 표시된 숨은 결측치:\n", (df_pd == '?').sum())
print("중복 행 개수:", df_pd.duplicated().sum())


# ── Polars로 동일하게 로딩 ──
schema = {
    'age': pl.Int64,
    'workclass': pl.Utf8,
    'fnlwgt': pl.Int64,
    'education': pl.Utf8,
    'education-num': pl.Int64,
    'marital-status': pl.Utf8,
    'occupation': pl.Utf8,
    'relationship': pl.Utf8,
    'race': pl.Utf8,
    'sex': pl.Utf8,
    'capital-gain': pl.Int64,
    'capital-loss': pl.Int64,
    'hours-per-week': pl.Int64,
    'native-country': pl.Utf8,
    'income': pl.Utf8,
}

df_pl = pl.read_csv(
    url,
    has_header=False,
    new_columns=cols,
    null_values=" ?",
    schema=schema,   # infer_schema_length 대신 이걸로 명확하게 지정
)

print("\n=== Polars EDA ===")
print(df_pl.schema)
print(df_pl.describe())
print("'?' 로 표시된 숨은 결측치:")
for col in df_pl.columns:
    if df_pl[col].dtype == pl.Utf8:
        cnt = (df_pl[col] == '?').sum()
        if cnt > 0:
            print(f'  {col}: {cnt}')
dup_count_pl = df_pl.height - df_pl.unique().height
print("중복 행 개수:", dup_count_pl)


# ── 결측치 처리: '?' → 실제 NaN으로 변환 후 처리 ──
df_pd_clean = df_pd.replace('?', pd.NA)
print("\n'?' -> NaN 변환 후 결측치:\n", df_pd_clean.isna().sum())

# 결측 있는 행 제거 (또는 최빈값으로 채우기 등 선택)
df_pd_clean = df_pd_clean.dropna()
print(f"\n처리 전 {len(df_pd)}행 -> 처리 후 {len(df_pd_clean)}행")

# Polars 결측치 처리 (null_values=" ?"로 이미 null 처리된 상태)
df_pl_clean = df_pl.drop_nulls()
print(f"\nPolars 처리 전 {df_pl.height}행 -> 처리 후 {df_pl_clean.height}행")


# ── 중복 제거 ──
before = len(df_pd_clean)
df_pd_clean = df_pd_clean.drop_duplicates()
after = len(df_pd_clean)
print(f"중복 제거 전 {before}행 -> 제거 후 {after}행")

before_pl = df_pl_clean.height
df_pl_clean = df_pl_clean.unique()
after_pl = df_pl_clean.height
print(f"Polars 중복 제거 전 {before_pl}행 -> 제거 후 {after_pl}행")

matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

OUT_DIR = Path(__file__).parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

# ── Seaborn 정적 차트: age 분포 ──
plt.figure(figsize=(8, 5))
sns.histplot(df_pd_clean['age'], kde=True, bins=30)
plt.title('나이(age) 분포')
plt.xlabel('나이')
plt.ylabel('빈도')
plt.tight_layout()
plt.savefig(OUT_DIR / 'age_distribution.png')
plt.close()
print(f'Seaborn 차트 저장 완료: {OUT_DIR / "age_distribution.png"}')


# ── Plotly 인터랙티브 차트: education별 income 비율 비교 ──
edu_income = (
    df_pd_clean.groupby(['education', 'income'])
    .size()
    .reset_index(name='count')
)

fig = px.bar(
    edu_income,
    x='education', y='count', color='income',
    barmode='group',
    title='학력별 소득 구간 비교'
)
html_path = OUT_DIR / 'education_income.html'
fig.write_html(html_path)
print(f'Plotly 차트 저장 완료: {html_path}')

# ── 1) 기술통계 산출 (수치형 컬럼 전체) ──
num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

print("\n=== 기술통계 (수치형 컬럼 전체) ===")
desc = df_pd_clean[num_cols].describe()
print(desc)


# ── 2) 변수 간 상관계수 계산 (동일한 num_cols 재사용) ──
print("\n=== 수치형 변수 간 상관계수 ===")
corr_matrix = df_pd_clean[num_cols].corr()
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('수치형 변수 간 상관관계')
plt.tight_layout()
plt.savefig(OUT_DIR / 'correlation_heatmap.png')
plt.close()
print(f'상관관계 히트맵 저장 완료: {OUT_DIR / "correlation_heatmap.png"}')


# ── 3) t-test: income(<=50K vs >50K)에 따른 age 차이 검정 ──
income_low = df_pd_clean[df_pd_clean['income'] == '<=50K']['age']
income_high = df_pd_clean[df_pd_clean['income'] == '>50K']['age']

t_stat, p_value = stats.ttest_ind(income_low, income_high, equal_var=False)

print(f"\n=== t-검정: 소득 구간(<=50K vs >50K)에 따른 나이 차이 ===")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4e}")

if p_value < 0.05:
    print("-> 두 소득 구간 간 평균 나이 차이는 통계적으로 유의미합니다 (p < 0.05)")
else:
    print("-> 두 소득 구간 간 평균 나이 차이는 통계적으로 유의미하지 않습니다 (p >= 0.05)")

print(f"참고: <=50K 평균 나이 = {income_low.mean():.1f}세, >50K 평균 나이 = {income_high.mean():.1f}세")

# ── 데이터 준비 ──
df_ml = df_pd_clean.copy()

cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country']

X = df_ml[num_cols + cat_cols]
y = (df_ml['income'] == '>50K').astype(int)   # 문자열 -> 0/1로 변환

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 전처리 구성 ──
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
    ]), num_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('oh', OneHotEncoder(handle_unknown='ignore')),
    ]), cat_cols),
])

# ── 전처리 + 모델을 하나의 Pipeline으로 ──
pipe = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
])

# ── 학습 ──
pipe.fit(X_tr, y_tr)

# ── 평가 지표 출력 ──
y_pred = pipe.predict(X_te)
y_proba = pipe.predict_proba(X_te)[:, 1]

acc = accuracy_score(y_te, y_pred)
auc = roc_auc_score(y_te, y_proba)

print(f"\n=== 모델 평가 ===")
print(f"정확도(Accuracy): {acc:.3f}")
print(f"ROC-AUC: {auc:.3f}")
print("\n분류 리포트:")
print(classification_report(y_te, y_pred, target_names=['<=50K', '>50K']))

# ── joblib으로 모델 저장 ──
model_path = OUT_DIR / 'income_model.joblib'
joblib.dump(pipe, model_path)
print(f'모델 저장 완료: {model_path}')

# ── 재로딩 확인 ──
reloaded_pipe = joblib.load(model_path)
reloaded_acc = accuracy_score(y_te, reloaded_pipe.predict(X_te))
print(f'재로딩 후 정확도 일치 확인: {reloaded_acc:.3f}')