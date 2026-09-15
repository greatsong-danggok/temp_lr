import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ==================================================
# 기본 설정
# ==================================================

st.set_page_config(
    page_title="서울 기온 30차 다항회귀",
    layout="wide"
)

st.title("서울 연평균기온 30차 다항회귀 분석")

URL = "https://raw.githubusercontent.com/greatsong/modudata/refs/heads/main/data/seoul_daily.csv"


# ==================================================
# 데이터 불러오기
# ==================================================

@st.cache_data
def load_data():
    df = pd.read_csv(URL)

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    return df


df = load_data()


# ==================================================
# 1. 데이터 기본 정보
# ==================================================

st.header("1. 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 행 수",
    f"{len(df):,}개"
)

valid_dates = df["날짜"].dropna()

if len(valid_dates) > 0:

    start_date = valid_dates.min()
    end_date = valid_dates.max()

    col2.metric(
        "기록 시작",
        start_date.strftime("%Y-%m-%d")
    )

    col3.metric(
        "기록 마지막",
        end_date.strftime("%Y-%m-%d")
    )

col4.metric(
    "평균기온 결측값",
    f"{df['평균기온'].isna().sum():,}개"
)

st.write("**열 이름**")
st.write(list(df.columns))


# ==================================================
# 2. 중복 날짜 확인
# ==================================================

st.header("2. 중복 날짜 확인")

duplicate_dates = df[
    df["날짜"].duplicated(keep=False)
].sort_values("날짜")

if len(duplicate_dates) > 0:

    st.warning(
        f"중복된 날짜가 "
        f"{duplicate_dates['날짜'].nunique()}개 있습니다."
    )

    st.dataframe(
        duplicate_dates,
        use_container_width=True
    )

else:

    st.success("중복된 날짜가 없습니다.")


# ==================================================
# 3. 연도별 유효한 평균기온 일수
# ==================================================

df["연도"] = df["날짜"].dt.year

all_years = range(
    int(df["연도"].min()),
    int(df["연도"].max()) + 1
)

valid_days = (
    df.dropna(subset=["평균기온"])
      .groupby("연도")["평균기온"]
      .count()
      .reindex(all_years, fill_value=0)
      .astype(int)
)


st.header("3. 연도별 유효한 평균기온 일수")

valid_days_df = pd.DataFrame({
    "연도": valid_days.index,
    "유효한 평균기온 일수": valid_days.values
})

st.dataframe(
    valid_days_df,
    use_container_width=True
)


# ==================================================
# 4. 연평균기온 계산
# ==================================================

annual_temp = (
    df.dropna(subset=["평균기온"])
      .groupby("연도")["평균기온"]
      .mean()
)

annual_df = pd.DataFrame({
    "연도": annual_temp.index,
    "연평균기온": annual_temp.values,
    "유효일수": valid_days.loc[annual_temp.index].values
})


# ==================================================
# 5. 350일 이상인 연도만 분석
# ==================================================

analysis_df = annual_df[
    annual_df["유효일수"] >= 350
].copy()

excluded_df = annual_df[
    annual_df["유효일수"] < 350
].copy()


st.header("4. 분석에 사용할 연도")

st.write(
    "연평균기온을 계산할 때 "
    "유효한 평균기온이 350일 이상인 연도만 사용합니다."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "분석에 사용한 연도",
        f"{len(analysis_df)}개"
    )

with col2:
    st.metric(
        "제외된 연도",
        f"{len(excluded_df)}개"
    )


if len(excluded_df) > 0:

    st.subheader("분석에서 제외된 연도")

    st.dataframe(
        excluded_df[
            ["연도", "연평균기온", "유효일수"]
        ],
        use_container_width=True
    )


# ==================================================
# 6. 학습용 / 평가용 데이터 분리
# ==================================================

train_df = analysis_df[
    analysis_df["연도"] <= 2004
].copy()

test_df = analysis_df[
    analysis_df["연도"] >= 2005
].copy()


st.header("5. 학습용 데이터와 평가용 데이터")

col1, col2 = st.columns(2)

with col1:

    st.subheader("학습용 데이터")

    st.write(
        f"{train_df['연도'].min()}년 ~ "
        f"{train_df['연도'].max()}년"
    )

    st.write(
        f"사용 연도: {len(train_df)}개"
    )

with col2:

    st.subheader("평가용 데이터")

    st.write(
        f"{test_df['연도'].min()}년 ~ "
        f"{test_df['연도'].max()}년"
    )

    st.write(
        f"사용 연도: {len(test_df)}개"
    )


# ==================================================
# 7. 학습용 / 평가용 데이터 준비
# ==================================================

X_train_year = train_df[["연도"]].values
X_test_year = test_df[["연도"]].values

y_train = train_df["연평균기온"].values
y_test = test_df["연평균기온"].values


# ==================================================
# 8. 연도를 학습용 데이터 기준으로 변환
# ==================================================

scaler = StandardScaler()

# 학습용 데이터로 기준을 만들기
X_train_scaled = scaler.fit_transform(
    X_train_year
)

# 평가용 데이터에는 같은 기준 적용
X_test_scaled = scaler.transform(
    X_test_year
)


# ==================================================
# 9. 30차 다항회귀
# ==================================================

st.header("6. 30차 다항회귀")

st.write(
    "2004년까지의 학습용 데이터만 이용하여 "
    "30차 다항회귀 모델을 학습합니다."
)

poly = PolynomialFeatures(
    degree=30,
    include_bias=False
)


# 학습용 데이터로 30차 기준 만들기
X_train_poly = poly.fit_transform(
    X_train_scaled
)

# 평가용 데이터에는 같은 기준 적용
X_test_poly = poly.transform(
    X_test_scaled
)


model = LinearRegression()

model.fit(
    X_train_poly,
    y_train
)


# ==================================================
# 10. 학습용 / 평가용 예측
# ==================================================

train_pred = model.predict(
    X_train_poly
)

test_pred = model.predict(
    X_test_poly
)


# ==================================================
# 11. R² 계산
# ==================================================

train_r2 = r2_score(
    y_train,
    train_pred
)

test_r2 = r2_score(
    y_test,
    test_pred
)


# ==================================================
# 12. 2050년 예측
# ==================================================

year_2050 = np.array([
    [2050]
])

year_2050_scaled = scaler.transform(
    year_2050
)

year_2050_poly = poly.transform(
    year_2050_scaled
)

prediction_2050 = model.predict(
    year_2050_poly
)[0]


# ==================================================
# 13. 결과 표
# ==================================================

st.header("7. 30차 다항회귀 결과")

result_df = pd.DataFrame({
    "모델": ["30차 다항회귀"],
    "학습용 R²": [train_r2],
    "평가용 R²": [test_r2],
    "2050년 예측값(℃)": [prediction_2050]
})


st.dataframe(
    result_df.style.format({
        "학습용 R²": "{:.4f}",
        "평가용 R²": "{:.4f}",
        "2050년 예측값(℃)": "{:.2f}"
    }),
    use_container_width=True
)


# ==================================================
# 14. 과적합 확인
# ==================================================

st.header("8. 과적합 확인")

r2_difference = train_r2 - test_r2

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "학습용 R²",
        f"{train_r2:.4f}"
    )

with col2:

    st.metric(
        "평가용 R²",
        f"{test_r2:.4f}"
    )

with col3:

    st.metric(
        "R² 차이",
        f"{r2_difference:.4f}"
    )


st.info(
    "학습용 R²과 평가용 R²의 차이가 크다면 "
    "모델이 학습 데이터의 특징을 지나치게 따라간 "
    "과적합 가능성을 생각할 수 있습니다."
)


# ==================================================
# 15. 그래프용 30차 곡선
# ==================================================
#
# 중요:
# 2050년까지 곡선을 그리지 않습니다.
#
# 30차 다항식은 학습 범위를 벗어나면
# 값이 매우 크게 튈 수 있기 때문입니다.
#
# 따라서 그래프는 실제 데이터가 존재하는
# 마지막 연도까지만 표시합니다.
#


graph_years = np.arange(
    analysis_df["연도"].min(),
    analysis_df["연도"].max() + 1
).reshape(-1, 1)


graph_years_scaled = scaler.transform(
    graph_years
)

graph_years_poly = poly.transform(
    graph_years_scaled
)

graph_predictions = model.predict(
    graph_years_poly
)


# ==================================================
# 16. 그래프
# ==================================================

st.header("9. 학습용·평가용 데이터와 30차 곡선")

fig = go.Figure()


# --------------------------------------------------
# 학습용 실제값
# --------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=train_df["연도"],
        y=train_df["연평균기온"],
        mode="markers",
        name="학습용 실제값",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 평가용 실제값
# --------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=test_df["연도"],
        y=test_df["연평균기온"],
        mode="markers",
        name="평가용 실제값",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 30차 다항회귀 곡선
# --------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=graph_years.flatten(),
        y=graph_predictions,
        mode="lines",
        name="30차 다항회귀",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "30차 예측값: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 학습 / 평가 구간 경계
# --------------------------------------------------

fig.add_vline(
    x=2004.5,
    line_dash="dash",
    annotation_text="학습 → 평가",
    annotation_position="top"
)


# --------------------------------------------------
# 그래프 설정
# --------------------------------------------------

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    height=650,
    legend_title="표시 항목"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# 17. 2050년 예측값 별도 표시
# ==================================================

st.header("10. 2050년 예측")

st.metric(
    "30차 다항회귀의 2050년 예측 연평균기온",
    f"{prediction_2050:.2f} ℃"
)

st.warning(
    "2050년은 현재 데이터의 마지막 연도 이후이므로 "
    "그래프에는 2050년까지 곡선을 연장하지 않았습니다. "
    "2050년 값은 학습된 30차 다항식을 이용한 외삽 결과입니다."
)


# ==================================================
# 18. 해석
# ==================================================

st.header("11. 결과 해석")

st.write(
    f"""
### 학습용 데이터

학습용 데이터의 R²은 **{train_r2:.4f}**입니다.

### 평가용 데이터

학습에 사용하지 않은 평가용 데이터의 R²은
**{test_r2:.4f}**입니다.

### 두 R²의 차이

학습용 R²과 평가용 R²의 차이는
**{r2_difference:.4f}**입니다.

학습용 R²은 매우 높은데 평가용 R²이 상대적으로 낮다면,
모델이 과거의 학습 데이터를 지나치게 따라간
**과적합** 가능성을 생각할 수 있습니다.

### 2050년

30차 다항회귀가 계산한 2050년 예측값은
**{prediction_2050:.2f}℃**입니다.

이 값은 실제 2050년의 기온을 의미하는 것이 아니라,
과거 자료를 이용해 만든 30차 함수를
2050년까지 바깥쪽으로 연장해서 얻은 **외삽값**입니다.
"""
)
