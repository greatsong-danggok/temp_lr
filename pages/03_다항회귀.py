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
    page_title="서울 기온 20차 다항회귀",
    layout="wide"
)

st.title("서울 연평균기온 20차 다항회귀 분석")

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

    col2.metric(
        "기록 시작",
        valid_dates.min().strftime("%Y-%m-%d")
    )

    col3.metric(
        "기록 마지막",
        valid_dates.max().strftime("%Y-%m-%d")
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
# 3. 연도 만들기
# ==================================================

df["연도"] = df["날짜"].dt.year


# ==================================================
# 4. 연도별 유효한 평균기온 일수
# ==================================================

st.header("3. 연도별 유효한 평균기온 일수")

# 전체 기록 기간의 모든 연도를 만듭니다.
# 데이터가 하나도 없는 연도도 0일로 남습니다.

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

valid_days_df = pd.DataFrame({
    "연도": valid_days.index,
    "유효한 평균기온 일수": valid_days.values
})

st.dataframe(
    valid_days_df,
    use_container_width=True
)


# ==================================================
# 5. 연평균기온 계산
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
# 6. 350일 이상인 연도만 분석
# ==================================================

st.header("4. 분석에 사용할 연도")

st.write(
    "유효한 평균기온이 350일 이상인 연도만 "
    "연평균기온 분석에 사용합니다."
)

analysis_df = annual_df[
    annual_df["유효일수"] >= 350
].copy()

excluded_df = valid_days_df[
    valid_days_df["유효한 평균기온 일수"] < 350
].copy()


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
        excluded_df,
        use_container_width=True
    )


# ==================================================
# 7. 학습용 / 평가용 데이터 분리
# ==================================================

st.header("5. 학습용 데이터와 평가용 데이터")

train_df = analysis_df[
    analysis_df["연도"] <= 2004
].copy()

test_df = analysis_df[
    analysis_df["연도"] >= 2005
].copy()


col1, col2 = st.columns(2)

with col1:

    st.subheader("학습용 데이터")

    if len(train_df) > 0:

        st.write(
            f"{train_df['연도'].min()}년 ~ "
            f"{train_df['연도'].max()}년"
        )

        st.write(
            f"사용 연도: {len(train_df)}개"
        )

with col2:

    st.subheader("평가용 데이터")

    if len(test_df) > 0:

        st.write(
            f"{test_df['연도'].min()}년 ~ "
            f"{test_df['연도'].max()}년"
        )

        st.write(
            f"사용 연도: {len(test_df)}개"
        )


# ==================================================
# 8. 데이터 준비
# ==================================================

X_train_year = train_df[["연도"]].values
X_test_year = test_df[["연도"]].values

y_train = train_df["연평균기온"].values
y_test = test_df["연평균기온"].values


# ==================================================
# 9. 연도 변환
# ==================================================
# 중요:
# 학습용 데이터만 이용하여 기준을 만들고
# 평가용 데이터와 2050년에도 같은 기준을 적용합니다.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_year
)

X_test_scaled = scaler.transform(
    X_test_year
)


# ==================================================
# 10. 20차 다항회귀
# ==================================================

st.header("6. 20차 다항회귀 모델")

poly = PolynomialFeatures(
    degree=20,
    include_bias=False
)

X_train_poly = poly.fit_transform(
    X_train_scaled
)

X_test_poly = poly.transform(
    X_test_scaled
)


model = LinearRegression()

model.fit(
    X_train_poly,
    y_train
)


# ==================================================
# 11. 학습용 / 평가용 예측
# ==================================================

train_pred = model.predict(
    X_train_poly
)

test_pred = model.predict(
    X_test_poly
)


# ==================================================
# 12. R² 계산
# ==================================================

train_r2 = r2_score(
    y_train,
    train_pred
)

test_r2 = r2_score(
    y_test,
    test_pred
)

r2_difference = train_r2 - test_r2


# ==================================================
# 13. 2050년 예측
# ==================================================

year_2050 = np.array([
    [2050]
])

# 학습할 때 사용한 scaler를 그대로 사용
year_2050_scaled = scaler.transform(
    year_2050
)

# 학습할 때 만든 20차 변환을 그대로 사용
year_2050_poly = poly.transform(
    year_2050_scaled
)

prediction_2050 = model.predict(
    year_2050_poly
)[0]


# ==================================================
# 14. 결과 표
# ==================================================

st.header("7. 20차 다항회귀 결과")

result_df = pd.DataFrame({
    "모델": ["20차 다항회귀"],
    "학습용 R²": [train_r2],
    "평가용 R²": [test_r2],
    "R² 차이": [r2_difference],
    "2050년 예측값(℃)": [prediction_2050]
})

st.dataframe(
    result_df.style.format({
        "학습용 R²": "{:.4f}",
        "평가용 R²": "{:.4f}",
        "R² 차이": "{:.4f}",
        "2050년 예측값(℃)": "{:.2f}"
    }),
    use_container_width=True
)


# ==================================================
# 15. 과적합 확인
# ==================================================

st.header("8. 과적합 확인")

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
    "학습용 R²은 모델을 만드는 데 사용한 데이터에 "
    "얼마나 잘 맞는지를 나타냅니다. "
    "평가용 R²은 모델을 만들 때 사용하지 않은 "
    "2005년 이후 데이터에 얼마나 잘 맞는지를 나타냅니다. "
    "두 값의 차이가 크다면 과적합 가능성을 생각할 수 있습니다."
)


# ==================================================
# 16. 첫 번째 그래프
#     학습용 데이터 + 20차 함수
# ==================================================

st.header("9. 학습용 데이터에 맞춘 20차 함수")

st.write(
    "첫 번째 그래프는 2004년까지의 학습용 데이터와 "
    "그 데이터에 맞춰 만든 20차 함수를 보여줍니다."
)

fig_train = go.Figure()


# --------------------------------------------------
# 학습용 실제값
# --------------------------------------------------

fig_train.add_trace(
    go.Scatter(
        x=train_df["연도"],
        y=train_df["연평균기온"],
        mode="markers",
        name="학습용 실제값",
        marker=dict(size=7),
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 20차 함수
# 2004년까지만 표시
# --------------------------------------------------

train_graph_years = np.arange(
    int(analysis_df["연도"].min()),
    2004 + 1
).reshape(-1, 1)

train_graph_scaled = scaler.transform(
    train_graph_years
)

train_graph_poly = poly.transform(
    train_graph_scaled
)

train_graph_pred = model.predict(
    train_graph_poly
)


fig_train.add_trace(
    go.Scatter(
        x=train_graph_years.flatten(),
        y=train_graph_pred,
        mode="lines",
        name="20차 다항회귀",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}<br>"
            "20차 예측값: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig_train.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600,
    hovermode="x unified",
    legend_title="표시 항목"
)

st.plotly_chart(
    fig_train,
    use_container_width=True
)


# ==================================================
# 17. 두 번째 그래프
#     학습 + 평가 데이터 + 2050년까지 함수 연장
# ==================================================

st.header("10. 2050년까지 연장한 20차 함수")

st.write(
    "두 번째 그래프에서는 학습용 데이터와 평가용 데이터를 "
    "함께 표시하고, 학습용 데이터로 만든 20차 함수를 "
    "2050년까지 연장합니다."
)

fig_2050 = go.Figure()


# --------------------------------------------------
# 학습용 실제값
# --------------------------------------------------

fig_2050.add_trace(
    go.Scatter(
        x=train_df["연도"],
        y=train_df["연평균기온"],
        mode="markers",
        name="학습용 실제값",
        marker=dict(size=7),
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

fig_2050.add_trace(
    go.Scatter(
        x=test_df["연도"],
        y=test_df["연평균기온"],
        mode="markers",
        name="평가용 실제값",
        marker=dict(size=7),
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 20차 함수
# 실제 마지막 연도부터 2050년까지 연장
# --------------------------------------------------

graph_years_2050 = np.arange(
    int(analysis_df["연도"].min()),
    2050 + 1
).reshape(-1, 1)

graph_years_2050_scaled = scaler.transform(
    graph_years_2050
)

graph_years_2050_poly = poly.transform(
    graph_years_2050_scaled
)

graph_predictions_2050 = model.predict(
    graph_years_2050_poly
)


fig_2050.add_trace(
    go.Scatter(
        x=graph_years_2050.flatten(),
        y=graph_predictions_2050,
        mode="lines",
        name="20차 다항회귀",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}<br>"
            "20차 예측값: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# --------------------------------------------------
# 학습 → 평가 경계
# --------------------------------------------------

fig_2050.add_vline(
    x=2004.5,
    line_dash="dash",
    annotation_text="학습 → 평가",
    annotation_position="top"
)


# --------------------------------------------------
# 2050년 표시
# --------------------------------------------------

fig_2050.add_vline(
    x=2050,
    line_dash="dot",
    annotation_text="2050년",
    annotation_position="top"
)


fig_2050.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600,
    hovermode="x unified",
    legend_title="표시 항목"
)

st.plotly_chart(
    fig_2050,
    use_container_width=True
)


# ==================================================
# 18. 2050년 예측
# ==================================================

st.header("11. 2050년 예측")

st.metric(
    "20차 다항회귀의 2050년 예측",
    f"{prediction_2050:.2f} ℃"
)

st.warning(
    "2050년은 실제 관측값이 없는 미래이므로 "
    "이 값은 실제 기온이 아닙니다. "
    "2004년까지의 학습 데이터로 만든 20차 함수를 "
    "2050년까지 연장하여 계산한 외삽값입니다."
)


# ==================================================
# 19. 최종 해석
# ==================================================

st.header("12. 결과 해석")

st.write(
    f"""
### ① 학습 데이터

20차 다항회귀의 학습용 R²은 **{train_r2:.4f}**입니다.

이는 2004년까지의 학습 데이터에
20차 함수가 얼마나 잘 맞는지를 나타냅니다.

첫 번째 그래프에서 20차 함수가 학습용 데이터의
변화를 얼마나 세밀하게 따라가는지 확인할 수 있습니다.


### ② 평가 데이터

2005년 이후 평가용 데이터의 R²은
**{test_r2:.4f}**입니다.

평가용 데이터는 모델을 만들 때 사용하지 않았기 때문에
새로운 데이터에 대한 성능을 확인하는 데 사용할 수 있습니다.


### ③ 과적합

학습용 R²은 매우 높은데 평가용 R²이 상대적으로 낮다면,
20차 함수가 학습 데이터의 특징을 지나치게 따라간
**과적합**이 발생했을 가능성이 있습니다.

따라서 첫 번째 그래프와 두 번째 그래프를 비교하면
복잡한 함수가 학습 데이터에는 잘 맞지만
새로운 데이터에도 잘 맞는지는 별개의 문제라는 것을
확인할 수 있습니다.


### ④ 2050년

20차 다항회귀의 2050년 예측값은
**{prediction_2050:.2f}℃**입니다.

하지만 이 값은 실제 2050년의 기온을 의미하지 않습니다.

과거 학습 데이터에서 만들어진 20차 함수를
미래로 연장해서 계산한 **외삽값**입니다.

따라서 2050년 예측값은
실제 미래 기온을 확정적으로 보여주는 값이 아니라,
현재 만든 모델을 미래에 적용했을 때 나오는
하나의 계산 결과로 해석해야 합니다.
"""
)
