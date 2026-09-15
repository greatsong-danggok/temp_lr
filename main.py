```python
import streamlit as st
import pandas as pd

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="서울 연도별 평균기온 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연도별 평균기온 분석")

st.write(
    "서울 일별 기온 자료를 이용하여 연도별 평균기온을 계산합니다. "
    "평균기온이 있는 날짜가 한 해에 350일 이상인 연도만 최종 분석에 사용합니다."
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "refs/heads/main/data/seoul_daily.csv"
)

try:
    df = pd.read_csv(URL)
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# --------------------------------------------------
# 날짜 및 숫자형 데이터 정리
# --------------------------------------------------
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

# --------------------------------------------------
# 1. 기본 데이터 정보
# --------------------------------------------------
st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("열 이름")
    st.write(list(df.columns))

with col2:
    st.subheader("기록 기간")
    start_date = df["날짜"].min()
    end_date = df["날짜"].max()

    st.write(
        f"{start_date.strftime('%Y-%m-%d')} ~ "
        f"{end_date.strftime('%Y-%m-%d')}"
    )

with col3:
    st.subheader("평균기온 결측 수")
    missing_temp = df["평균기온"].isna().sum()
    st.metric("결측값", f"{missing_temp:,}개")

# --------------------------------------------------
# 2. 날짜 중복 확인
# --------------------------------------------------
st.header("2. 날짜 중복 여부 확인")

duplicate_mask = df["날짜"].duplicated(keep=False)
duplicate_dates = df.loc[duplicate_mask, "날짜"].drop_duplicates().sort_values()

if len(duplicate_dates) == 0:
    st.success("같은 날짜가 중복된 기록은 없습니다.")
else:
    st.warning(
        f"중복된 날짜가 {len(duplicate_dates):,}개 발견되었습니다."
    )

    duplicate_table = pd.DataFrame({
        "중복 날짜": duplicate_dates.dt.strftime("%Y-%m-%d")
    })

    st.dataframe(
        duplicate_table,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# 3. 연도 추출
# --------------------------------------------------
df["연도"] = df["날짜"].dt.year

# --------------------------------------------------
# 4. 연도별 평균기온 유효 날짜 수 계산
# --------------------------------------------------
yearly_valid_days = (
    df.dropna(subset=["평균기온"])
      .groupby("연도")
      .size()
      .rename("유효 날짜 수")
      .reset_index()
)

# --------------------------------------------------
# 5. 350일 미만 연도 제외
# --------------------------------------------------
excluded_years = yearly_valid_days[
    yearly_valid_days["유효 날짜 수"] < 350
].copy()

valid_years = yearly_valid_days[
    yearly_valid_days["유효 날짜 수"] >= 350
].copy()

# --------------------------------------------------
# 6. 제외된 연도 표시
# --------------------------------------------------
st.header("3. 분석에서 제외된 연도")

if len(excluded_years) == 0:
    st.success("유효 날짜가 350일 미만인 연도가 없습니다.")
else:
    st.write(
        "평균기온이 있는 날짜가 350일 미만인 연도는 "
        "연도별 평균기온 계산에서 제외했습니다."
    )

    excluded_display = excluded_years.copy()
    excluded_display["연도"] = excluded_display["연도"].astype(int)

    st.dataframe(
        excluded_display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"총 {len(excluded_years):,}개 연도가 제외되었습니다."
    )

# --------------------------------------------------
# 7. 분석에 사용되는 연도와 유효 날짜 수
# --------------------------------------------------
st.header("4. 최종 분석 대상 연도")

valid_display = valid_years.copy()
valid_display["연도"] = valid_display["연도"].astype(int)

st.dataframe(
    valid_display,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"총 {len(valid_years):,}개 연도가 최종 분석에 사용됩니다."
)

# --------------------------------------------------
# 8. 연도별 평균기온 계산
# --------------------------------------------------
filtered_df = df[
    df["연도"].isin(valid_years["연도"])
].copy()

yearly_mean = (
    filtered_df
    .dropna(subset=["평균기온"])
    .groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)

yearly_mean = yearly_mean.merge(
    valid_years,
    on="연도",
    how="left"
)

yearly_mean = yearly_mean[
    ["연도", "유효 날짜 수", "평균기온"]
].sort_values("연도")

# 소수점 정리
yearly_mean["평균기온"] = yearly_mean["평균기온"].round(2)

# --------------------------------------------------
# 9. 최종 결과
# --------------------------------------------------
st.header("5. 연도별 평균기온")

st.dataframe(
    yearly_mean,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 10. 연도별 평균기온 그래프
# --------------------------------------------------
st.header("6. 연도별 평균기온 변화")

chart_data = yearly_mean.set_index("연도")[["평균기온"]]

st.line_chart(chart_data)

# --------------------------------------------------
# 11. 분석 요약
# --------------------------------------------------
st.header("7. 분석 요약")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 연도 수",
        f"{len(yearly_valid_days):,}년"
    )

with col2:
    st.metric(
        "최종 분석 연도",
        f"{len(valid_years):,}년"
    )

with col3:
    st.metric(
        "제외 연도",
        f"{len(excluded_years):,}년"
    )

st.info(
    "※ 연도별 평균기온은 평균기온 자료가 존재하는 날짜만 사용하여 계산했으며, "
    "유효 날짜가 350일 이상인 연도만 최종 분석에 포함했습니다."
)
```
