import streamlit as st
import pandas as pd

st.title("서울 연도별 평균기온 분석")

URL = "https://raw.githubusercontent.com/greatsong/modudata/refs/heads/main/data/seoul_daily.csv"

# 데이터 읽기
df = pd.read_csv(URL)

# 날짜와 평균기온 정리
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

# --------------------------------------------------
# 1. 기본 데이터 정보
# --------------------------------------------------

st.header("1. 데이터 기본 정보")

st.write("열 이름")
st.write(list(df.columns))

기록시작 = df["날짜"].min()
기록종료 = df["날짜"].max()

st.write(f"기록 기간: {기록시작.date()} ~ {기록종료.date()}")
st.write(f"전체 데이터 수: {len(df):,}개")
st.write(f"평균기온 결측 수: {df['평균기온'].isna().sum():,}개")

# 같은 날짜 중복 확인
중복날짜수 = df["날짜"].duplicated().sum()

if 중복날짜수 == 0:
    st.write("같은 날짜의 중복 데이터: 없음")
else:
    st.write(f"같은 날짜의 중복 데이터: {중복날짜수:,}개")


# --------------------------------------------------
# 2. 연도별 유효 날짜 수
# --------------------------------------------------

st.header("2. 연도별 평균기온 유효 날짜 수")

df["연도"] = df["날짜"].dt.year

# 기록 기간에 포함되는 모든 연도를 만듦
모든연도 = range(기록시작.year, 기록종료.year + 1)

# 평균기온이 실제로 있는 날짜 수 계산
연도별유효날짜 = (
    df.dropna(subset=["평균기온"])
      .groupby("연도")
      .size()
      .reindex(모든연도, fill_value=0)
)

연도별유효날짜 = 연도별유효날짜.rename("평균기온이 있는 날짜 수")

st.dataframe(
    연도별유효날짜.reset_index().rename(columns={"index": "연도"}),
    use_container_width=True
)


# --------------------------------------------------
# 3. 350일 미만 연도 제외
# --------------------------------------------------

st.header("3. 분석에서 제외한 연도")

제외연도 = 연도별유효날짜[연도별유효날짜 < 350]

if len(제외연도) == 0:
    st.write("350일 미만인 연도가 없습니다.")
else:
    제외표 = 제외연도.reset_index()
    제외표.columns = ["연도", "평균기온이 있는 날짜 수"]

    st.dataframe(
        제외표,
        use_container_width=True
    )

    st.write(f"제외된 연도 수: {len(제외표)}개")


# --------------------------------------------------
# 4. 최종 분석 대상
# --------------------------------------------------

분석연도 = 연도별유효날짜[연도별유효날짜 >= 350].index

분석자료 = df[
    df["연도"].isin(분석연도) &
    df["평균기온"].notna()
].copy()

연도별평균기온 = (
    분석자료
    .groupby("연도")["평균기온"]
    .mean()
    .reset_index()
)

연도별평균기온.columns = ["연도", "평균기온"]

st.header("4. 최종 분석 대상 연도의 평균기온")

st.dataframe(
    연도별평균기온,
    use_container_width=True
)

st.write(f"최종 분석 연도 수: {len(연도별평균기온)}개")

# 그래프
st.header("5. 연도별 평균기온")

그래프자료 = 연도별평균기온.set_index("연도")

st.line_chart(그래프자료)
