import datetime
import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="UPRO 스노우볼 전략 매매 앱",
    page_icon="❄️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# 모바일 친화적 스타일링
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 540px; }
    .order-card { background-color: #f8f9fa; border-radius: 12px; padding: 16px; border: 1px solid #e9ecef; margin-bottom: 12px; }
    .buy-box { background-color: #e8f5e9; border-radius: 10px; padding: 14px; color: #1b5e20; }
    .sell-box { background-color: #ffebee; border-radius: 10px; padding: 14px; color: #b71c1c; }
    .guide-banner { background-color: #fff3cd; color: #856404; padding: 10px 14px; border-radius: 8px; font-weight: 600; font-size: 0.9rem; margin-bottom: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- URL 쿼리 파라미터로 실전 잔고(ov) 및 투자 설정(cfg) 영구 보존 ---
def load_json_param(key: str) -> dict:
    raw = st.query_params.get(key, "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def save_json_param(key: str, data: dict):
    if data:
        st.query_params[key] = json.dumps(data, separators=(",", ":"))
    elif key in st.query_params:
        del st.query_params[key]


@st.cache_data(ttl=3600)
def load_upro_data() -> pd.DataFrame:
    df = yf.download("UPRO", period="5y", interval="1d", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    df["MA200"] = df["Close"].rolling(window=200).mean()
    return df


def get_week_key(dt: pd.Timestamp) -> str:
    iso = dt.isocalendar()
    return f"{iso.year}-{iso.week:02d}"


def run_upro_simulation(
    raw_df: pd.DataFrame,
    start_date: datetime.date,
    initial_shares: int,
    initial_cash: float,
    buy_band_pct: float,
    sell_band_pct: float,
    buy_steps: int,
    sell_steps: int,
    rebalance_months: int,
    stock_ratio_pct: float,
    use_ma200: bool,
    sgov_rate_pct: float,
    fx_rate: float,
    monthly_deposit: float,
    manual_overrides: dict,
):
    raw_df = raw_df.copy()
    raw_df["WeekKey"] = raw_df["Date"].apply(get_week_key)

    # 주간 전체(월~금) 고/저가 사전 집계
    weekly_stats = (
        raw_df.groupby("WeekKey")
        .agg(High=("High", "max"), Low=("Low", "min"))
        .to_dict("index")
    )
    all_weeks = sorted(weekly_stats.keys())

    # 초기 자산이 0이고 수동 입력 기록이 있으면 가장 빠른 수동 입력일부터 시작
    effective_start = start_date
    if initial_shares == 0 and initial_cash == 0 and manual_overrides:
        earliest_ov_date = datetime.date.fromisoformat(min(manual_overrides.keys()))
        if earliest_ov_date < effective_start:
            effective_start = earliest_ov_date

    start_ts = pd.Timestamp(effective_start)
    prior_df = raw_df[raw_df["Date"] < start_ts]
    df = raw_df[raw_df["Date"] >= start_ts].reset_index(drop=True)

    if df.empty:
        df = raw_df.tail(1).reset_index(drop=True)

    first_price = float(df.loc[0, "Close"])
    first_week = df.loc[0, "WeekKey"]

    exact_prev_high = first_price
    exact_prev_low = first_price

    if not prior_df.empty:
        prior_last_week = prior_df.iloc[-1]["WeekKey"]
        if prior_last_week != first_week and prior_last_week in weekly_stats:
            exact_prev_high = weekly_stats[prior_last_week]["High"]
            exact_prev_low = weekly_stats[prior_last_week]["Low"]
    elif first_week in all_weeks:
        idx = all_weeks.index(first_week)
        if idx > 0:
            prev_wk = all_weeks[idx - 1]
            exact_prev_high = weekly_stats[prev_wk]["High"]
            exact_prev_low = weekly_stats[prev_wk]["Low"]

    shares = int(initial_shares)
    cash = float(initial_cash)
    buy_band = 1.0 + (buy_band_pct / 100.0)
    sell_band = 1.0 + (sell_band_pct / 100.0)
    daily_interest = (sgov_rate_pct / 100.0) / 252.0
    stock_ratio = np.clip(stock_ratio_pct / 100.0, 0.0, 1.0)

    buy_step = 1
    sell_step = 1
    last_traded_week = ""
    has_triggered_ma200 = False
    total_monthly_injected = 0.0

    cur_week = first_week
    cur_week_high = float(df.loc[0, "High"])
    cur_week_low = float(df.loc[0, "Low"])

    last_reb_year = df.loc[0, "Date"].year
    last_reb_month = df.loc[0, "Date"].month
    prev_month = last_reb_month

    records = []

    for i, row in df.iterrows():
        dt = row["Date"]
        date_str = dt.strftime("%Y-%m-%d")
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        ma200 = row["MA200"]
        week_num = row["WeekKey"]

        if week_num != cur_week:
            exact_prev_high = cur_week_high
            exact_prev_low = cur_week_low
            cur_week_high = high
            cur_week_low = low
            cur_week = week_num
        else:
            cur_week_high = max(cur_week_high, high)
            cur_week_low = min(cur_week_low, low)

        # 1. 예수금 일할 이자
        cash += cash * daily_interest

        # 2. 월별 적립금
        day_actions = []
        if dt.month != prev_month and monthly_deposit > 0:
            cash += monthly_deposit
            total_monthly_injected += monthly_deposit
            day_actions.append(f"💰 적립 +{monthly_deposit:,.0f}원")
        prev_month = dt.month

        trade_note = ""

        # 3. 수동 실전 잔고 오버라이드 확인
        if date_str in manual_overrides:
            ov = manual_overrides[date_str]
            if int(ov["shares"]) != shares:
                last_traded_week = week_num
            shares = int(ov["shares"])
            cash = float(ov["cash"])
            trade_note = f"✏️ 실전 반영 ({shares:,}주 / {cash:,.0f}원)"
        else:
            target_buy = exact_prev_low * buy_band
            target_sell = exact_prev_high * sell_band

            is_cross_ma200 = False
            if use_ma200 and i > 0 and not has_triggered_ma200 and buy_step == 1:
                prev_row = df.iloc[i - 1]
                if pd.notna(prev_row["MA200"]) and pd.notna(ma200):
                    if prev_row["Close"] <= prev_row["MA200"] and close > ma200:
                        is_cross_ma200 = True

            # 4. 정기 리밸런싱 (자산이 있을 때만 실행)
            if rebalance_months > 0 and (shares > 0 or cash > 0):
                months_passed = (dt.year - last_reb_year) * 12 + (dt.month - last_reb_month)
                if months_passed >= rebalance_months:
                    total_asset = cash + (shares * close * fx_rate)
                    shares = int(np.floor((total_asset * stock_ratio) / (close * fx_rate)))
                    cash = total_asset - (shares * close * fx_rate)
                    buy_step = 1
                    sell_step = 1
                    has_triggered_ma200 = False
                    last_reb_year = dt.year
                    last_reb_month = dt.month
                    last_traded_week = week_num
                    trade_note = f"⚖️ 정기 리밸런싱({int(stock_ratio_pct)}:{int(100 - stock_ratio_pct)})"

            # 5. 주간 1회 분할 매수 / 200일선 돌파 / 분할 매도
            if trade_note == "" and week_num != last_traded_week:
                is_grid_buy = low <= target_buy
                if buy_step <= buy_steps and (is_grid_buy or is_cross_ma200) and cash > 0:
                    frac = 1.0 / (buy_steps - buy_step + 1)
                    buy_price = close if is_cross_ma200 else target_buy
                    bought = int(np.floor((cash * frac) / (buy_price * fx_rate)))
                    if bought > 0:
                        shares += bought
                        cash -= bought * buy_price * fx_rate
                        if is_cross_ma200:
                            has_triggered_ma200 = True
                        reason = "200일선 돌파" if is_cross_ma200 else f"목표가 ${target_buy:.2f}"
                        trade_note = f"🟢 매수 {buy_step}단계 ({bought}주 · {reason})"
                        buy_step += 1
                        sell_step = 1
                        last_traded_week = week_num

                elif high >= target_sell and shares > 0:
                    s_frac = 1.0 if (sell_step == sell_steps or sell_steps == 1) else (1.0 / (sell_steps - sell_step + 1))
                    sold = int(np.floor(shares * s_frac))
                    if sold > 0:
                        shares -= sold
                        cash += sold * target_sell * fx_rate
                        trade_note = f"🔴 매도 {sell_step}회차 ({sold}주 · 목표가 ${target_sell:.2f})"
                        last_traded_week = week_num
                        sell_step = 1 if sell_step >= sell_steps else sell_step + 1
                        buy_step = 1
                        has_triggered_ma200 = False

        if day_actions:
            prefix = "[" + " / ".join(day_actions) + "]"
            trade_note = f"{trade_note} {prefix}".strip()

        total_val = cash + (shares * close * fx_rate)
        records.append(
            {
                "날짜": date_str,
                "종가($)": round(close, 2),
                "보유수량": shares,
                "예수금(원)": round(cash),
                "총자산(원)": round(total_val),
                "매매기록": trade_note,
            }
        )

    res_df = pd.DataFrame(records)
    res_df["전고점"] = res_df["총자산(원)"].cummax()
    res_df["낙폭(%)"] = np.where(
        res_df["전고점"] > 0,
        ((res_df["총자산(원)"] - res_df["전고점"]) / res_df["전고점"]) * 100.0,
        0.0,
    )

    now_ny = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    this_week = get_week_key(now_ny)
    is_weekend_or_new_week = (this_week != cur_week) or (now_ny.weekday() >= 5)

    # 루프 내 부분 집계가 아닌 전체 주간 집계(weekly_stats)에서 정확한 고·저가 조회
    cur_week_full = weekly_stats.get(cur_week, {"High": cur_week_high, "Low": cur_week_low})
    prev_week_full = {"High": exact_prev_high, "Low": exact_prev_low}
    if cur_week in all_weeks:
        c_idx = all_weeks.index(cur_week)
        if c_idx > 0:
            prev_week_full = weekly_stats[all_weeks[c_idx - 1]]

    ref_high = cur_week_full["High"] if is_weekend_or_new_week else prev_week_full["High"]
    ref_low = cur_week_full["Low"] if is_weekend_or_new_week else prev_week_full["Low"]

    next_buy_price = ref_low * buy_band
    next_sell_price = ref_high * sell_band
    next_buy_amt = cash * (1.0 / (buy_steps - buy_step + 1))
    next_buy_qty = max(0, int(np.floor(next_buy_amt / (next_buy_price * fx_rate))))
    next_sell_frac = 1.0 if (sell_step == sell_steps or sell_steps == 1) else (1.0 / (sell_steps - sell_step + 1))
    next_sell_qty = max(0, int(np.floor(shares * next_sell_frac)))

    principal = (initial_shares * first_price * fx_rate) + initial_cash + total_monthly_injected
    final_asset = res_df.iloc[-1]["총자산(원)"]
    total_ret = ((final_asset - principal) / principal * 100.0) if principal > 0 else 0.0
    days = (df.iloc[-1]["Date"] - df.iloc[0]["Date"]).days
    cagr = ((final_asset / principal) ** (365.25 / days) - 1) * 100.0 if (days > 0 and principal > 0) else 0.0
    mdd = float(res_df["낙폭(%)"].min())

    return {
        "df": res_df,
        "ref_high": ref_high,
        "ref_low": ref_low,
        "next_buy_price": next_buy_price,
        "next_sell_price": next_sell_price,
        "next_buy_qty": next_buy_qty,
        "next_sell_qty": next_sell_qty,
        "buy_step": buy_step,
        "sell_step": sell_step,
        "principal": round(principal),
        "final_asset": round(final_asset),
        "total_ret": total_ret,
        "cagr": cagr,
        "mdd": mdd,
    }


# --- UI 렌더링 ---
st.title("❄️ UPRO 스노우볼 매매 앱")

if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = load_json_param("ov")

saved_cfg = load_json_param("cfg")

try:
    raw_df = load_upro_data()
except Exception as e:
    st.error(f"UPRO 주가 데이터를 불러오지 못했습니다: {e}")
    st.stop()

latest_db_date = raw_df.iloc[-1]["Date"].strftime("%Y-%m-%d")
min_db_date = raw_df.iloc[0]["Date"].date()
max_db_date = raw_df.iloc[-1]["Date"].date()
st.caption(f"야후 파이낸스 UPRO 실시간 연동 · 기준일: {latest_db_date}")

tab_order, tab_cfg, tab_stats = st.tabs(["📋 실전 주문서", "⚙️ 내 투자 설정", "📊 성과 및 기록"])

with tab_cfg:
    st.subheader("내 계좌 및 전략 파라미터")
    st.caption("설정을 변경하면 URL에 자동 기록되어 새로고침해도 유지됩니다.")

    default_start = max_db_date
    if "sd" in saved_cfg:
        try:
            parsed_sd = datetime.date.fromisoformat(saved_cfg["sd"])
            default_start = min(max(parsed_sd, min_db_date), max_db_date)
        except Exception:
            pass

    start_date = st.date_input(
        "투자 시작일",
        value=default_start,
        min_value=min_db_date,
        max_value=max_db_date,
    )

    c1, c2 = st.columns(2)
    initial_shares = c1.number_input("초기 UPRO 수량 (주)", min_value=0, value=int(saved_cfg.get("sh", 0)), step=1)
    initial_cash = c2.number_input("초기 예수금 (원)", min_value=0, value=int(saved_cfg.get("ca", 0)), step=500000)

    c3, c4 = st.columns(2)
    buy_band_pct = c3.number_input("매수 밴드 (전주 저가 %)", value=float(saved_cfg.get("bb", -5.0)), step=1.0)
    sell_band_pct = c4.number_input("매도 밴드 (전주 고가 %)", value=float(saved_cfg.get("sb", 5.0)), step=1.0)

    c5, c6 = st.columns(2)
    buy_steps = c5.number_input("매수 분할 횟수", min_value=1, max_value=20, value=int(saved_cfg.get("bs", 5)), step=1)
    sell_steps = c6.number_input("매도 분할 횟수", min_value=1, max_value=20, value=int(saved_cfg.get("ss", 4)), step=1)

    c7, c8 = st.columns(2)
    rebalance_months = c7.number_input("정기 리밸런싱 주기 (개월)", min_value=0, max_value=60, value=int(saved_cfg.get("rm", 12)), step=1)
    stock_ratio_pct = c8.number_input("리밸런싱 주식 비중 (%)", min_value=0, max_value=100, value=int(saved_cfg.get("sr", 50)), step=5)

    c9, c10 = st.columns(2)
    ma200_idx = 0 if saved_cfg.get("ma", True) else 1
    use_ma200 = c9.selectbox("200일선 상향 돌파 매수", options=[True, False], index=ma200_idx, format_func=lambda x: "사용함" if x else "미사용")
    fx_rate = c10.number_input("적용 환율 (원/$)", min_value=900, max_value=2000, value=int(saved_cfg.get("fx", 1350)), step=10)

    c11, c12 = st.columns(2)
    sgov_rate_pct = c11.number_input("예수금 연이자율 (%)", min_value=0.0, max_value=15.0, value=float(saved_cfg.get("sg", 4.5)), step=0.1)
    monthly_deposit = c12.number_input("월별 적립금 (원)", min_value=0, value=int(saved_cfg.get("md", 0)), step=100000)

    # 기본값과 다른 항목만 압축해 URL(cfg)에 자동 저장
    defaults = {
        "sd": max_db_date.isoformat(),
        "sh": 0,
        "ca": 0,
        "bb": -5.0,
        "sb": 5.0,
        "bs": 5,
        "ss": 4,
        "rm": 12,
        "sr": 50,
        "ma": True,
        "fx": 1350,
        "sg": 4.5,
        "md": 0,
    }
    current_cfg = {
        "sd": start_date.isoformat(),
        "sh": int(initial_shares),
        "ca": int(initial_cash),
        "bb": float(buy_band_pct),
        "sb": float(sell_band_pct),
        "bs": int(buy_steps),
        "ss": int(sell_steps),
        "rm": int(rebalance_months),
        "sr": int(stock_ratio_pct),
        "ma": bool(use_ma200),
        "fx": int(fx_rate),
        "sg": float(sgov_rate_pct),
        "md": int(monthly_deposit),
    }
    diff_cfg = {k: v for k, v in current_cfg.items() if v != defaults[k]}
    save_json_param("cfg", diff_cfg)

    if diff_cfg:
        if st.button("🔄 투자 설정 기본값으로 초기화", use_container_width=True):
            save_json_param("cfg", {})
            st.rerun()

res = run_upro_simulation(
    raw_df=raw_df,
    start_date=start_date,
    initial_shares=initial_shares,
    initial_cash=initial_cash,
    buy_band_pct=buy_band_pct,
    sell_band_pct=sell_band_pct,
    buy_steps=buy_steps,
    sell_steps=sell_steps,
    rebalance_months=rebalance_months,
    stock_ratio_pct=stock_ratio_pct,
    use_ma200=use_ma200,
    sgov_rate_pct=sgov_rate_pct,
    fx_rate=fx_rate,
    monthly_deposit=monthly_deposit,
    manual_overrides=st.session_state.manual_overrides,
)

latest_row = res["df"].iloc[-1]

with tab_order:
    st.markdown(
        f"""
        <div class="order-card">
            <div style="display:flex; justify-content:space-between; color:#6c757d; font-size:0.85rem; margin-bottom:4px;">
                <span>UPRO 최근 종가 ({latest_row['날짜']})</span>
                <span>보유 {latest_row['보유수량']:,}주 · 예수금 {latest_row['예수금(원)']:,}원</span>
            </div>
            <div style="font-size:2rem; font-weight:800; margin-bottom:8px;">${latest_row['종가($)']:.2f}</div>
            <div class="guide-banner">
                {latest_row['매매기록'] if latest_row['매매기록'] else '🔍 관망 (그리드 목표가 도달 대기 중)'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_b, col_s = st.columns(2)
    with col_b:
        st.markdown(
            f"""
            <div class="buy-box">
                <div style="font-size:0.8rem;">전주 저가(${res['ref_low']:.2f}) {buy_band_pct:+.0f}%</div>
                <div style="font-size:1.5rem; font-weight:800; margin:4px 0;">${res['next_buy_price']:.2f}</div>
                <div style="font-weight:700; font-size:0.9rem;">🛒 매수 {res['next_buy_qty']}주 ({res['buy_step']}단계)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s:
        st.markdown(
            f"""
            <div class="sell-box">
                <div style="font-size:0.8rem;">전주 고가(${res['ref_high']:.2f}) {sell_band_pct:+.0f}%</div>
                <div style="font-size:1.5rem; font-weight:800; margin:4px 0;">${res['next_sell_price']:.2f}</div>
                <div style="font-weight:700; font-size:0.9rem;">🎯 매도 {res['next_sell_qty']}주 ({res['sell_step']}회차)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("✏️ 실전 체결 잔고 반영")
    st.caption("증권사 계좌에서 실제 매매 체결 후 수량과 예수금을 입력하면 URL에 자동 저장되어 새로고침해도 유지됩니다.")

    with st.form("manual_trade_form"):
        m_date = st.date_input("체결 기준일", value=datetime.date.fromisoformat(latest_row["날짜"]))
        mc1, mc2 = st.columns(2)
        m_shares = mc1.number_input("실제 보유 수량 (주)", min_value=0, value=int(latest_row["보유수량"]), step=1)
        m_cash = mc2.number_input("실제 예수금 (원)", min_value=0, value=int(latest_row["예수금(원)"]), step=10000)
        submitted = st.form_submit_button("💾 실전 잔고 적용하기", use_container_width=True)
        if submitted:
            st.session_state.manual_overrides[m_date.strftime("%Y-%m-%d")] = {
                "shares": m_shares,
                "cash": m_cash,
            }
            save_json_param("ov", st.session_state.manual_overrides)
            st.rerun()

    if st.session_state.manual_overrides:
        if st.button("🗑️ 수동 입력 기록 초기화", use_container_width=True):
            st.session_state.manual_overrides = {}
            save_json_param("ov", {})
            st.rerun()

with tab_stats:
    m1, m2 = st.columns(2)
    m1.metric("최종 평가 자산", f"{res['final_asset']:,}원", f"{res['total_ret']:+.2f}%")
    m2.metric("연복리 수익률 (CAGR)", f"{res['cagr']:.2f}%", f"MDD {res['mdd']:.2f}%", delta_color="inverse")

    st.write(f"**총 투입 원금:** {res['principal']:,}원")
    st.line_chart(res["df"].set_index("날짜")[["총자산(원)", "전고점"]])

    st.subheader("최근 매매 및 잔고 내역")
    st.dataframe(
        res["df"].iloc[::-1].head(30),
        use_container_width=True,
        hide_index=True,
    )
