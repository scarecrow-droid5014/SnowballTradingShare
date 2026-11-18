import base64
import datetime
import hashlib
import json
import re
import numpy as np
import pandas as pd
import requests
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
    .user-badge { background-color: #eef2ff; color: #3730a3; padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; font-weight: 600; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- GitHub API 영구 저장소 연동 ---
def get_github_config():
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo = st.secrets.get("GITHUB_REPO", "scarecrow-droid5014/SnowballTradingShare")
    return token, repo


def sanitize_username(username: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9가-힣_-]", "", username.strip().lower())
    return cleaned[:30]


def hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.strip().encode("utf-8")).hexdigest()


def make_auto_token(username: str, pin_hash: str) -> str:
    """닉네임과 PIN 해시로 고정 자동로그인 인증키(16자리) 생성"""
    raw = f"{username}:{pin_hash}:snowball_auto_v1".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_user_from_github(username: str):
    token, repo = get_github_config()
    if not token:
        return None, None, "Streamlit Secrets에 GITHUB_TOKEN이 설정되지 않았습니다."

    safe_user = sanitize_username(username)
    if not safe_user:
        return None, None, "올바른 닉네임을 입력해 주세요."

    url = f"https://api.github.com/repos/{repo}/contents/user_data/{safe_user}.json"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    res = requests.get(url, headers=headers, timeout=10)
    if res.status_code == 200:
        body = res.json()
        raw_content = base64.b64decode(body["content"]).decode("utf-8")
        return json.loads(raw_content), body["sha"], None
    elif res.status_code == 404:
        return None, None, None
    else:
        return None, None, f"GitHub 조회 오류 ({res.status_code}): {res.text}"


def save_user_to_github(username: str, pin_hash: str, cfg: dict, overrides: dict, sha: str = None):
    token, repo = get_github_config()
    if not token:
        return None, "Streamlit Secrets에 GITHUB_TOKEN이 설정되지 않았습니다."

    safe_user = sanitize_username(username)
    url = f"https://api.github.com/repos/{repo}/contents/user_data/{safe_user}.json"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    if not sha:
        check_res = requests.get(url, headers=headers, timeout=10)
        if check_res.status_code == 200:
            sha = check_res.json().get("sha")

    payload_obj = {
        "username": safe_user,
        "pin_hash": pin_hash,
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cfg": cfg,
        "overrides": overrides,
    }
    encoded_content = base64.b64encode(
        json.dumps(payload_obj, ensure_ascii=False, indent=2).encode("utf-8")
    ).decode("utf-8")

    body = {
        "message": f"Update user data: {safe_user}",
        "content": encoded_content,
    }
    if sha:
        body["sha"] = sha

    put_res = requests.put(url, headers=headers, json=body, timeout=10)
    if put_res.status_code in (200, 201):
        new_sha = put_res.json().get("content", {}).get("sha")
        return new_sha, None
    elif put_res.status_code == 409:
        # 다중 기기 충돌 시 최신 sha 재조회 후 1회 재시도
        check_res = requests.get(url, headers=headers, timeout=10)
        if check_res.status_code == 200:
            latest_sha = check_res.json().get("sha")
            body["sha"] = latest_sha
            retry_res = requests.put(url, headers=headers, json=body, timeout=10)
            if retry_res.status_code in (200, 201):
                return retry_res.json().get("content", {}).get("sha"), None
        return sha, f"GitHub 저장 충돌 오류 ({put_res.status_code}): 다시 시도해 주세요."
    else:
        return sha, f"GitHub 저장 실패 ({put_res.status_code}): {put_res.text}"


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
    initial_cash_krw: float,
    initial_cash_usd: float,
    buy_band_pct: float,
    sell_band_pct: float,
    buy_steps: int,
    sell_steps: int,
    rebalance_months: int,
    stock_ratio_pct: float,
    use_ma200: bool,
    sgov_rate_pct: float,
    fx_rate: float,
    threshold_asset: float,
    monthly_deposit: float,
    monthly_withdrawal: float,
    extra_cash_amount: float,
    extra_cash_step: int,
    manual_overrides: dict,
):
    raw_df = raw_df.copy()
    raw_df["WeekKey"] = raw_df["Date"].apply(get_week_key)

    weekly_stats = (
        raw_df.groupby("WeekKey")
        .agg(High=("High", "max"), Low=("Low", "min"))
        .to_dict("index")
    )
    all_weeks = sorted(weekly_stats.keys())

    effective_start = start_date
    if initial_shares == 0 and initial_cash_krw == 0 and initial_cash_usd == 0 and manual_overrides:
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
    # 원화 예수금과 달러 예수금(원화 환산) 합산 운용
    cash = float(initial_cash_krw) + (float(initial_cash_usd) * fx_rate)
    initial_total_cash_krw = cash

    buy_band = 1.0 + (buy_band_pct / 100.0)
    sell_band = 1.0 + (sell_band_pct / 100.0)
    daily_interest = (sgov_rate_pct / 100.0) / 252.0
    stock_ratio = np.clip(stock_ratio_pct / 100.0, 0.0, 1.0)

    buy_step = 1
    sell_step = 1
    last_traded_week = ""
    has_triggered_ma200 = False
    total_extra_injected = 0.0
    total_monthly_injected = 0.0
    total_withdrawal = 0.0

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

        cash += cash * daily_interest

        day_actions = []
        if dt.month != prev_month:
            current_total_asset = cash + (shares * close * fx_rate)
            if current_total_asset < threshold_asset:
                if monthly_deposit > 0:
                    cash += monthly_deposit
                    total_monthly_injected += monthly_deposit
                    day_actions.append(f"💰 적립 +{monthly_deposit:,.0f}원")
            else:
                if monthly_withdrawal > 0:
                    if cash >= monthly_withdrawal:
                        cash -= monthly_withdrawal
                        day_actions.append(f"💸 인출 -{monthly_withdrawal:,.0f}원")
                    else:
                        needed = monthly_withdrawal - cash
                        sell_for_withdraw = int(np.ceil(needed / (close * fx_rate)))
                        cash = 0.0
                        shares = max(0, shares - sell_for_withdraw)
                        day_actions.append(f"💸 인출 -{monthly_withdrawal:,.0f}원({sell_for_withdraw}주 매도)")
                    total_withdrawal += monthly_withdrawal
        prev_month = dt.month

        trade_note = ""

        # 수동 실전 잔고 반영 (원화 + 달러 예수금 지원 및 기존 단일 cash 데이터 호환)
        if date_str in manual_overrides:
            ov = manual_overrides[date_str]
            new_shares = int(ov["shares"])
            ov_krw = float(ov.get("cash_krw", ov.get("cash", 0)))
            ov_usd = float(ov.get("cash_usd", 0.0))
            new_cash = ov_krw + (ov_usd * fx_rate)

            cash_desc = f"{ov_krw:,.0f}원 + ${ov_usd:,.2f}" if ov_usd > 0 else f"{ov_krw:,.0f}원"

            if new_shares > shares:
                disp_step = min(buy_step, buy_steps)
                trade_note = f"✏️ 실전 매수 {disp_step}단계 반영 ({new_shares:,}주 / {cash_desc})"
                buy_step = min(buy_steps + 1, buy_step + 1)
                sell_step = 1
                last_traded_week = week_num
            elif new_shares < shares:
                trade_note = f"✏️ 실전 매도 {sell_step}회차 반영 ({new_shares:,}주 / {cash_desc})"
                if new_shares == 0:
                    sell_step = 1
                else:
                    sell_step = 1 if sell_step >= sell_steps else sell_step + 1
                buy_step = 1
                has_triggered_ma200 = False
                last_traded_week = week_num
            else:
                trade_note = f"✏️ 실전 반영 ({new_shares:,}주 / {cash_desc})"

            shares = new_shares
            cash = new_cash
        else:
            target_buy = exact_prev_low * buy_band
            target_sell = exact_prev_high * sell_band

            is_cross_ma200 = False
            if use_ma200 and i > 0 and not has_triggered_ma200 and buy_step == 1:
                prev_row = df.iloc[i - 1]
                if pd.notna(prev_row["MA200"]) and pd.notna(ma200):
                    if prev_row["Close"] <= prev_row["MA200"] and close > ma200:
                        is_cross_ma200 = True

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

            if trade_note == "" and week_num != last_traded_week:
                is_grid_buy = low <= target_buy
                can_extra_inject = extra_cash_step > 0 and buy_step == extra_cash_step and extra_cash_amount > 0

                if buy_step <= buy_steps and (is_grid_buy or is_cross_ma200) and (cash > 0 or can_extra_inject):
                    if can_extra_inject:
                        cash += extra_cash_amount
                        total_extra_injected += extra_cash_amount
                        day_actions.append(f"🆘 위기투입 +{extra_cash_amount:,.0f}원")

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
        real_total_val = total_val + total_withdrawal
        records.append(
            {
                "날짜": date_str,
                "종가($)": round(close, 2),
                "보유수량": shares,
                "예수금합계(원)": round(cash),
                "계좌잔고(원)": round(total_val),
                "누적인출(원)": round(total_withdrawal),
                "실질총자산(원)": round(real_total_val),
                "매매기록": trade_note,
            }
        )

    res_df = pd.DataFrame(records)
    res_df["전고점"] = res_df["실질총자산(원)"].cummax()
    res_df["낙폭(%)"] = np.where(
        res_df["전고점"] > 0,
        ((res_df["실질총자산(원)"] - res_df["전고점"]) / res_df["전고점"]) * 100.0,
        0.0,
    )

    now_ny = pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    this_week = get_week_key(now_ny)
    is_weekend_or_new_week = (this_week != cur_week) or (now_ny.weekday() >= 5)

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

    # ZeroDivisionError 방지
    if buy_step <= buy_steps and next_buy_price > 0:
        next_buy_amt = cash * (1.0 / (buy_steps - buy_step + 1))
        next_buy_qty = max(0, int(np.floor(next_buy_amt / (next_buy_price * fx_rate))))
    else:
        next_buy_qty = 0

    next_sell_frac = 1.0 if (sell_step == sell_steps or sell_steps == 1) else (1.0 / (sell_steps - sell_step + 1))
    next_sell_qty = max(0, int(np.floor(shares * next_sell_frac)))

    principal = (initial_shares * first_price * fx_rate) + initial_total_cash_krw + total_monthly_injected + total_extra_injected
    final_asset = res_df.iloc[-1]["실질총자산(원)"]
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
        "total_withdrawal": round(total_withdrawal),
        "total_ret": total_ret,
        "cagr": cagr,
        "mdd": mdd,
    }


# --- UI 렌더링 ---
st.title("❄️ UPRO 스노우볼 매매 앱")

try:
    raw_df = load_upro_data()
except Exception as e:
    st.error(f"UPRO 주가 데이터를 불러오지 못했습니다: {e}")
    st.stop()

latest_db_date = raw_df.iloc[-1]["Date"].strftime("%Y-%m-%d")
min_db_date = raw_df.iloc[0]["Date"].date()
max_db_date = raw_df.iloc[-1]["Date"].date()
st.caption(f"야후 파이낸스 UPRO 실시간 연동 · 기준일: {latest_db_date}")

DEFAULT_CFG = {
    "sd": max_db_date.isoformat(),
    "sh": 0,
    "ca": 0,      # 초기 원화 예수금 (기존 데이터 호환)
    "ca_usd": 0.0,  # 초기 달러 예수금
    "bb": -5.0,
    "sb": 5.0,
    "bs": 5,
    "ss": 4,
    "rm": 12,
    "sr": 50,
    "ma": True,
    "fx": 1350,
    "sg": 4.5,
    "th": 100000000,
    "md": 0,
    "mw": 0,
    "ea": 0,
    "es": 0,
}

if "logged_user" not in st.session_state:
    st.session_state.logged_user = ""
if "pin_hash" not in st.session_state:
    st.session_state.pin_hash = ""
if "file_sha" not in st.session_state:
    st.session_state.file_sha = None
if "saved_cfg" not in st.session_state:
    st.session_state.saved_cfg = DEFAULT_CFG.copy()
if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = {}

# --- 자동 로그인 처리 (?u=닉네임&k=인증키) ---
qp_user = st.query_params.get("u", "")
qp_key = st.query_params.get("k", "")
if not st.session_state.logged_user and qp_user and qp_key:
    safe_u = sanitize_username(qp_user)
    if safe_u:
        existing_data, sha, err = load_user_from_github(safe_u)
        if existing_data and not err:
            stored_hash = existing_data.get("pin_hash", "")
            if make_auto_token(safe_u, stored_hash) == qp_key:
                merged = DEFAULT_CFG.copy()
                merged.update(existing_data.get("cfg", {}))
                st.session_state.logged_user = safe_u
                st.session_state.pin_hash = stored_hash
                st.session_state.file_sha = sha
                st.session_state.saved_cfg = merged
                st.session_state.manual_overrides = existing_data.get("overrides", {})

# --- 상단 내 계좌 불러오기 / 로그인 바 ---
if not st.session_state.logged_user:
    with st.expander("🔑 내 계좌 설정·기록 불러오기 (또는 새 사용자 등록)", expanded=True):
        st.caption("닉네임과 비밀번호(PIN)를 입력하면 클라우드에 내 설정과 매매 기록이 영구 저장됩니다.")
        with st.form("user_login_form"):
            lc1, lc2 = st.columns(2)
            in_user = lc1.text_input("닉네임 (영문/한글/숫자)", placeholder="예: lucas")
            in_pin = lc2.text_input("비밀번호 (PIN)", type="password", placeholder="4자리 이상")
            remember_me = st.checkbox("📱 이 기기에서 자동 로그인 유지 (홈 화면 추가/북마크용)", value=True)
            login_btn = st.form_submit_button("📂 불러오기 / 새 계좌 시작", use_container_width=True)

            if login_btn:
                safe_u = sanitize_username(in_user)
                if not safe_u or len(in_pin.strip()) < 4:
                    st.warning("닉네임과 4자리 이상의 비밀번호(PIN)를 입력해 주세요.")
                else:
                    p_hash = hash_pin(in_pin)
                    existing_data, sha, err = load_user_from_github(safe_u)
                    if err:
                        st.error(err)
                    elif existing_data is not None:
                        if existing_data.get("pin_hash") != p_hash:
                            st.error("비밀번호(PIN)가 일치하지 않습니다.")
                        else:
                            merged = DEFAULT_CFG.copy()
                            merged.update(existing_data.get("cfg", {}))
                            st.session_state.logged_user = safe_u
                            st.session_state.pin_hash = p_hash
                            st.session_state.file_sha = sha
                            st.session_state.saved_cfg = merged
                            st.session_state.manual_overrides = existing_data.get("overrides", {})
                            if remember_me:
                                st.query_params["u"] = safe_u
                                st.query_params["k"] = make_auto_token(safe_u, p_hash)
                            st.rerun()
                    else:
                        new_sha, save_err = save_user_to_github(safe_u, p_hash, DEFAULT_CFG.copy(), {}, None)
                        if save_err:
                            st.error(save_err)
                        else:
                            st.session_state.logged_user = safe_u
                            st.session_state.pin_hash = p_hash
                            st.session_state.file_sha = new_sha
                            st.session_state.saved_cfg = DEFAULT_CFG.copy()
                            st.session_state.manual_overrides = {}
                            if remember_me:
                                st.query_params["u"] = safe_u
                                st.query_params["k"] = make_auto_token(safe_u, p_hash)
                            st.rerun()
else:
    uc1, uc2 = st.columns([3, 1])
    uc1.markdown(
        f'<div class="user-badge">👤 접속 계정: <b>{st.session_state.logged_user}</b> (클라우드 자동 동기화 중)</div>',
        unsafe_allow_html=True,
    )
    if uc2.button("로그아웃", use_container_width=True):
        st.session_state.logged_user = ""
        st.session_state.pin_hash = ""
        st.session_state.file_sha = None
        st.session_state.saved_cfg = DEFAULT_CFG.copy()
        st.session_state.manual_overrides = {}
        st.query_params.clear()
        st.rerun()

saved_cfg = st.session_state.saved_cfg

tab_order, tab_cfg, tab_stats = st.tabs(["📋 실전 주문서", "⚙️ 내 투자 설정", "📊 성과 및 기록"])

with tab_cfg:
    st.subheader("내 계좌 및 전략 파라미터")
    if st.session_state.logged_user:
        st.caption(f"**{st.session_state.logged_user}** 계정의 설정과 잔고는 저장 시 GitHub 클라우드에 영구 기록됩니다.")
    else:
        st.warning("상단에서 닉네임과 PIN으로 로그인하면 설정과 매매 기록이 영구 저장됩니다.")

    try:
        parsed_sd = datetime.date.fromisoformat(saved_cfg["sd"])
        default_start = min(max(parsed_sd, min_db_date), max_db_date)
    except Exception:
        default_start = max_db_date

    with st.form("strategy_cfg_form"):
        c0_1, c0_2 = st.columns(2)
        start_date = c0_1.date_input(
            "투자 시작일",
            value=default_start,
            min_value=min_db_date,
            max_value=max_db_date,
        )
        initial_shares = c0_2.number_input("초기 UPRO 수량 (주)", min_value=0, value=int(saved_cfg["sh"]), step=1)

        c1, c2 = st.columns(2)
        initial_cash_krw = c1.number_input("초기 원화 예수금 (원)", min_value=0, value=int(saved_cfg.get("ca", 0)), step=500000)
        initial_cash_usd = c2.number_input("초기 달러 예수금 ($)", min_value=0.0, value=float(saved_cfg.get("ca_usd", 0.0)), step=100.0, format="%.2f")

        c3, c4 = st.columns(2)
        buy_band_pct = c3.number_input("매수 밴드 (전주 저가 %)", value=float(saved_cfg["bb"]), step=1.0)
        sell_band_pct = c4.number_input("매도 밴드 (전주 고가 %)", value=float(saved_cfg["sb"]), step=1.0)

        c5, c6 = st.columns(2)
        buy_steps = c5.number_input("매수 분할 횟수", min_value=1, max_value=20, value=int(saved_cfg["bs"]), step=1)
        sell_steps = c6.number_input("매도 분할 횟수", min_value=1, max_value=20, value=int(saved_cfg["ss"]), step=1)

        c7, c8 = st.columns(2)
        rebalance_months = c7.number_input("정기 리밸런싱 주기 (개월)", min_value=0, max_value=60, value=int(saved_cfg["rm"]), step=1)
        stock_ratio_pct = c8.number_input("리밸런싱 주식 비중 (%)", min_value=0, max_value=100, value=int(saved_cfg["sr"]), step=5)

        c9, c10 = st.columns(2)
        ma200_idx = 0 if saved_cfg["ma"] else 1
        use_ma200 = c9.selectbox("200일선 상향 돌파 매수", options=[True, False], index=ma200_idx, format_func=lambda x: "사용함" if x else "미사용")
        fx_rate = c10.number_input("적용 환율 (원/$)", min_value=900, max_value=2000, value=int(saved_cfg["fx"]), step=10)

        c11, c12 = st.columns(2)
        sgov_rate_pct = c11.number_input("예수금 연이자율 (%)", min_value=0.0, max_value=15.0, value=float(saved_cfg["sg"]), step=0.1)
        threshold_asset = c12.number_input("적립/인출 기준 자산 (원)", min_value=0, value=int(saved_cfg["th"]), step=10000000)

        c13, c14 = st.columns(2)
        monthly_deposit = c13.number_input("월별 적립금 (기준 미만 시)", min_value=0, value=int(saved_cfg["md"]), step=100000)
        monthly_withdrawal = c14.number_input("월별 인출금 (기준 이상 시)", min_value=0, value=int(saved_cfg["mw"]), step=100000)

        c15, c16 = st.columns(2)
        extra_cash_step = c15.number_input("위기 추가투입 단계 (0=미사용)", min_value=0, max_value=20, value=int(saved_cfg["es"]), step=1)
        extra_cash_amount = c16.number_input("위기 추가투입금 (원)", min_value=0, value=int(saved_cfg["ea"]), step=500000)

        save_cfg_btn = st.form_submit_button("💾 투자 설정 저장하기", use_container_width=True)
        if save_cfg_btn:
            new_cfg = {
                "sd": start_date.isoformat(),
                "sh": int(initial_shares),
                "ca": int(initial_cash_krw),
                "ca_usd": float(initial_cash_usd),
                "bb": float(buy_band_pct),
                "sb": float(sell_band_pct),
                "bs": int(buy_steps),
                "ss": int(sell_steps),
                "rm": int(rebalance_months),
                "sr": int(stock_ratio_pct),
                "ma": bool(use_ma200),
                "fx": int(fx_rate),
                "sg": float(sgov_rate_pct),
                "th": int(threshold_asset),
                "md": int(monthly_deposit),
                "mw": int(monthly_withdrawal),
                "ea": int(extra_cash_amount),
                "es": int(extra_cash_step),
            }
            st.session_state.saved_cfg = new_cfg
            if st.session_state.logged_user:
                new_sha, err = save_user_to_github(
                    st.session_state.logged_user,
                    st.session_state.pin_hash,
                    new_cfg,
                    st.session_state.manual_overrides,
                    st.session_state.file_sha,
                )
                if err:
                    st.error(err)
                else:
                    st.session_state.file_sha = new_sha
                    st.success("클라우드에 설정이 저장되었습니다!")
                    st.rerun()
            else:
                st.rerun()

    st.divider()
    st.subheader("🔗 앱 공유 링크")
    st.caption("지인에게 공유할 때는 아래 기본 주소를 복사해 보내세요 (내 계정 정보는 포함되지 않습니다).")
    st.code("https://snowballtradingshare.streamlit.app", language=None)

    if st.button("🔄 모든 설정 및 기록 초기화", use_container_width=True):
        st.session_state.saved_cfg = DEFAULT_CFG.copy()
        st.session_state.manual_overrides = {}
        if st.session_state.logged_user:
            new_sha, _ = save_user_to_github(
                st.session_state.logged_user,
                st.session_state.pin_hash,
                DEFAULT_CFG.copy(),
                {},
                st.session_state.file_sha,
            )
            st.session_state.file_sha = new_sha
        st.rerun()

res = run_upro_simulation(
    raw_df=raw_df,
    start_date=start_date,
    initial_shares=initial_shares,
    initial_cash_krw=initial_cash_krw,
    initial_cash_usd=initial_cash_usd,
    buy_band_pct=buy_band_pct,
    sell_band_pct=sell_band_pct,
    buy_steps=buy_steps,
    sell_steps=sell_steps,
    rebalance_months=rebalance_months,
    stock_ratio_pct=stock_ratio_pct,
    use_ma200=use_ma200,
    sgov_rate_pct=sgov_rate_pct,
    fx_rate=fx_rate,
    threshold_asset=threshold_asset,
    monthly_deposit=monthly_deposit,
    monthly_withdrawal=monthly_withdrawal,
    extra_cash_amount=extra_cash_amount,
    extra_cash_step=extra_cash_step,
    manual_overrides=st.session_state.manual_overrides,
)

latest_row = res["df"].iloc[-1]

# 실전 잔고 입력 폼 기본값 (가장 최근 수동 입력 기록이 있으면 분리된 원/달러 표시, 없으면 합산 예수금 표시)
default_m_shares = int(latest_row["보유수량"])
default_m_krw = int(latest_row["예수금합계(원)"])
default_m_usd = 0.0

if st.session_state.manual_overrides:
    last_ov_key = max(st.session_state.manual_overrides.keys())
    if last_ov_key == latest_row["날짜"]:
        last_ov = st.session_state.manual_overrides[last_ov_key]
        default_m_krw = int(last_ov.get("cash_krw", last_ov.get("cash", 0)))
        default_m_usd = float(last_ov.get("cash_usd", 0.0))
elif len(res["df"]) == 1 and initial_cash_usd > 0:
    default_m_krw = int(initial_cash_krw)
    default_m_usd = float(initial_cash_usd)

with tab_order:
    st.markdown(
        f"""
        <div class="order-card">
            <div style="display:flex; justify-content:space-between; color:#6c757d; font-size:0.85rem; margin-bottom:4px;">
                <span>UPRO 최근 종가 ({latest_row['날짜']})</span>
                <span>보유 {latest_row['보유수량']:,}주 · 예수금 {latest_row['예수금합계(원)']:,}원</span>
            </div>
            <div style="font-size:2rem; font-weight:800; margin-bottom:8px;">${latest_row['종가($)']:.2f}</div>
            <div class="guide-banner">
                {latest_row['매매기록'] if latest_row['매매기록'] else '🔍 관망 (그리드 목표가 도달 대기 중)'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    buy_est_usd = res["next_buy_price"] * res["next_buy_qty"]
    buy_est_krw = round(buy_est_usd * fx_rate)
    sell_est_usd = res["next_sell_price"] * res["next_sell_qty"]
    sell_est_krw = round(sell_est_usd * fx_rate)

    buy_step_label = f"{res['buy_step']}단계" if res["buy_step"] <= buy_steps else "분할완료"

    col_b, col_s = st.columns(2)
    with col_b:
        st.markdown(
            f"""
            <div class="buy-box">
                <div style="font-size:0.8rem;">전주 저가(${res['ref_low']:.2f}) {buy_band_pct:+.0f}%</div>
                <div style="font-size:1.5rem; font-weight:800; margin:4px 0;">${res['next_buy_price']:.2f}</div>
                <div style="font-weight:700; font-size:0.9rem;">🛒 매수 {res['next_buy_qty']}주 ({buy_step_label})</div>
                <div style="font-size:0.75rem; margin-top:4px; opacity:0.85;">예상 ${buy_est_usd:,.1f} (약 {buy_est_krw:,}원)</div>
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
                <div style="font-size:0.75rem; margin-top:4px; opacity:0.85;">예상 ${sell_est_usd:,.1f} (약 {sell_est_krw:,}원)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("✏️ 실전 체결 잔고 반영")
    st.caption(f"체결 후 증권사 계좌의 보유 수량과 원화·달러 예수금을 각각 입력하면 자동 합산(환율 {fx_rate:,}원)되어 저장됩니다.")

    with st.form("manual_trade_form"):
        fc1, fc2 = st.columns(2)
        m_date = fc1.date_input("체결 기준일", value=datetime.date.fromisoformat(latest_row["날짜"]))
        m_shares = fc2.number_input("실제 보유 수량 (주)", min_value=0, value=default_m_shares, step=1)

        mc1, mc2 = st.columns(2)
        m_cash_krw = mc1.number_input("실제 원화 예수금 (원)", min_value=0, value=default_m_krw, step=10000)
        m_cash_usd = mc2.number_input("실제 달러 예수금 ($)", min_value=0.0, value=default_m_usd, step=10.0, format="%.2f")

        submitted = st.form_submit_button("💾 실전 잔고 적용하기", use_container_width=True)
        if submitted:
            st.session_state.manual_overrides[m_date.strftime("%Y-%m-%d")] = {
                "shares": int(m_shares),
                "cash_krw": int(m_cash_krw),
                "cash_usd": float(m_cash_usd),
            }
            if st.session_state.logged_user:
                new_sha, err = save_user_to_github(
                    st.session_state.logged_user,
                    st.session_state.pin_hash,
                    st.session_state.saved_cfg,
                    st.session_state.manual_overrides,
                    st.session_state.file_sha,
                )
                if err:
                    st.error(err)
                else:
                    st.session_state.file_sha = new_sha
                    st.rerun()
            else:
                st.rerun()

    if st.session_state.manual_overrides:
        if st.button("🗑️ 수동 입력 기록 초기화", use_container_width=True):
            st.session_state.manual_overrides = {}
            if st.session_state.logged_user:
                new_sha, _ = save_user_to_github(
                    st.session_state.logged_user,
                    st.session_state.pin_hash,
                    st.session_state.saved_cfg,
                    {},
                    st.session_state.file_sha,
                )
                st.session_state.file_sha = new_sha
            st.rerun()

with tab_stats:
    m1, m2 = st.columns(2)
    m1.metric("실질 총자산", f"{res['final_asset']:,}원", f"{res['total_ret']:+.2f}%")
    m2.metric("연복리 수익률 (CAGR)", f"{res['cagr']:.2f}%", f"MDD {res['mdd']:.2f}%", delta_color="inverse")

    st.write(f"**총 투입 원금:** {res['principal']:,}원 · **누적 인출액:** {res['total_withdrawal']:,}원")
    st.line_chart(res["df"].set_index("날짜")[["실질총자산(원)", "전고점"]])

    st.subheader("최근 매매 및 잔고 내역")
    st.dataframe(
        res["df"].iloc[::-1].head(30),
        use_container_width=True,
        hide_index=True,
    )
