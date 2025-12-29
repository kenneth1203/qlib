#!/usr/bin/env python3
"""
Bull vs Bear Decision Module

Consumes prediction output from predict_today and gates buy decisions
for selected instruments using widely known technical indicators.

Usage:
  python -m qlib.tests.bull_bear_decision \
    --recorder_id <run_id> \
    --experiment_name workflow \
    --provider_uri ~/.qlib/qlib_data/hk_data \
    --topk 10 \
    --lookback 180 \
    --liq_threshold 1000000 \
    --liq_window 20

This script prefers a persisted selection CSV `selected_today_<recorder_id>.csv`.
If missing, it reconstructs selected instruments from `pred_today_<recorder_id>.pkl`.
"""
import argparse
import os
import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_HK
from qlib.workflow import R
from qlib.data import D


def to_qlib_inst(x: str) -> str:
    s = str(x).lower()
    if s.startswith("hk."):
        s = s.split(".", 1)[1]
    if "." in s:
        s = s.split(".", 1)[0]
    if s.isdigit():
        s = s.zfill(5)
    return s.upper() + ".HK"


def calendar_last_day(today: datetime.date) -> str:
    cal = D.calendar(
        start_time=(today - datetime.timedelta(days=14)).strftime("%Y-%m-%d"),
        end_time=today.strftime("%Y-%m-%d"),
        freq="day",
    )
    return today.strftime("%Y-%m-%d") if len(cal) == 0 else cal[-1]


def load_selection(recorder_id: str, topk: int, target_day: str) -> List[str]:
    """Load selected instruments for target_day.
    Prefer CSV next to working dir; fallback to pred pickle reconstruction.
    """
    csv_path = f"selected_today_{recorder_id}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c.lower().startswith("instrument")]
        if len(cols) > 0:
            return [to_qlib_inst(i) for i in df[cols[0]].astype(str).tolist()][:topk]

    # fallback: load prediction pickle
    pkl_path = f"pred_today_{recorder_id}.pkl"
    if not os.path.exists(pkl_path):
        raise RuntimeError(f"Missing selection CSV and prediction pickle: {csv_path}, {pkl_path}")

    pred = pd.read_pickle(pkl_path)
    # normalize to Series and filter to target day
    if isinstance(pred, pd.DataFrame):
        s = pred.iloc[:, 0]
    else:
        s = pred
    if not isinstance(s, pd.Series):
        raise RuntimeError(f"Unexpected prediction type: {type(pred)}")
    names = list(getattr(s.index, "names", []))
    dt_level = names.index("datetime") if "datetime" in names else None
    inst_level = names.index("instrument") if "instrument" in names else None
    ss = s
    if dt_level is not None:
        try:
            ts_target = pd.Timestamp(target_day)
            mask = ss.index.get_level_values(dt_level) == ts_target
            slice_ss = ss[mask]
            if len(slice_ss) > 0:
                ss = slice_ss
        except Exception:
            pass
    if isinstance(ss.index, pd.MultiIndex) and inst_level is not None and len(ss.index.names) > 1:
        try:
            ss = ss.groupby(level=inst_level).last()
        except Exception:
            # drop non-instrument levels
            keep = [inst_level]
            drop_levels = [n for i, n in enumerate(ss.index.names) if i not in keep]
            ss = ss.droplevel(drop_levels)
    top = ss.sort_values(ascending=False).head(max(topk, 500))
    return [to_qlib_inst(i) for i in top.index][:topk]


def fetch_ohlcv(instruments: List[str], start_dt: str, end_dt: str) -> pd.DataFrame:
    # try to include $vwap if available
    fields = ["$close", "$high", "$low", "$volume", "$vwap"]
    try:
        df = D.features(instruments, fields, start_time=start_dt, end_time=end_dt, freq="day")
        # some providers may lack $vwap; align columns safely
        cols = list(df.columns)
        rename = {}
        if len(cols) >= 4:
            rename = {cols[0]: "$close", cols[1]: "$high", cols[2]: "$low", cols[3]: "$volume"}
            if len(cols) >= 5:
                rename[cols[4]] = "$vwap"
        df.columns = [rename.get(c, c) for c in cols]
    except Exception:
        df = D.features(instruments, ["$close", "$high", "$low", "$volume"], start_time=start_dt, end_time=end_dt, freq="day")
        df.columns = ["$close", "$high", "$low", "$volume"]
    return df


def try_fetch_hsi(start_dt: str, end_dt: str) -> pd.DataFrame:
    """Fetch HSI OHLC via qlib if available, else fallback to yahooquery.
    Returns DataFrame with columns: $close,$high,$low indexed by datetime.
    """
    # Try qlib first (if HSI exists)
    try:
        df = D.features(["800000.HK"], ["$close", "$high", "$low"], start_time=start_dt, end_time=end_dt, freq="day")
        if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
            df.columns = ["$close", "$high", "$low"]
            # reduce to single index by datetime
            try:
                df = df.reset_index().set_index("datetime").sort_index()
            except Exception:
                pass
            return df[["$close", "$high", "$low"]]
    except Exception:
        pass
    # Fallback: yahooquery ^HSI
    try:
        from yahooquery import Ticker  # type: ignore
        t = Ticker("^HSI", asynchronous=False)
        hist = t.history(interval="1d", start=start_dt, end=end_dt)
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            df = hist.copy()
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            if "date" in df.columns:
                df = df.rename(columns={"date": "datetime"})
            df = df.set_index("datetime").sort_index()
            # align columns like qlib
            cols = {}
            if "close" in df.columns: cols["close"] = "$close"
            if "high" in df.columns: cols["high"] = "$high"
            if "low" in df.columns: cols["low"] = "$low"
            df = df.rename(columns=cols)
            need = ["$close", "$high", "$low"]
            if all(c in df.columns for c in need):
                return df[need]
    except Exception:
        pass
    return pd.DataFrame(columns=["$close", "$high", "$low"])


# ---- Indicator computations (pandas) ----

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(close: pd.Series, period=14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def roc(close: pd.Series, period=10) -> pd.Series:
    # Explicitly disable NA padding to avoid FutureWarning from pandas
    return close.pct_change(periods=period, fill_method=None)


def momentum(close: pd.Series, period=10) -> pd.Series:
    return close.diff(period)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume.fillna(0)).cumsum()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    return true_range(high, low, close).rolling(period).mean()


def rsv(close: pd.Series, high: pd.Series, low: pd.Series, period=9) -> pd.Series:
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    denom = (hh - ll).replace(0, np.nan)
    return (close - ll) / denom * 100


def kdj(close: pd.Series, high: pd.Series, low: pd.Series, period=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    rsv_s = rsv(close, high, low, period).fillna(50)
    K = rsv_s.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J


def williams_r(close: pd.Series, high: pd.Series, low: pd.Series, period=14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    denom = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / denom


def supertrend(close: pd.Series, high: pd.Series, low: pd.Series, period=10, multiplier=3.0) -> Tuple[pd.Series, pd.Series]:
    # returns (supertrend value, direction: +1 bull / -1 bear)
    atr_s = atr(high, low, close, period)
    basic_ub = ((high + low) / 2) + multiplier * atr_s
    basic_lb = ((high + low) / 2) - multiplier * atr_s
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    for i in range(1, len(close)):
        final_ub.iloc[i] = basic_ub.iloc[i] if (basic_ub.iloc[i] < final_ub.iloc[i-1]) or (close.iloc[i-1] > final_ub.iloc[i-1]) else final_ub.iloc[i-1]
        final_lb.iloc[i] = basic_lb.iloc[i] if (basic_lb.iloc[i] > final_lb.iloc[i-1]) or (close.iloc[i-1] < final_lb.iloc[i-1]) else final_lb.iloc[i-1]
    st = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)
    st.iloc[0] = final_ub.iloc[0]
    direction.iloc[0] = -1
    for i in range(1, len(close)):
        if st.iloc[i-1] == final_ub.iloc[i-1]:
            st.iloc[i] = final_ub.iloc[i] if close.iloc[i] <= final_ub.iloc[i] else final_lb.iloc[i]
        else:
            st.iloc[i] = final_lb.iloc[i] if close.iloc[i] >= final_lb.iloc[i] else final_ub.iloc[i]
        direction.iloc[i] = 1 if close.iloc[i] >= st.iloc[i] else -1
    return st, direction


def bollinger_bands(close: pd.Series, period=20, std_mult=2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, width


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    # Wilder's ADX with proper directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr_w = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period=20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    cci_val = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci_val


def donchian(high: pd.Series, low: pd.Series, period=20) -> Tuple[pd.Series, pd.Series]:
    up = high.rolling(period).max()
    dn = low.rolling(period).min()
    return up, dn


def rsi_divergence(close: pd.Series, rsi_s: pd.Series, window: int = 30) -> Tuple[bool, bool]:
    # bullish: price makes lower low, RSI does not make lower low
    # bearish: price makes higher high, RSI does not make higher high
    if len(close) < window + 1:
        return False, False
    c_win = close.iloc[-window:]
    r_win = rsi_s.iloc[-window:]
    # compare last value to prior window extrema
    c_last = c_win.iloc[-1]
    r_last = r_win.iloc[-1]
    c_prev_min = c_win.iloc[:-1].min()
    c_prev_max = c_win.iloc[:-1].max()
    r_prev_min = r_win.iloc[:-1].min()
    r_prev_max = r_win.iloc[:-1].max()
    bullish = (c_last < c_prev_min) and (r_last >= r_prev_min)
    bearish = (c_last > c_prev_max) and (r_last <= r_prev_max)
    return bullish, bearish


def score_instrument(df: pd.DataFrame, params: Dict) -> Tuple[int, int, Dict]:
    """Compute Bull/Bear score for one instrument dataframe with columns $close,$high,$low,$volume.
    Returns bull_score, bear_score, snapshot dict (last values).
    """
    c = df["$close"].astype(float)
    h = df["$high"].astype(float)
    l = df["$low"].astype(float)
    v = df["$volume"].astype(float)
    vwap = df["$vwap"].astype(float) if "$vwap" in df.columns else ((h + l + c) / 3.0)
    min_needed = max(
        params["ma_120"],
        params["macd_slow"] + params["macd_signal"],
        params["rsi_window"],
        params["vol_ma_window"],
        params["atr_ma_window"],
        params["bb_window"],
        params["adx_window"],
        params["cci_window"],
        params["don_window"],
        params["z_window"],
        params["lookback_min"],
    )
    allow_partial = bool(params.get("allow_partial", False))
    if len(c) < min_needed and not allow_partial:
        # insufficient history
        return 0, 1, {"reason": "insufficient_history"}

    ma10 = c.rolling(params["ma_10"]).mean()
    ma20 = c.rolling(params["ma_20"]).mean()
    ma60 = c.rolling(params["ma_60"]).mean()
    ma120 = c.rolling(params["ma_120"]).mean()
    macd_line, macd_sig, _ = macd(c, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    rsi_val = rsi(c, params["rsi_window"])
    roc_val = roc(c, params["roc_window"]) 
    mom_val = momentum(c, params["mom_window"]) 
    vol_ma = v.rolling(params["vol_ma_window"]).mean()
    obv_s = obv(c, v)
    obv_slope = (obv_s.diff(params.get("slope_window", 5))).iloc[-1]
    atr_s = atr(h, l, c, params["atr_window"]) 
    atr_change = atr_s.diff().iloc[-1]
    atr_ma = atr_s.rolling(params["atr_ma_window"]).mean()
    K, D, J = kdj(c, h, l, params["kdj_window"]) 
    wpr = williams_r(c, h, l, params["wpr_window"]) 
    st_val, st_dir = supertrend(c, h, l, params["st_period"], params["st_multiplier"]) 
    mid, bb_up, bb_dn, bb_width = bollinger_bands(c, params["bb_window"], params["bb_mult"]) 
    adx_val = adx(h, l, c, params["adx_window"]) 
    cci_val = cci(h, l, c, params["cci_window"]) 
    don_up, don_dn = donchian(h, l, params["don_window"]) 
    zscore = (c - ma20) / (c.rolling(params["z_window"]).std(ddof=0).replace(0, np.nan))
    rsi_bull_div, rsi_bear_div = rsi_divergence(c, rsi_val, window=params["rsi_div_window"])

    # Relative Strength vs market (stock return minus market return over window)
    rs_val = np.nan
    rs_window = params.get("rs_window", 60)
    mkt_close = params.get("market_close")
    if mkt_close is not None and isinstance(mkt_close, pd.Series):
        try:
            mkt_aligned = mkt_close.reindex(c.index)
            if len(c) > rs_window and pd.notna(c.iloc[-rs_window]) and pd.notna(mkt_aligned.iloc[-rs_window]) and pd.notna(mkt_aligned.iloc[-1]):
                stock_ret = c.iloc[-1] / c.iloc[-rs_window] - 1.0
                mkt_ret = mkt_aligned.iloc[-1] / mkt_aligned.iloc[-rs_window] - 1.0
                rs_val = stock_ret - mkt_ret
        except Exception:
            rs_val = np.nan

    bull = 0
    bear = 0
    bull_sources = []
    bear_sources = []

    def add_bull(cond: bool, label: str):
        nonlocal bull
        if cond:
            bull += 1
            bull_sources.append(label)

    def add_bear(cond: bool, label: str):
        nonlocal bear
        if cond:
            bear += 1
            bear_sources.append(label)

    # Trend (multi-MA)
    if pd.notna(ma10.iloc[-1]) and pd.notna(ma20.iloc[-1]):
        add_bull((ma10.iloc[-1] > ma20.iloc[-1]), "MA10>MA20")
        add_bear((ma10.iloc[-1] <= ma20.iloc[-1]), "MA10<=MA20")
    elif not allow_partial:
        add_bear(True, "MA10/MA20_missing")
    if pd.notna(ma20.iloc[-1]) and pd.notna(ma60.iloc[-1]):
        add_bull((ma20.iloc[-1] > ma60.iloc[-1]), "MA20>MA60")
        add_bear((ma20.iloc[-1] <= ma60.iloc[-1]), "MA20<=MA60")
    elif not allow_partial:
        add_bear(True, "MA20/MA60_missing")
    if pd.notna(ma60.iloc[-1]) and pd.notna(ma120.iloc[-1]):
        add_bull((ma60.iloc[-1] > ma120.iloc[-1]), "MA60>MA120")
        add_bear((ma60.iloc[-1] <= ma120.iloc[-1]), "MA60<=MA120")
    elif not allow_partial:
        add_bear(True, "MA60/MA120_missing")

    # MACD cross + zero-axis
    if pd.notna(macd_line.iloc[-1]) and pd.notna(macd_sig.iloc[-1]):
        add_bull((macd_line.iloc[-1] > macd_sig.iloc[-1]), "MACD_cross_up")
        add_bear((macd_line.iloc[-1] <= macd_sig.iloc[-1]), "MACD_cross_down")
    elif not allow_partial:
        add_bear(True, "MACD_missing")
    if pd.notna(macd_line.iloc[-1]):
        add_bull((macd_line.iloc[-1] > 0), "MACD_above_zero")
        add_bear((macd_line.iloc[-1] <= 0), "MACD_below_zero")
    elif not allow_partial:
        add_bear(True, "MACD_zero_missing")

    # Momentum
    add_bull((rsi_val.iloc[-1] > 50), "RSI>50")
    add_bear((rsi_val.iloc[-1] > 70), "RSI>70_overbought")
    add_bull((rsi_val.iloc[-1] < 30), "RSI<30_oversold")
    add_bull(rsi_bull_div, "RSI_bull_div")
    add_bear(rsi_bear_div, "RSI_bear_div")

    add_bull((roc_val.iloc[-1] > 0), "ROC>0")
    add_bull((mom_val.iloc[-1] > 0), "Momentum>0")

    # Relative Strength vs market
    if pd.notna(rs_val):
        add_bull(rs_val > 0, "RS>0")
        add_bear(rs_val < 0, "RS<0")
    elif not allow_partial:
        add_bear(True, "RS_missing")

    # Volume
    if pd.notna(v.iloc[-1]) and pd.notna(vol_ma.iloc[-1]):
        add_bull((v.iloc[-1] > vol_ma.iloc[-1]), "Vol>VolMA")
        add_bull((v.iloc[-1] > params["vol_surge_mult"] * vol_ma.iloc[-1]), "Vol_surge")
    elif not allow_partial:
        add_bear(True, "Volume_missing")
    # VWAP deviation
    if pd.notna(c.iloc[-1]) and pd.notna(vwap.iloc[-1]):
        add_bull((c.iloc[-1] > vwap.iloc[-1]), "Close>VWAP")
        add_bear((c.iloc[-1] <= vwap.iloc[-1]), "Close<=VWAP")
    elif not allow_partial:
        add_bear(True, "VWAP_missing")
    add_bull((obv_slope is not None and obv_slope > 0), "OBV_up")

    # Volatility
    add_bull((atr_change is not None and atr_change > 0), "ATR_rising")
    add_bear((atr_change is not None and atr_change < 0), "ATR_falling")
    # ATR squeeze
    add_bear(pd.notna(atr_s.iloc[-1]) and pd.notna(atr_ma.iloc[-1]) and (atr_s.iloc[-1] < params["atr_squeeze_mult"] * atr_ma.iloc[-1]), "ATR_squeeze")

    # Overbought/Oversold
    add_bear((K.iloc[-1] > 80), "K>80_overbought")
    add_bull((K.iloc[-1] < 20), "K<20_oversold")
    add_bear((wpr.iloc[-1] > -20), "W%R>-20_overbought")
    add_bull((wpr.iloc[-1] < -80), "W%R<-80_oversold")

    # SuperTrend
    if pd.notna(st_dir.iloc[-1]):
        add_bull((st_dir.iloc[-1] == 1), "Supertrend_up")
        add_bear((st_dir.iloc[-1] != 1), "Supertrend_down")
    elif not allow_partial:
        add_bear(True, "Supertrend_missing")

    # Bollinger squeeze
    # consider squeeze if bandwidth in bottom quantile over window
    try:
        bw_series = bb_width.dropna()
        if len(bw_series) >= params["bb_window"]:
            thresh = bw_series.quantile(params["bb_squeeze_quantile"])
            if bb_width.iloc[-1] <= thresh:
                add_bear(True, "BB_squeeze")
    except Exception:
        pass

    # ADX trend strength
    if pd.notna(adx_val.iloc[-1]):
        add_bull(adx_val.iloc[-1] >= 25, "ADX_strong_trend")
        add_bear(adx_val.iloc[-1] <= 20, "ADX_weak_trend")

    # CCI sensitivity
    if pd.notna(cci_val.iloc[-1]):
        add_bull(cci_val.iloc[-1] > 100, "CCI>100")
        add_bear(cci_val.iloc[-1] < -100, "CCI<-100")

    # Donchian breakout
    if len(don_up) >= 2 and len(don_dn) >= 2 and pd.notna(don_up.iloc[-2]) and pd.notna(don_dn.iloc[-2]):
        add_bull(c.iloc[-1] > don_up.iloc[-2], "Donchian_breakout_up")
        add_bear(c.iloc[-1] < don_dn.iloc[-2], "Donchian_breakdown")

    # Z-score extremes
    if pd.notna(zscore.iloc[-1]):
        add_bear(zscore.iloc[-1] > 2, "Zscore>2")
        add_bull(zscore.iloc[-1] < -2, "Zscore<-2")

    snap = {
        "ma10": float(ma10.iloc[-1]),
        "ma20": float(ma20.iloc[-1]),
        "ma60": float(ma60.iloc[-1]),
        "ma120": float(ma120.iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "signal": float(macd_sig.iloc[-1]),
        "rsi": float(rsi_val.iloc[-1]),
        "roc": float(roc_val.iloc[-1]),
        "mom": float(mom_val.iloc[-1]),
        "vol": float(v.iloc[-1]),
        "vol_ma20": float(vol_ma.iloc[-1]),
        "obv_slope": float(obv_slope) if pd.notna(obv_slope) else 0.0,
        "atr": float(atr_s.iloc[-1]),
        "atr_change": float(atr_change) if pd.notna(atr_change) else 0.0,
        "kdj_k": float(K.iloc[-1]),
        "wpr": float(wpr.iloc[-1]),
        "supertrend_dir": int(st_dir.iloc[-1]),
        "bb_width": float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else 0.0,
        "adx": float(adx_val.iloc[-1]) if pd.notna(adx_val.iloc[-1]) else 0.0,
        "cci": float(cci_val.iloc[-1]) if pd.notna(cci_val.iloc[-1]) else 0.0,
        "don_up": float(don_up.iloc[-2]) if pd.notna(don_up.iloc[-2]) else 0.0,
        "don_dn": float(don_dn.iloc[-2]) if pd.notna(don_dn.iloc[-2]) else 0.0,
        "zscore": float(zscore.iloc[-1]) if pd.notna(zscore.iloc[-1]) else 0.0,
        "vwap": float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else float('nan'),
        "rsi_bull_div": bool(rsi_bull_div),
        "rsi_bear_div": bool(rsi_bear_div),
        "rs": float(rs_val) if pd.notna(rs_val) else float('nan'),
        "bull_sources": "|".join(bull_sources),
        "bear_sources": "|".join(bear_sources),
    }
    return bull, bear, snap


def main(args):
    provider_uri = os.path.expanduser(args.provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_HK)

    today_dt = datetime.date.today()
    target_day = calendar_last_day(today_dt)
    print("target_day:", target_day)

    # selection
    selected = load_selection(args.recorder_id, args.topk, target_day)
    if len(selected) == 0:
        raise RuntimeError("No selected instruments.")
    print("Selected:", selected)

    # Load model scores from pred_today_<recorder_id>.pkl if available
    model_scores: Dict[str, float] = {}
    try:
        pkl_path = f"pred_today_{args.recorder_id}.pkl"
        if os.path.exists(pkl_path):
            pred = pd.read_pickle(pkl_path)
            if isinstance(pred, pd.DataFrame):
                s = pred.iloc[:, 0]
            else:
                s = pred
            if isinstance(s, pd.Series):
                names = list(getattr(s.index, "names", []))
                dt_level = names.index("datetime") if "datetime" in names else None
                inst_level = names.index("instrument") if "instrument" in names else None
                ss = s
                if dt_level is not None:
                    try:
                        ts_target = pd.Timestamp(target_day)
                        mask = ss.index.get_level_values(dt_level) == ts_target
                        slice_ss = ss[mask]
                        if len(slice_ss) > 0:
                            ss = slice_ss
                    except Exception:
                        pass
                if isinstance(ss.index, pd.MultiIndex) and inst_level is not None and len(ss.index.names) > 1:
                    try:
                        ss = ss.groupby(level=inst_level).last()
                    except Exception:
                        keep = [inst_level]
                        drop_levels = [n for i, n in enumerate(ss.index.names) if i not in keep]
                        ss = ss.droplevel(drop_levels)
                for idx, val in ss.items():
                    inst = to_qlib_inst(idx)
                    try:
                        model_scores[inst] = float(val)
                    except Exception:
                        model_scores[inst] = 0.0
    except Exception:
        model_scores = {}

    # Load Chinese names mapping (if available) with robust key variants
    chinese_name: Dict[str, str] = {}
    try:
        name_paths = [
            os.path.join(os.path.expanduser("~"), ".qlib", "qlib_data", "hk_data", "boardlot", "chinese_name.txt"),
            r"C:\\Users\\kennethlao\\.qlib\\qlib_data\\hk_data\\boardlot\\chinese_name.txt",
        ]
        lines = []
        for p in name_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        lines = fh.readlines()
                except Exception:
                    try:
                        with open(p, "r", encoding="gb18030") as fh:
                            lines = fh.readlines()
                    except Exception:
                        lines = []
                break

        def add_name_keys(code: str, name: str):
            # generate multiple key variants for robust lookup
            c = code.strip()
            c_low = c.lower()
            # extract numeric part
            if c_low.startswith("hk."):
                num = c_low.split(".", 1)[1]
            elif c_low.endswith(".hk"):
                num = c_low.split(".", 1)[0]
            else:
                num = c_low
            if num.isdigit():
                num5 = num.zfill(5)
            else:
                num5 = num
            keys = set()
            keys.add(num5 + ".hk")
            keys.add(num5.upper() + ".HK")
            keys.add("hk." + num5)
            keys.add("HK." + num5)
            keys.add(num5)
            keys.add(num5.upper())
            for k in keys:
                chinese_name[k] = name

        for ln in lines:
            parts = ln.strip().split()
            if len(parts) >= 2:
                code = parts[0]
                name = " ".join(parts[1:])
                add_name_keys(code, name)
    except Exception:
        chinese_name = {}

    # helper: resolve chinese name from mapping with robust keys
    def resolve_chinese(inst_code: str) -> str:
        base = inst_code.split(".", 1)[0]
        candidates = [
            inst_code,
            inst_code.lower(),
            base,
            base.lower(),
            base.upper(),
            base.zfill(5),
            base.zfill(5).lower(),
            base.zfill(5).upper(),
            f"{base.zfill(5)}.hk",
            f"{base.zfill(5)}.HK",
            f"hk.{base.zfill(5)}",
            f"HK.{base.zfill(5)}",
        ]
        for k in candidates:
            if k in chinese_name:
                return chinese_name[k]
        return ""

    # data window
    start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=args.lookback)).strftime("%Y-%m-%d")
    ohlcv = fetch_ohlcv(selected, start_dt, target_day)

    # avg dollar volume gate (liq)
    try:
        base = D.features(selected, ["$close", "$volume"], start_time=start_dt, end_time=target_day, freq="day")
        base.columns = ["$close", "$volume"]
        def _tail_mean_dollar(df):
            df2 = df.dropna()
            if df2.empty:
                return 0.0
            dv = (df2["$close"] * df2["$volume"]).tail(args.liq_window)
            return float(dv.mean()) if len(dv) > 0 else 0.0
        avg_dollar = base.groupby(level="instrument").apply(_tail_mean_dollar).to_dict()
    except Exception:
        avg_dollar = {inst: 0.0 for inst in selected}

    market_close = None

    # ---- Market regime (HSI trend/vol and liquidity proxy) ----
    try:
        # HSI trend and volatility (ATR/MA)
        hsi_df = try_fetch_hsi(start_dt, target_day)
        trend_text = "N/A"
        vol_text = "N/A"
        liq_text = "N/A"
        trend_text_zh = "無法判定恆指趨勢"
        regime_text_zh = "無法判定市場結構"
        vol_text_zh = "無法判定波動率 regime"
        liq_text_zh = "無法判定流動性 regime"
        if isinstance(hsi_df, pd.DataFrame) and not hsi_df.empty:
            c_hsi = hsi_df["$close"].astype(float)
            market_close = c_hsi
            h_hsi = hsi_df["$high"].astype(float)
            l_hsi = hsi_df["$low"].astype(float)
            ma20_hsi = c_hsi.rolling(20).mean()
            ma60_hsi = c_hsi.rolling(60).mean()
            if pd.notna(ma20_hsi.iloc[-1]) and pd.notna(ma60_hsi.iloc[-1]):
                short_above = (ma20_hsi.iloc[-1] > ma60_hsi.iloc[-1])
                trend_text = f"HSI MA20>MA60: {bool(short_above)}"
                trend_text_zh = f"恆指短期趨勢（20 日）{'高於' if short_above else '低於'}中期趨勢（60 日）"
                regime_text_zh = ("市場處於 上行 / 強勢 / 牛市結構" if short_above else "市場處於 下行 / 弱勢 / 熊市結構")
            atr14 = atr(h_hsi, l_hsi, c_hsi, period=14)
            c_ma20 = c_hsi.rolling(20).mean()
            if pd.notna(atr14.iloc[-1]) and pd.notna(c_ma20.iloc[-1]) and c_ma20.iloc[-1] != 0:
                vol_ratio = atr14.iloc[-1] / c_ma20.iloc[-1]
                vol_text = f"HSI ATR14/MA20: {vol_ratio*100:.3f}"
                if vol_ratio < 0.02:
                    vol_text_zh = "波動率 regime：極低波動（盤整 / 假突破風險高）"
                elif vol_ratio < 0.04:
                    vol_text_zh = "波動率 regime：正常波動"
                else:
                    vol_text_zh = "波動率 regime：高波動（趨勢行情）"

        # Liquidity proxy from selected basket: MA20/MA60 of aggregate dollar volume
        try:
            if 'base' in locals() and isinstance(base, pd.DataFrame) and not base.empty:
                dv_series = (base["$close"] * base["$volume"]).groupby(level="datetime").sum().sort_index()
                ma20_dv = dv_series.rolling(20).mean()
                ma60_dv = dv_series.rolling(60).mean()
                if pd.notna(ma20_dv.iloc[-1]) and pd.notna(ma60_dv.iloc[-1]) and ma60_dv.iloc[-1] != 0:
                    liq_ratio = ma20_dv.iloc[-1] / ma60_dv.iloc[-1]
                    liq_text = f"Liquidity MA20/MA60: {liq_ratio:.3f}"
                    if liq_ratio >= 1.0:
                        liq_text_zh = "流動性 regime：改善中"
                    elif liq_ratio >= 0.9:
                        liq_text_zh = "流動性 regime：下降中（尚未進入危險區）"
                    else:
                        liq_text_zh = "流動性 regime：危險（資金枯竭風險）"
        except Exception:
            pass

        print("\nMarket regime / 市場結構:")
        # 1) 趨勢 regime
        print(f"1) {trend_text} -> {trend_text_zh}")
        print(f"   {regime_text_zh}（這是最重要的 regime 訊號之一）")
        # 2) 波動率 regime
        print(f"2) {vol_text} -> {vol_text_zh}")
        print("   <2% 低波動（盤整/假突破多），2-4% 正常，>4% 高波動（趨勢行情）")
        # 3) 流動性 regime
        print(f"3) {liq_text} -> {liq_text_zh}")
        print("   >1 改善，<1 惡化，<0.9 危險（小票易被砸，無量反彈風險高）")
    except Exception:
        # Don't fail the run if regime calc has issues
        pass

    params = {
        "ma_10": 10,
        "ma_20": 20,
        "ma_60": 60,
        "ma_120": 120,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_window": 14,
        "rsi_neutral_low": 40,
        "rsi_neutral_high": 60,
        "rsi_div_window": 30,
        "roc_window": 10,
        "mom_window": 10,
        "vol_ma_window": 20,
        "vol_surge_mult": 2.0,
        "atr_window": 14,
        "atr_ma_window": 20,
        "atr_squeeze_mult": 0.7,
        "kdj_window": 9,
        "kdj_overbought": 80,
        "kdj_oversold": 20,
        "wpr_window": 14,
        "wpr_overbought": -20,
        "wpr_oversold": -80,
        "st_period": 10,
        "st_multiplier": 3.0,
        "slope_window": 5,
        "lookback_min": 120,
        "bb_window": 20,
        "bb_mult": 2.0,
        "bb_squeeze_quantile": 0.1,
        "adx_window": 20,
        "cci_window": 20,
        "cci_overbought": 100,
        "cci_oversold": -100,
        "don_window": 20,
        "z_window": 20,
        "rs_window": 60,
        "market_close": market_close,
        "allow_partial": bool(getattr(args, "allow_partial", False)),
    }

    rows = []
    for inst in selected:
        try:
            df_inst = ohlcv.xs(inst, level="instrument")
            bull, bear, snap = score_instrument(df_inst, params)
            net = bull - bear
            liq_ok = avg_dollar.get(inst, 0.0) >= float(args.liq_threshold)
            buy_flag = (bull > bear) and liq_ok
            # determine listing age (number of available trading days with non-null close)
            try:
                hist = D.features([inst], ["$close"], start_time="2005-01-01", end_time=target_day, freq="day", disk_cache=True)
                # extract series for instrument if needed
                if isinstance(hist, pd.DataFrame):
                    if "instrument" in hist.index.names:
                        try:
                            s_close = hist.xs(inst, level="instrument")[hist.columns[0]]
                        except Exception:
                            s_close = hist.iloc[:, 0]
                    else:
                        s_close = hist.iloc[:, 0]
                    listed_days = int(s_close.dropna().shape[0])
                else:
                    listed_days = 0
            except Exception:
                listed_days = 0

            is_new_listing = bool(listed_days < 120)

            rows.append({
                "instrument": inst,
                "bull_score": bull,
                "bear_score": bear,
                "net_score": net,
                "avg_dollar_vol": int(round(avg_dollar.get(inst, 0.0))),
                "buy": bool(buy_flag),
                "model_score": float(model_scores.get(inst, 0.0)),
                "chinese_name": resolve_chinese(inst),
                **snap,
                "newly_listed_days": int(listed_days),
                "is_new_listing": bool(is_new_listing),
            })
        except Exception as e:
            rows.append({
                "instrument": inst,
                "error": str(e),
                "bull_score": 0,
                "bear_score": 1,
                "net_score": -1,
                "avg_dollar_vol": int(round(avg_dollar.get(inst, 0.0))),
                "buy": False,
            })

    out_df = pd.DataFrame(rows)
    # persist composite score to CSV as well
    try:
        out_df["composite_score"] = out_df["model_score"].fillna(0.0) * out_df["net_score"].fillna(0.0)
    except Exception:
        pass
    out_path = f"bull_bear_today_{args.recorder_id}.csv"
    # Use UTF-8 BOM to improve Excel Chinese display on Windows
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved decision table to {out_path}")

    # print concise table (sorted by composite = model_score * net_score)
    base_cols = ["instrument", "chinese_name", "model_score", "bull_score", "bear_score", "net_score", "avg_dollar_vol", "is_new_listing", "buy"]
    disp = out_df[base_cols].copy()
    disp["composite_score"] = disp["model_score"].fillna(0.0) * disp["net_score"].fillna(0.0)
    # Improve East Asian alignment in console
    try:
        pd.set_option("display.unicode.east_asian_width", True)
        pd.set_option("display.colheader_justify", "center")
    except Exception:
        pass
    # sort: buy first, then composite desc, then model_score desc
    disp_sorted = disp.sort_values(["buy", "composite_score", "model_score"], ascending=[False, False, False])
    # format a view for pretty printing without touching CSV types
    view = disp_sorted.head(args.topk).copy()
    view["model_score"] = view["model_score"].map(lambda x: f"{x:.6f}")
    view["composite_score"] = view["composite_score"].map(lambda x: f"{x:.6f}")
    view["avg_dollar_vol"] = view["avg_dollar_vol"].map(lambda x: f"{int(x):,}")
    print("\nDecision preview (sorted by composite = model_score * net_score):")
    # reorder columns for display
    disp_cols = ["instrument", "chinese_name", "model_score", "bull_score", "bear_score", "net_score", "composite_score", "avg_dollar_vol", "is_new_listing", "buy"]
    try:
        # ensure is_new_listing shown as True/False string
        if "is_new_listing" in view.columns:
            view["is_new_listing"] = view["is_new_listing"].map(lambda x: "True" if bool(x) else "False")
        print(view[disp_cols].to_string(index=False))
    except Exception:
        print(view[disp_cols])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder_id", default="9e51767a8a1842f7889118799b99d6a0", help="recorder/run id")
    parser.add_argument("--experiment_name", default="workflow", help="experiment name (unused here)")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/hk_data", help="qlib data dir")
    parser.add_argument("--topk", type=int, default=20, help="number of instruments to score")
    parser.add_argument("--lookback", type=int, default=180, help="lookback days for indicators")
    parser.add_argument("--liq_threshold", type=float, default=1000000.0, help="avg dollar vol gate")
    parser.add_argument("--liq_window", type=int, default=20, help="window for avg dollar vol")
    parser.add_argument("--allow_partial", action="store_true", help="allow partial scoring for insufficient-history instruments (skip missing indicators instead of penalizing)")
    args = parser.parse_args()
    main(args)
