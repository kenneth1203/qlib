"""
Indicator computations and scoring utilities shared by decision scripts.
Extracted from bull_bear_decision_v2 to keep the main script lean.
"""
from typing import Tuple, Dict

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    # Explicitly disable NA padding to avoid FutureWarning from pandas
    return close.pct_change(periods=period, fill_method=None)


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
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


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(period).mean()


def rsv(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 9) -> pd.Series:
    ll = low.rolling(period).min()
    hh = high.rolling(period).max()
    denom = (hh - ll).replace(0, np.nan)
    return (close - ll) / denom * 100


def kdj(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    rsv_s = rsv(close, high, low, period).fillna(50)
    k = rsv_s.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def williams_r(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    denom = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / denom


def supertrend(close: pd.Series, high: pd.Series, low: pd.Series, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    atr_s = atr(high, low, close, period)
    basic_ub = ((high + low) / 2) + multiplier * atr_s
    basic_lb = ((high + low) / 2) - multiplier * atr_s
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    for i in range(1, len(close)):
        final_ub.iloc[i] = basic_ub.iloc[i] if (basic_ub.iloc[i] < final_ub.iloc[i - 1]) or (close.iloc[i - 1] > final_ub.iloc[i - 1]) else final_ub.iloc[i - 1]
        final_lb.iloc[i] = basic_lb.iloc[i] if (basic_lb.iloc[i] > final_lb.iloc[i - 1]) or (close.iloc[i - 1] < final_lb.iloc[i - 1]) else final_lb.iloc[i - 1]
    st = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)
    st.iloc[0] = final_ub.iloc[0]
    direction.iloc[0] = -1
    for i in range(1, len(close)):
        if st.iloc[i - 1] == final_ub.iloc[i - 1]:
            st.iloc[i] = final_ub.iloc[i] if close.iloc[i] <= final_ub.iloc[i] else final_lb.iloc[i]
        else:
            st.iloc[i] = final_lb.iloc[i] if close.iloc[i] >= final_lb.iloc[i] else final_ub.iloc[i]
        direction.iloc[i] = 1 if close.iloc[i] >= st.iloc[i] else -1
    return st, direction


def bollinger_bands(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, width


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_w.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period).mean()
    mad = (tp - sma).abs().rolling(period).mean()
    cci_val = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    return cci_val


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
    up = high.rolling(period).max()
    dn = low.rolling(period).min()
    return up, dn


def rsi_divergence(close: pd.Series, rsi_s: pd.Series, window: int = 30) -> Tuple[bool, bool]:
    if len(close) < window + 1:
        return False, False
    c_win = close.iloc[-window:]
    r_win = rsi_s.iloc[-window:]
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

    Returns
    -------
    bull_score, bear_score, snapshot dict (last values)
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
    k_val, d_val, j_val = kdj(c, h, l, params["kdj_window"])
    wpr_val = williams_r(c, h, l, params["wpr_window"])
    st_val, st_dir = supertrend(c, h, l, params["st_period"], params["st_multiplier"])
    mid, bb_up, bb_dn, bb_width = bollinger_bands(c, params["bb_window"], params["bb_mult"])
    adx_val = adx(h, l, c, params["adx_window"])
    cci_val = cci(h, l, c, params["cci_window"])
    don_up, don_dn = donchian(h, l, params["don_window"])
    zscore = (c - ma20) / (c.rolling(params["z_window"]).std(ddof=0).replace(0, np.nan))
    rsi_bull_div, rsi_bear_div = rsi_divergence(c, rsi_val, window=params["rsi_div_window"])

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

    add_bull((rsi_val.iloc[-1] > 50), "RSI>50")
    add_bear((rsi_val.iloc[-1] > 70), "RSI>70_overbought")
    add_bull((rsi_val.iloc[-1] < 30), "RSI<30_oversold")
    add_bull(rsi_bull_div, "RSI_bull_div")
    add_bear(rsi_bear_div, "RSI_bear_div")

    add_bull((roc_val.iloc[-1] > 0), "ROC>0")
    add_bull((mom_val.iloc[-1] > 0), "Momentum>0")

    if pd.notna(rs_val):
        add_bull(rs_val > 0, "RS>0")
        add_bear(rs_val < 0, "RS<0")
    elif not allow_partial:
        add_bear(True, "RS_missing")

    if pd.notna(v.iloc[-1]) and pd.notna(vol_ma.iloc[-1]):
        add_bull((v.iloc[-1] > vol_ma.iloc[-1]), "Vol>VolMA")
        add_bull((v.iloc[-1] > params["vol_surge_mult"] * vol_ma.iloc[-1]), "Vol_surge")
    elif not allow_partial:
        add_bear(True, "Volume_missing")
    if pd.notna(c.iloc[-1]) and pd.notna(vwap.iloc[-1]):
        add_bull((c.iloc[-1] > vwap.iloc[-1]), "Close>VWAP")
        add_bear((c.iloc[-1] <= vwap.iloc[-1]), "Close<=VWAP")
    elif not allow_partial:
        add_bear(True, "VWAP_missing")
    add_bull((obv_slope is not None and obv_slope > 0), "OBV_up")

    add_bull((atr_change is not None and atr_change > 0), "ATR_rising")
    add_bear((atr_change is not None and atr_change < 0), "ATR_falling")
    add_bear(pd.notna(atr_s.iloc[-1]) and pd.notna(atr_ma.iloc[-1]) and (atr_s.iloc[-1] < params["atr_squeeze_mult"] * atr_ma.iloc[-1]), "ATR_squeeze")

    add_bear((k_val.iloc[-1] > 80), "K>80_overbought")
    add_bull((k_val.iloc[-1] < 20), "K<20_oversold")
    add_bear((wpr_val.iloc[-1] > -20), "W%R>-20_overbought")
    add_bull((wpr_val.iloc[-1] < -80), "W%R<-80_oversold")

    if pd.notna(st_dir.iloc[-1]):
        add_bull((st_dir.iloc[-1] == 1), "Supertrend_up")
        add_bear((st_dir.iloc[-1] != 1), "Supertrend_down")
    elif not allow_partial:
        add_bear(True, "Supertrend_missing")

    try:
        bw_series = bb_width.dropna()
        if len(bw_series) >= params["bb_window"]:
            thresh = bw_series.quantile(params["bb_squeeze_quantile"])
            if bb_width.iloc[-1] <= thresh:
                add_bear(True, "BB_squeeze")
    except Exception:
        pass

    if pd.notna(adx_val.iloc[-1]):
        add_bull(adx_val.iloc[-1] >= 25, "ADX_strong_trend")
        add_bear(adx_val.iloc[-1] <= 20, "ADX_weak_trend")

    if pd.notna(cci_val.iloc[-1]):
        add_bull(cci_val.iloc[-1] > 100, "CCI>100")
        add_bear(cci_val.iloc[-1] < -100, "CCI<-100")

    if len(don_up) >= 2 and len(don_dn) >= 2 and pd.notna(don_up.iloc[-2]) and pd.notna(don_dn.iloc[-2]):
        add_bull(c.iloc[-1] > don_up.iloc[-2], "Donchian_breakout_up")
        add_bear(c.iloc[-1] < don_dn.iloc[-2], "Donchian_breakdown")

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
        "kdj_k": float(k_val.iloc[-1]),
        "wpr": float(wpr_val.iloc[-1]),
        "supertrend_dir": int(st_dir.iloc[-1]),
        "bb_width": float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else 0.0,
        "adx": float(adx_val.iloc[-1]) if pd.notna(adx_val.iloc[-1]) else 0.0,
        "cci": float(cci_val.iloc[-1]) if pd.notna(cci_val.iloc[-1]) else 0.0,
        "don_up": float(don_up.iloc[-2]) if pd.notna(don_up.iloc[-2]) else 0.0,
        "don_dn": float(don_dn.iloc[-2]) if pd.notna(don_dn.iloc[-2]) else 0.0,
        "zscore": float(zscore.iloc[-1]) if pd.notna(zscore.iloc[-1]) else 0.0,
        "vwap": float(vwap.iloc[-1]) if pd.notna(vwap.iloc[-1]) else float("nan"),
        "rsi_bull_div": bool(rsi_bull_div),
        "rsi_bear_div": bool(rsi_bear_div),
        "rs": float(rs_val) if pd.notna(rs_val) else float("nan"),
        "bull_sources": "|".join(bull_sources),
        "bear_sources": "|".join(bear_sources),
    }
    return bull, bear, snap


__all__ = [
    "ema",
    "macd",
    "rsi",
    "roc",
    "momentum",
    "obv",
    "true_range",
    "atr",
    "rsv",
    "kdj",
    "williams_r",
    "supertrend",
    "bollinger_bands",
    "adx",
    "cci",
    "donchian",
    "rsi_divergence",
    "score_instrument",
]
