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
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_HK
from qlib.workflow import R
from qlib.data import D
from qlib.utils.indicators import score_instrument, atr
from qlib.utils.func import (
    calendar_last_day,
    compute_avg_dollar_volume,
    fetch_base_close_vol,
    fetch_ohlcv,
    next_trading_day_from_future,
    resolve_chinese,
    to_qlib_inst,
    load_chinese_name_map,
)
# Ensure stdout is UTF-8 on Windows to avoid GBK-related mojibake when printing Chinese
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib.pyplot as plt
from qlib.utils.notify import TelegramNotifier, resolve_notify_params
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None

# ---- Kronos imports (optional) ----
_KRONOS_AVAILABLE = True
try:
    # Try importing from installed package first
    from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
except Exception:
    # Fallback: add local Kronos repo to path (../../.. -> repo root; then Kronos)
    try:
        _here = os.path.dirname(__file__)
        _root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
        _kronos_dir = os.path.join(_root, "Kronos")
        if os.path.isdir(_kronos_dir) and _kronos_dir not in sys.path:
            sys.path.append(_kronos_dir)
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
    except Exception:
        _KRONOS_AVAILABLE = False

def _dated_pred_path(recorder_id: str, target_day: str) -> Path:
    day_str = pd.to_datetime(target_day).strftime("%Y%m%d")
    return Path(f"pred_{day_str}_{recorder_id}.pkl")


def _list_dated_pred_paths(recorder_id: str, target_day: str, lookback_days: int) -> List[Path]:
    paths: List[Path] = []
    target_ts = pd.to_datetime(target_day)
    start_ts = target_ts - pd.Timedelta(days=lookback_days * 2)
    pat = f"pred_*_{recorder_id}.pkl"
    for p in Path('.').glob(pat):
        name = p.name
        m = re.match(r"pred_(\d{8})_" + re.escape(recorder_id) + r"\.pkl", name)
        if not m:
            continue
        try:
            dt = pd.to_datetime(m.group(1))
        except Exception:
            continue
        if dt <= target_ts and dt >= start_ts:
            paths.append(p)
    paths_sorted = sorted(paths, key=lambda x: x.name)
    return paths_sorted


def load_selection(recorder_id: str, topk: int, target_day: str) -> List[str]:
    """Load selected instruments for target_day.
    Prefer CSV next to working dir; fallback to pred pickle reconstruction.
    """
    day_str = pd.to_datetime(target_day).strftime("%Y%m%d")
    csv_path = f"selection_{day_str}_{recorder_id}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c.lower().startswith("id")]
        if len(cols) > 0:
            return [to_qlib_inst(i) for i in df[cols[0]].astype(str).tolist()][:topk]

    # fallback: load prediction pickle
    dated_path = _dated_pred_path(recorder_id, target_day)
    pkl_path = dated_path
    print(f"load_selection using pred file: {pkl_path}")
    if not pkl_path.exists():
        raise RuntimeError(f"Missing selection CSV and prediction pickle: {csv_path}, {pkl_path}")

    pred = pd.read_pickle(pkl_path)
    # If T1 dumped a list of selected instruments, use it directly
    if isinstance(pred, list):
        return [to_qlib_inst(i) for i in pred][:topk]
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


def load_model_scores(recorder_id: str, target_day: str) -> Dict[str, float]:
    """Load per-instrument model scores from pred_today_<recorder_id>.pkl."""
    model_scores: Dict[str, float] = {}
    try:
        dated_path = _dated_pred_path(recorder_id, target_day)
        pkl_path = dated_path
        if not pkl_path.exists():
            return model_scores
        print(f"load_model_scores using pred file: {pkl_path}")

        pred = pd.read_pickle(pkl_path)
        s = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
        if not isinstance(s, pd.Series):
            return model_scores

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
    return model_scores


def load_selection_history(recorder_id: str, topk: int, target_day: str, lookback_days: int = 30) -> Dict[str, List[str]]:
    """Load per-day selections from pred_today_<recorder_id>.pkl for streak counting."""
    history: Dict[str, List[str]] = {}
    try:
        paths = _list_dated_pred_paths(recorder_id, target_day, lookback_days)
        if not paths:
            return history

        for pkl_path in paths:
            print(f"load_selection_history reading: {pkl_path}")
            pred = pd.read_pickle(pkl_path)
            s = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
            if not isinstance(s, pd.Series):
                continue

            names = list(getattr(s.index, "names", []))
            if "datetime" not in names:
                continue
            dt_level = names.index("datetime")
            inst_level = names.index("instrument") if "instrument" in names else None

            grouped = s.groupby(level=dt_level)
            for dt_val, series in grouped:
                ss = series
                if isinstance(ss.index, pd.MultiIndex) and inst_level is not None and len(ss.index.names) > 1:
                    try:
                        ss = ss.groupby(level=inst_level).last()
                    except Exception:
                        keep = [inst_level]
                        drop_levels = [n for i, n in enumerate(ss.index.names) if i not in keep]
                        ss = ss.droplevel(drop_levels)
                top = ss.sort_values(ascending=False).head(max(topk, 500))
                day_str = pd.Timestamp(dt_val).strftime("%Y-%m-%d")
                history[day_str] = [to_qlib_inst(i) for i in top.index][:topk]

        try:
            target_ts = pd.to_datetime(target_day)
            cutoff = target_ts - pd.Timedelta(days=lookback_days * 2)
            history = {d: v for d, v in history.items() if pd.to_datetime(d) >= cutoff and pd.to_datetime(d) <= target_ts}
        except Exception:
            pass
    except Exception:
        history = {}
    return history


def compute_consecutive_selected(selected_today: List[str], selection_history: Dict[str, List[str]], target_day: str, lookback_days: int = 30) -> Dict[str, int]:
    """Return consecutive-day selection streaks ending at target_day (trading days only)."""
    counts: Dict[str, int] = {inst: 1 for inst in selected_today}
    try:
        target_ts = pd.to_datetime(target_day)
        start_ts = target_ts - pd.Timedelta(days=lookback_days)
        cal = D.calendar(start_time=start_ts.strftime("%Y-%m-%d"), end_time=target_ts.strftime("%Y-%m-%d"), freq="day")
        cal_ts = [pd.Timestamp(d) for d in cal if pd.Timestamp(d) <= target_ts]
    except Exception:
        cal_ts = [pd.to_datetime(target_day)]

    day_str = pd.to_datetime(target_day).strftime("%Y-%m-%d")
    history = dict(selection_history)
    if day_str not in history:
        history[day_str] = list(selected_today)

    for inst in selected_today:
        streak = 0
        for dt_val in reversed(cal_ts):
            dstr = pd.Timestamp(dt_val).strftime("%Y-%m-%d")
            picks = history.get(dstr)
            if picks is None:
                if dstr == day_str:
                    streak += 1
                else:
                    break
            elif inst in picks:
                streak += 1
            else:
                break
        counts[inst] = streak

    return counts

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

def compute_market_regime(base: pd.DataFrame, start_dt: str, target_day: str):
    """Return dict with regime texts and metrics; never raises."""
    result = {
        "trend_text": "N/A",
        "trend_text_zh": "無法判定恆指趨勢",
        "regime_text_zh": "無法判定市場結構",
        "vol_text": "N/A",
        "vol_text_zh": "無法判定波動率 regime",
        "liq_text": "N/A",
        "liq_text_zh": "無法判定流動性 regime",
        "vol_ratio": None,
        "market_close": None,
    }

    try:
        hsi_df = try_fetch_hsi(start_dt, target_day)
        if isinstance(hsi_df, pd.DataFrame) and not hsi_df.empty:
            c_hsi = hsi_df["$close"].astype(float)
            result["market_close"] = c_hsi
            h_hsi = hsi_df["$high"].astype(float)
            l_hsi = hsi_df["$low"].astype(float)
            ma20_hsi = c_hsi.rolling(20).mean()
            ma60_hsi = c_hsi.rolling(60).mean()
            if pd.notna(ma20_hsi.iloc[-1]) and pd.notna(ma60_hsi.iloc[-1]):
                short_above = (ma20_hsi.iloc[-1] > ma60_hsi.iloc[-1])
                result["trend_text"] = f"HSI MA20>MA60: {bool(short_above)}"
                result["trend_text_zh"] = f"恆指短期趨勢（20 日）{'高於' if short_above else '低於'}中期趨勢（60 日）"
                result["regime_text_zh"] = ("市場處於 上行 / 強勢 / 牛市結構" if short_above else "市場處於 下行 / 弱勢 / 熊市結構")
            atr14 = atr(h_hsi, l_hsi, c_hsi, period=14)
            c_ma20 = c_hsi.rolling(20).mean()
            if pd.notna(atr14.iloc[-1]) and pd.notna(c_ma20.iloc[-1]) and c_ma20.iloc[-1] != 0:
                vol_ratio = atr14.iloc[-1] / c_ma20.iloc[-1]
                result["vol_ratio"] = float(vol_ratio)
                result["vol_text"] = f"HSI ATR14/MA20: {vol_ratio*100:.3f}"
                if vol_ratio < 0.02:
                    result["vol_text_zh"] = "波動率 regime：極低波動（盤整 / 假突破風險高）"
                elif vol_ratio < 0.04:
                    result["vol_text_zh"] = "波動率 regime：正常波動"
                else:
                    result["vol_text_zh"] = "波動率 regime：高波動（趨勢行情）"

        if isinstance(base, pd.DataFrame) and not base.empty:
            dv_series = (base["$close"] * base["$volume"]).groupby(level="datetime").sum().sort_index()
            ma20_dv = dv_series.rolling(20).mean()
            ma60_dv = dv_series.rolling(60).mean()
            if pd.notna(ma20_dv.iloc[-1]) and pd.notna(ma60_dv.iloc[-1]) and ma60_dv.iloc[-1] != 0:
                liq_ratio = ma20_dv.iloc[-1] / ma60_dv.iloc[-1]
                result["liq_text"] = f"Liquidity MA20/MA60: {liq_ratio:.3f}"
                if liq_ratio >= 1.0:
                    result["liq_text_zh"] = "流動性 regime：改善中"
                elif liq_ratio >= 0.9:
                    result["liq_text_zh"] = "流動性 regime：下降中（尚未進入危險區）"
                else:
                    result["liq_text_zh"] = "流動性 regime：危險（資金枯竭風險）"
    except Exception:
        pass
    return result


def setup_kronos(args):
    if not args.enable_kronos:
        return None
    if not _KRONOS_AVAILABLE:
        print("Warning: Kronos modules not found; proceeding without Kronos scoring.")
        return None
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        kronos_model_id = args.kronos_model if args.kronos_model else "NeoQuasar/Kronos-small"
        kronos_model = Kronos.from_pretrained(kronos_model_id)
        predictor = KronosPredictor(kronos_model, tokenizer, device=args.kronos_device, max_context=args.kronos_max_context)
        print(f"Kronos loaded: model={kronos_model_id}, device={args.kronos_device}")
        return predictor
    except Exception as e:
        print(f"Warning: Kronos unavailable ({e}); proceeding without Kronos scoring.")
        return None


def make_kronos_scorer(predictor, args):
    if predictor is None:
        return lambda inst_code, df_inst: float("nan")

    def _score(inst_code: str, df_inst: pd.DataFrame) -> float:
        if predictor is None:
            return float("nan")
        try:
            df_inst_sorted = df_inst.sort_index()
            min_len = int(getattr(args, "kronos_min_bars", 100))
            if df_inst_sorted.shape[0] < min_len:
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos skip {inst_code}: insufficient history {df_inst_sorted.shape[0]}< {min_len}")
                return float("nan")

            cols_avail = set(df_inst_sorted.columns)
            open_s = (df_inst_sorted["$open"] if "$open" in cols_avail else df_inst_sorted["$close"]).astype(float)
            high_s = df_inst_sorted["$high"].astype(float)
            low_s = df_inst_sorted["$low"].astype(float)
            close_s = df_inst_sorted["$close"].astype(float)
            volume_s = df_inst_sorted["$volume"].astype(float)
            amount_s = None
            if "$amount" in cols_avail:
                amt = df_inst_sorted["$amount"].astype(float)
                amount_s = None if amt.isna().all() else amt
            if amount_s is None:
                amount_s = (close_s * volume_s).astype(float)

            kdf = pd.DataFrame({
                "open": open_s,
                "high": high_s,
                "low": low_s,
                "close": close_s,
                "volume": volume_s,
                "amount": amount_s,
            }).replace([np.inf, -np.inf], np.nan)

            if getattr(args, "kronos_fill_missing", False):
                core_cols = ["open", "high", "low", "close", "volume", "amount"]
                before = int(kdf[core_cols].isna().sum().sum())
                kdf[core_cols] = kdf[core_cols].ffill().bfill()
                after = int(kdf[core_cols].isna().sum().sum())
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos fill_missing {inst_code}: gaps {before} -> {after}")

            kdf_clean = kdf.dropna(subset=["open", "high", "low", "close", "volume"]).sort_index()
            if kdf_clean.empty:
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos skip {inst_code}: no valid rows after cleaning")
                return float("nan")

            look = min(int(args.kronos_lookback), len(kdf_clean))
            if getattr(args, "kronos_debug", False):
                print(f"Kronos processing {inst_code}: using lookback={look} from available {len(kdf_clean)} rows")
            x_df = kdf_clean.tail(look)
            x_timestamp = pd.Series(pd.to_datetime(x_df.index))
            last_dt = pd.to_datetime(kdf.index[-1]).to_pydatetime().date()
            cal_next = D.calendar(
                start_time=(last_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                end_time=(last_dt + datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                freq="day",
            )
            if len(cal_next) > 0:
                y_timestamp = pd.Series(pd.to_datetime(cal_next[: args.kronos_pred_len]))
            else:
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos calendar fallback for {inst_code}: no future trading days from {last_dt}; using pandas bdate_range")
                start_pd = pd.Timestamp(last_dt) + pd.Timedelta(days=1)
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos predict start date for {inst_code}: {start_pd.strftime('%Y-%m-%d')} for {int(args.kronos_pred_len)} days")
                y_timestamp = pd.Series(pd.bdate_range(start=start_pd, periods=int(args.kronos_pred_len), freq="B"))
            if len(y_timestamp) < int(args.kronos_pred_len):
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos warn {inst_code}: y_timestamp shorter than pred_len {len(y_timestamp)}<{args.kronos_pred_len}")

            pred = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=args.kronos_pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=20,
                verbose=False,
            )

            # Debug plotting is kept but muted to avoid blocking runs
            if getattr(args, "kronos_debug", False):
                try:
                    hist_df = kdf_clean.tail(look)[["close", "volume"]].copy()
                    hist_df.index = pd.to_datetime(hist_df.index)
                    pred_df = pred.copy() if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred)
                    try:
                        pred_df.index = pd.to_datetime(pred_df.index)
                    except Exception:
                        try:
                            pred_df.index = pd.to_datetime(y_timestamp)
                        except Exception:
                            pred_df.index = pd.RangeIndex(start=0, stop=len(pred_df))
                    pred_close_col = None
                    for cname in pred_df.columns:
                        if str(cname).lower() == "close" or str(cname).lower().endswith("close"):
                            pred_close_col = cname
                            break
                    if pred_close_col is None:
                        for cname in pred_df.columns:
                            if "price" in str(cname).lower():
                                pred_close_col = cname
                                break
                    if pred_close_col is not None:
                        plot_close = pred_df[pred_close_col].astype(float)
                        plot_vol = pred_df["volume"].astype(float) if "volume" in pred_df.columns else pd.Series(0.0, index=plot_close.index)
                        _ = pd.DataFrame({"close": plot_close, "volume": plot_vol}, index=plot_close.index)
                except Exception as e:
                    print(f"Kronos debug plot failure for {inst_code}: {e}")

            if isinstance(pred, pd.DataFrame) and not pred.empty:
                last_close = float(kdf["close"].iloc[-1])
                col_close = None
                for cname in pred.columns:
                    lc = str(cname).lower()
                    if lc == "close" or lc.endswith("close"):
                        col_close = cname
                        break
                if col_close is None:
                    if getattr(args, "kronos_debug", False):
                        print(f"Kronos skip {inst_code}: no 'close' in prediction columns={list(pred.columns)}")
                    return float("nan")
                pred_close_series = pred[col_close].astype(float)
                if last_close == 0:
                    r = 0.0
                else:
                    returns = (pred_close_series / last_close) - 1.0
                    # Trim extreme paths, then average to avoid所有樣本都被頂到同一個截斷值
                    trimmed = returns.clip(lower=-0.1, upper=0.1)
                    r = float(trimmed.mean())
                if getattr(args, "kronos_debug", False):
                    try:
                        dates = y_timestamp.dt.strftime("%Y-%m-%d").tolist()
                    except Exception:
                        dates = [str(x) for x in y_timestamp.tolist()]
                    print(f"Kronos pred {inst_code}: horizon_days={len(returns)}, dates={dates[:min(5,len(dates))]}..., last_close={last_close:.4f}, avg_pred_close={pred_close_series.mean():.4f}, median_ret_clip={r:.4%}")
                alpha = float(getattr(args, "kronos_sigmoid_alpha", 5.0))
                s = 1.0 / (1.0 + np.exp(-alpha * r))
                return float(s)
            else:
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos skip {inst_code}: empty/invalid pred type={type(pred)}")
        except Exception as e:
            if getattr(args, "kronos_debug", False):
                print(f"Kronos error {inst_code}: {e}")
            return float("nan")
        return float("nan")

    return _score


def compute_final_score(model_score, kronos_score, net_score, vol_ratio):
    kronos_val = 0.0 if pd.isna(kronos_score) else float(kronos_score)
    kronos_c = float(np.clip(kronos_val, 0.00, 0.95))

    net_val = 0.0 if pd.isna(net_score) else float(net_score)
    net_c = 1.0 / (1.0 + np.exp(-net_val / 3.0))

    try:
        v = 0.0 if vol_ratio is None or pd.isna(vol_ratio) else float(vol_ratio)
    except Exception:
        v = 0.0

    if v > 0.04:
        w_model, w_net = 0.6, 0.2
    else:
        w_model, w_net = 0.5, 0.25

    ms = 0.0 if pd.isna(model_score) else float(model_score)

    base = w_model * ms + w_net * net_c
    final = base * kronos_c * 100
    return float(final)


def buy_filter(row, liq_threshold: float, final_threshold: float):
    try:
        ks = row.get("kronos_score", np.nan)
        if pd.notna(ks) and float(ks) < 0.35:
            return False
    except Exception:
        return False

    try:
        if float(row.get("avg_dollar_vol", 0.0)) < float(liq_threshold):
            return False
    except Exception:
        return False

    try:
        bull = float(row.get("bull_score", 0.0))
        bear = float(row.get("bear_score", 0.0))
        if bull <= bear and (bear - bull) > 3.0:
            return False
    except Exception:
        return False

    try:
        if float(row.get("final_score", 0.0)) < float(final_threshold):
            return False
    except Exception:
        return False

    return True

def main(args, notifier: Optional[TelegramNotifier] = None):
    provider_uri = os.path.expanduser(args.provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_HK)

    today_dt = datetime.date.today()
    target_day = calendar_last_day(today_dt)
    print("target_day:", target_day)

    msg_day = next_trading_day_from_future(provider_uri, target_day) or target_day

    selected = load_selection(args.recorder_id, args.topk, target_day)
    if len(selected) == 0:
        raise RuntimeError("No selected instruments.")
    print("Selected:", selected)

    model_scores = load_model_scores(args.recorder_id, target_day)
    selection_history = load_selection_history(args.recorder_id, args.topk, target_day, lookback_days=30)
    chinese_map = load_chinese_name_map()

    start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=args.lookback)).strftime("%Y-%m-%d")
    ohlcv = fetch_ohlcv(selected, start_dt, target_day)
    dates = ohlcv.index.get_level_values("datetime").unique()
    print(f"實際交易日數: {len(dates)}, 從 {dates.min()} 到 {dates.max()}")

    base = fetch_base_close_vol(selected, start_dt, target_day)
    avg_dollar = compute_avg_dollar_volume(base, selected, args.liq_window)

    regime = compute_market_regime(base, start_dt, target_day)
    try:
        print("\nMarket regime / 市場結構:")
        print(f"1) {regime['trend_text']} -> {regime['trend_text_zh']}")
        print(f"   {regime['regime_text_zh']}（這是最重要的 regime 訊號之一）")
        print(f"2) {regime['vol_text']} -> {regime['vol_text_zh']}")
        print("   <2% 低波動（盤整/假突破多），2-4% 正常，>4% 高波動（趨勢行情）")
        print(f"3) {regime['liq_text']} -> {regime['liq_text_zh']}")
        print("   >1 改善，<1 惡化，<0.9 危險（小票易被砸，無量反彈風險高）")
    except Exception:
        pass

    kronos_predictor = setup_kronos(args)
    kronos_score_fn = make_kronos_scorer(kronos_predictor, args)

    params = {
        "ma_10": 10,
        "ma_20": 20,
        "ma_60": 60,
        "ma_120": 120,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "rsi_window": 14,
        "rsi_div_window": 30,
        "roc_window": 10,
        "mom_window": 10,
        "vol_ma_window": 20,
        "vol_surge_mult": 2.0,
        "atr_window": 14,
        "atr_ma_window": 20,
        "atr_squeeze_mult": 0.7,
        "kdj_window": 9,
        "wpr_window": 14,
        "st_period": 10,
        "st_multiplier": 3.0,
        "slope_window": 5,
        "lookback_min": 120,
        "bb_window": 20,
        "bb_mult": 2.0,
        "bb_squeeze_quantile": 0.1,
        "adx_window": 20,
        "cci_window": 20,
        "don_window": 20,
        "z_window": 20,
        "rs_window": 60,
        "market_close": regime.get("market_close"),
        "allow_partial": bool(getattr(args, "allow_partial", False)),
    }

    consecutive_counts = compute_consecutive_selected(selected, selection_history, target_day, lookback_days=30)
    rows = []
    iter_inst = selected
    use_kronos_bar = (_tqdm is not None) and (kronos_predictor is not None)
    if use_kronos_bar:
        iter_inst = _tqdm(selected, desc="Kronos scoring", leave=False)
    for inst in iter_inst:
        try:
            df_inst = ohlcv.xs(inst, level="instrument")
            bull, bear, snap = score_instrument(df_inst, params)
            net = bull - bear
            liq_ok = avg_dollar.get(inst, 0.0) >= float(args.liq_threshold)
            buy_flag = (bull > bear) and liq_ok
            try:
                hist = D.features([inst], ["$close"], start_time="2005-01-01", end_time=target_day, freq="day", disk_cache=True)
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
            kronos_score = kronos_score_fn(inst, df_inst)
            rows.append({
                "instrument": inst,
                "bull_score": bull,
                "bear_score": bear,
                "net_score": net,
                "avg_dollar_vol": int(round(avg_dollar.get(inst, 0.0))),
                "buy": bool(buy_flag),
                "model_score": float(model_scores.get(inst, 0.0)),
                "kronos_score": float(kronos_score) if pd.notna(kronos_score) else float("nan"),
                "chinese_name": resolve_chinese(inst, chinese_map),
                **snap,
                "newly_listed_days": int(listed_days),
                "is_new_listing": bool(is_new_listing),
                "streak_days": int(consecutive_counts.get(inst, 1)),
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
                "streak_days": 0,
            })

    out_df = pd.DataFrame(rows)
    try:
        out_df["kronos_score"] = out_df.get("kronos_score", np.nan)
        market_vol_ratio = regime.get("vol_ratio", 0.0)
        out_df["final_score"] = out_df.apply(
            lambda r: compute_final_score(
                r.get("model_score", 0.0),
                r.get("kronos_score", 0.0),
                r.get("net_score", 0.0),
                market_vol_ratio,
            ),
            axis=1,
        )
        out_df["net_c"] = 1.0 / (1.0 + np.exp(-out_df["net_score"].fillna(0.0) / 3.0))
        out_df["composite_score"] = out_df["model_score"].fillna(0.0) * out_df["net_score"].fillna(0.0)
    except Exception:
        pass

    if "final_score" not in out_df.columns:
        out_df["final_score"] = 0.0
    if "net_c" not in out_df.columns:
        out_df["net_c"] = np.nan

    for col, default in {
        "model_score": 0.0,
        "kronos_score": np.nan,
        "bull_score": np.nan,
        "bear_score": np.nan,
        "avg_dollar_vol": 0.0,
        "is_new_listing": False,
        "buy": False,
        "streak_days": 0,
    }.items():
        if col not in out_df.columns:
            out_df[col] = default

    try:
        out_df["buy"] = out_df.apply(lambda r: bool(buy_filter(r, args.liq_threshold, args.final_threshold)), axis=1)
    except Exception:
        pass

    preferred = [
        "instrument",
        "chinese_name",
        "model_score",
        "kronos_score",
        "bull_score",
        "bear_score",
        "net_c",
        "final_score",
        "streak_days",
        "avg_dollar_vol",
        "is_new_listing",
        "buy",
    ]
    remaining = [c for c in out_df.columns if c not in preferred]

    # Robust date string (handles Timestamp with time component)
    day_str = pd.to_datetime(target_day).strftime("%Y%m%d")
    out_path = f"decision_{day_str}_{args.recorder_id}.csv"
    # Sort by final_score (then model_score) before persisting for easier scanning
    sorted_out = out_df.sort_values(["final_score", "model_score"], ascending=[False, False])
    sorted_out[preferred + remaining].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved decision table to {out_path}")

    base_cols = ["instrument", "chinese_name", "model_score", "kronos_score", "bull_score", "bear_score", "net_c", "final_score", "streak_days", "avg_dollar_vol", "is_new_listing", "buy"]
    disp = out_df[base_cols].copy()
    try:
        pd.set_option("display.unicode.east_asian_width", True)
        pd.set_option("display.colheader_justify", "center")
    except Exception:
        pass
    disp_sorted = disp.sort_values(["buy", "final_score", "model_score"], ascending=[False, False, False])
    view = disp_sorted.head(args.topk).copy()
    view["model_score"] = view["model_score"].map(lambda x: f"{x:.6f}")
    if "kronos_score" in view.columns:
        view["kronos_score"] = view["kronos_score"].map(lambda x: ("" if pd.isna(x) else f"{x:.6f}"))
    if "net_c" in view.columns:
        view["net_c"] = view["net_c"].map(lambda x: ("" if pd.isna(x) else f"{x:.6f}"))
    view["final_score"] = view["final_score"].map(lambda x: f"{x:.3f}")
    if "streak_days" in view.columns:
        view["streak_days"] = view["streak_days"].map(lambda x: f"{int(x)}")
    view["avg_dollar_vol"] = view["avg_dollar_vol"].map(lambda x: f"{int(x):,}")
    print(f"\nDecision preview: buy = (kronos>=0.5) & (not new) & (liq>={int(args.liq_threshold):,}) & (bull>bear) & (final_score>={args.final_threshold:.3f}).")
    disp_cols = ["instrument", "chinese_name", "model_score", "kronos_score", "bull_score", "bear_score", "net_c", "final_score", "streak_days", "avg_dollar_vol", "is_new_listing", "buy"]
    try:
        if "is_new_listing" in view.columns:
            view["is_new_listing"] = view["is_new_listing"].map(lambda x: "True" if bool(x) else "False")
        print(view[disp_cols].to_string(index=False))
    except Exception:
        print(view[disp_cols])

    if notifier:
        header = [
            f"T2 decision for {msg_day}",
            f"recorder_id: {args.recorder_id}",
        ]
        table_lines: List[str] = []
        try:
            mobile_cols = ["instrument", "chinese_name", "final_score", "buy", "kronos_score", "model_score", "net_c", "streak_days"]
            v2 = view[mobile_cols].copy()
            v2["final_score"] = v2["final_score"].astype(str)
            v2["kronos_score"] = v2.get("kronos_score", "").astype(str)
            v2["model_score"] = v2.get("model_score", "").astype(str)
            v2["net_c"] = v2.get("net_c", "").astype(str)
            widths = {c: max(len(str(c)), int(v2[c].map(lambda x: len(str(x))).max())) for c in mobile_cols}
            header_line = " ".join([str(c).ljust(widths[c]) for c in mobile_cols])
            table_lines.append(header_line)
            for _, row in v2.iterrows():
                cells = [str(row[c]).ljust(widths[c]) for c in mobile_cols]
                table_lines.append(" ".join(cells))
        except Exception:
            try:
                table_lines.append(view[disp_cols].to_string(index=False))
            except Exception:
                pass

        if table_lines:
            payload_lines = ["```"] + header + table_lines + ["```"]
        else:
            payload_lines = header
        notifier.send("\n".join(payload_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder_id", default=None, help="recorder/run id")
    parser.add_argument("--experiment_name", default="workflow", help="experiment name (unused here)")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/hk_data", help="qlib data dir")
    parser.add_argument("--topk", type=int, default=20, help="number of instruments to score")
    parser.add_argument("--lookback", type=int, default=365, help="lookback days for indicators")
    parser.add_argument("--liq_threshold", type=float, default=3000000.0, help="avg dollar vol gate")
    parser.add_argument("--liq_window", type=int, default=60, help="window for avg dollar vol")
    parser.add_argument("--allow_partial", action="store_true", help="allow partial scoring for insufficient-history instruments (skip missing indicators instead of penalizing)")
    parser.add_argument("--final_threshold", type=float, default=2.0, help="minimum final_score threshold for buy filter")
    # Kronos options
    parser.add_argument("--enable_kronos", action="store_true", default=True, help="enable Kronos 7-day uptrend scoring (default on)")
    parser.add_argument("--kronos_model", default="NeoQuasar/Kronos-base", help="Kronos model name or path (default Kronos-small)")
    parser.add_argument("--kronos_device", default="cuda:0", help="device for Kronos (e.g., cpu or cuda:0)")
    parser.add_argument("--kronos_lookback", type=int, default=300, help="lookback bars for Kronos input")
    parser.add_argument("--kronos_pred_len", type=int, default=14, help="prediction horizon for Kronos (days)")
    parser.add_argument("--kronos_max_context", type=int, default=512, help="max context tokens for Kronos predictor")
    parser.add_argument("--kronos_sigmoid_alpha", type=float, default=5.0, help="alpha for deterministic sigmoid mapping of 7D return")
    parser.add_argument("--kronos_debug", action="store_true", help="print Kronos debug info when scoring fails")
    parser.add_argument("--kronos_min_bars", type=int, default=100, help="minimum number of bars required to run Kronos scoring")
    parser.add_argument("--kronos_fill_missing", action="store_true", default=True, help="ffill/bfill missing OHLCV/amount before Kronos input cleaning (default on)")
    parser.add_argument("--telegram_token", help="telegram bot token (optional)")
    parser.add_argument("--telegram_chat_id", help="telegram chat id (optional)")
    parser.add_argument("--notify_config", help="path to JSON config with telegram_token/chat_id")
    args = parser.parse_args()
    # If recorder_id not provided, pick latest run folder under ./mlruns
    if args.recorder_id is None:
        mlruns_dir = os.path.join(".", "mlruns")
        if os.path.isdir(mlruns_dir):
            runs = []
            try:
                for exp in os.listdir(mlruns_dir):
                    exp_path = os.path.join(mlruns_dir, exp)
                    if not os.path.isdir(exp_path):
                        continue
                    for run in os.listdir(exp_path):
                        run_path = os.path.join(exp_path, run)
                        if os.path.isdir(run_path):
                            runs.append(run_path)
            except Exception:
                runs = []
            if runs:
                latest = max(runs, key=lambda p: os.path.getmtime(p))
                args.recorder_id = os.path.basename(os.path.normpath(latest))
                print(f"Auto-detected recorder_id from mlruns: {args.recorder_id}")
        if args.recorder_id is None:
            print("No recorder_id provided and no runs found in ./mlruns. Please supply --recorder_id")
            raise SystemExit(1)
    tok, chat = resolve_notify_params(args.telegram_token, args.telegram_chat_id, args.notify_config)
    notifier = TelegramNotifier(tok, chat, parse_mode="MarkdownV2")
    main(args, notifier)
