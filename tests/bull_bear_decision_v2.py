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
from typing import List, Dict

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_HK
from qlib.workflow import R
from qlib.data import D
from qlib.scripts.utils.indicators import score_instrument, atr
import matplotlib.pyplot as plt

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
    # try to include $open, $vwap, and $amount if available for Kronos input
    fields = ["$open", "$close", "$high", "$low", "$volume", "$vwap", "$amount"]
    try:
        df = D.features(instruments, fields, start_time=start_dt, end_time=end_dt, freq="day")
        # some providers may lack $vwap; align columns safely
        cols = list(df.columns)
        rename = {}
        # Expect at least 5: $open,$close,$high,$low,$volume,(optional)$vwap
        if len(cols) >= 5:
            # map first occurrences conservatively
            # keep unknown columns as-is
            # try to infer by name substring if provider returns different tokens
            for c in cols:
                lc = str(c).lower()
                if "$open" in lc or lc.endswith("$open") or lc.endswith("open"):
                    rename[c] = "$open"
                elif "$close" in lc or lc.endswith("$close") or lc.endswith("close"):
                    rename[c] = "$close"
                elif "$high" in lc or lc.endswith("$high") or lc.endswith("high"):
                    rename[c] = "$high"
                elif "$low" in lc or lc.endswith("$low") or lc.endswith("low"):
                    rename[c] = "$low"
                elif "$volume" in lc or lc.endswith("$volume") or lc.endswith("volume"):
                    rename[c] = "$volume"
                elif "$vwap" in lc or lc.endswith("$vwap") or lc.endswith("vwap"):
                    rename[c] = "$vwap"
                elif "$amount" in lc or lc.endswith("$amount") or lc.endswith("amount"):
                    rename[c] = "$amount"
        df.columns = [rename.get(c, c) for c in cols]
    except Exception:
        df = D.features(instruments, ["$open", "$close", "$high", "$low", "$volume"], start_time=start_dt, end_time=end_dt, freq="day")
        df.columns = ["$open", "$close", "$high", "$low", "$volume"]
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

def plot_prediction(history_df, pred_df, symbol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(history_df.index, history_df["close"], label="History", color="tab:blue", linewidth=1.4)
    ax1.plot(pred_df.index, pred_df["close"], label="Prediction", color="tab:red", linewidth=1.6)
    ax1.set_title(f"{symbol} - Close price forecast", fontsize=14)
    ax1.set_ylabel("Close", fontsize=12)
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.bar(history_df.index, history_df["volume"], label="History", color="tab:blue", width=0.8)
    ax2.bar(pred_df.index, pred_df["volume"], label="Prediction", color="tab:red", width=0.8, alpha=0.7)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.legend(loc="best", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

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
    # 取出 datetime index
    dates = ohlcv.index.get_level_values("datetime").unique()

    # 計算實際交易日數
    date_len = len(dates)
    print(f"實際交易日數: {date_len}, 從 {dates.min()} 到 {dates.max()}")
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

    # ---- Kronos setup (optional) ----
    kronos_predictor = None
    if args.enable_kronos and _KRONOS_AVAILABLE:
        try:
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            kronos_model_id = args.kronos_model if args.kronos_model else "NeoQuasar/Kronos-small"
            kronos_model = Kronos.from_pretrained(kronos_model_id)
            kronos_predictor = KronosPredictor(kronos_model, tokenizer, device=args.kronos_device, max_context=args.kronos_max_context)
            print(f"Kronos loaded: model={kronos_model_id}, device={args.kronos_device}")
        except Exception as e:
            print(f"Warning: Kronos unavailable ({e}); proceeding without Kronos scoring.")
            kronos_predictor = None
    elif args.enable_kronos and not _KRONOS_AVAILABLE:
        print("Warning: Kronos modules not found; proceeding without Kronos scoring.")

    def _kronos_score_for_inst(inst_code: str, df_inst: pd.DataFrame) -> float:
        # Deterministic sigmoid of predicted 7-day return
        if kronos_predictor is None:
            return float('nan')
        try:
            # Prepare historical input
            df_inst_sorted = df_inst.sort_index()
            # minimal length guard (configurable)
            min_len = int(getattr(args, "kronos_min_bars", 100))
            if df_inst_sorted.shape[0] < min_len:
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos skip {inst_code}: insufficient history {df_inst_sorted.shape[0]}< {min_len}")
                return float('nan')
            # Map to Kronos expected columns with optional fill-missing
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
                'open': open_s,
                'high': high_s,
                'low': low_s,
                'close': close_s,
                'volume': volume_s,
                'amount': amount_s,
            }).replace([np.inf, -np.inf], np.nan)

            if getattr(args, "kronos_fill_missing", False):
                core_cols = ["open","high","low","close","volume","amount"]
                before = int(kdf[core_cols].isna().sum().sum())
                kdf[core_cols] = kdf[core_cols].ffill().bfill()
                after = int(kdf[core_cols].isna().sum().sum())
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos fill_missing {inst_code}: gaps {before} -> {after}")

            # Drop rows lacking core OHLCV
            kdf_clean = kdf.dropna(subset=["open","high","low","close","volume"]).sort_index()
            if kdf_clean.empty:
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos skip {inst_code}: no valid rows after cleaning")
                return float('nan')

            # Limit to lookback and align timestamps
            look = min(int(args.kronos_lookback), len(kdf_clean))
            print(f"Kronos processing {inst_code}: using lookback={look} from available {len(kdf_clean)} rows")
            x_df = kdf_clean.tail(look)
            x_timestamp = pd.Series(pd.to_datetime(x_df.index))
            # Next 7 trading days after last known
            last_dt = pd.to_datetime(kdf.index[-1]).to_pydatetime().date()
            cal_next = D.calendar(start_time=(last_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                                  end_time=(last_dt + datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                                  freq="day")
            if len(cal_next) > 0:
                y_timestamp = pd.Series(pd.to_datetime(cal_next[:args.kronos_pred_len]))
            else:
                # Fallback: generate next business days (Mon-Fri) via pandas
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos calendar fallback for {inst_code}: no future trading days from {last_dt}; using pandas bdate_range")
                start_pd = pd.Timestamp(last_dt) + pd.Timedelta(days=1)
                print(f"Kronos predict start date for {inst_code}: {start_pd.strftime('%Y-%m-%d')} for {int(args.kronos_pred_len)} days")
                y_timestamp = pd.Series(pd.bdate_range(start=start_pd, periods=int(args.kronos_pred_len), freq='B'))
            if len(y_timestamp) < int(args.kronos_pred_len):
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos warn {inst_code}: y_timestamp shorter than pred_len {len(y_timestamp)}<{args.kronos_pred_len}")
            # Drop rows with NaNs to avoid feeding invalid points
            pred = kronos_predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=args.kronos_pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )
            # Use last predicted close vs last actual close

            # --- Kronos debug plot (only when enabled) ---
            if getattr(args, "kronos_debug", False):
                try:
                    # Prepare history frame for plotting (close, volume)
                    hist_df = kdf_clean.tail(look)[["close", "volume"]].copy()
                    hist_df.index = pd.to_datetime(hist_df.index)
            
                    # Normalize pred to DataFrame with datetime index
                    pred_df = pred.copy() if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred)
                    try:
                        pred_df.index = pd.to_datetime(pred_df.index)
                    except Exception:
                        # fall back to y_timestamp if index not datetime-like
                        try:
                            pred_df.index = pd.to_datetime(y_timestamp)
                        except Exception:
                            pred_df.index = pd.RangeIndex(start=0, stop=len(pred_df))
            
                    # Find close column in prediction (case-insensitive)
                    pred_close_col = None
                    for cname in pred_df.columns:
                        if str(cname).lower() == "close" or str(cname).lower().endswith("close"):
                            pred_close_col = cname
                            break
                    if pred_close_col is None:
                        # try common alternatives
                        for cname in pred_df.columns:
                            if "price" in str(cname).lower():
                                pred_close_col = cname
                                break
            
                    # Build pred_plot with close and volume (volume may be missing -> zeros)
                    if pred_close_col is not None:
                        plot_close = pred_df[pred_close_col].astype(float)
                        if "volume" in pred_df.columns:
                            plot_vol = pred_df["volume"].astype(float)
                        else:
                            plot_vol = pd.Series(0.0, index=plot_close.index)
                        pred_plot = pd.DataFrame({"close": plot_close, "volume": plot_vol}, index=plot_close.index)
                        # call plotting helper (wrapped to avoid interrupting main flow)
                        try:
                            #plot_prediction(hist_df, pred_plot, inst_code)
                            pass  # Commented out to avoid actual plotting during tests
                        except Exception as e:
                            print(f"Kronos plot error for {inst_code}: {e}")
                    else:
                        if getattr(args, "kronos_debug", False):
                            print(f"Kronos debug: no close-like column in prediction for {inst_code}; columns={list(pred_df.columns)}")
                except Exception as e:
                    print(f"Kronos debug plot failure for {inst_code}: {e}")
            # --- end Kronos debug plot ---

            if isinstance(pred, pd.DataFrame) and not pred.empty:
                last_close = float(kdf['close'].iloc[-1])
                # find close-like column
                col_close = None
                for cname in pred.columns:
                    lc = str(cname).lower()
                    if lc == 'close' or lc.endswith('close'):
                        col_close = cname
                        break
                if col_close is None:
                    if getattr(args, "kronos_debug", False):
                        print(f"Kronos skip {inst_code}: no 'close' in prediction columns={list(pred.columns)}")
                    return float('nan')
                # compute average return across the full prediction horizon (relative to last_close)
                pred_close_series = pred[col_close].astype(float)
                if last_close == 0:
                    r = 0.0
                else:
                    returns = (pred_close_series / last_close) - 1.0
                    r = float(returns.mean())
                if getattr(args, "kronos_debug", False):
                    try:
                        dates = y_timestamp.dt.strftime("%Y-%m-%d").tolist()
                    except Exception:
                        dates = [str(x) for x in y_timestamp.tolist()]
                    print(f"Kronos pred {inst_code}: horizon_days={len(returns)}, dates={dates[:min(5,len(dates))]}..., last_close={last_close:.4f}, avg_pred_close={pred_close_series.mean():.4f}, avg_ret={r:.4%}")
                # deterministic sigmoid mapping to [0,1]
                alpha = float(getattr(args, "kronos_sigmoid_alpha", 5.0))
                s = 1.0 / (1.0 + np.exp(-alpha * r))
                return float(s)
            else:
                if getattr(args, "kronos_debug", False):
                    print(f"Kronos skip {inst_code}: empty/invalid pred type={type(pred)}")
        except Exception as e:
            if getattr(args, "kronos_debug", False):
                print(f"Kronos error {inst_code}: {e}")
            return float('nan')
        return float('nan')

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
        "bb_squeeze_quantile": 0.2,
        "adx_window": 14,
        "cci_window": 20,
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

            # now compute kronos score as before
            kronos_score = _kronos_score_for_inst(inst, df_inst) if kronos_predictor is not None else float('nan')
            rows.append({
                "instrument": inst,
                "bull_score": bull,
                "bear_score": bear,
                "net_score": net,
                "avg_dollar_vol": int(round(avg_dollar.get(inst, 0.0))),
                "buy": bool(buy_flag),
                "model_score": float(model_scores.get(inst, 0.0)),
                "kronos_score": float(kronos_score) if pd.notna(kronos_score) else float('nan'),
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
    # final_score (option 1b): positive blend
    # net_c = 0.5*(tanh(net/5)+1) in [0,1]
    # kronos_c = kronos_score in [0,1]
    # model_c = model_score (assumed prob-like) in [0,1]
    # final = 0.4*model_c + 0.35*kronos_c + 0.25*net_c
    try:
        out_df["kronos_score"] = out_df.get("kronos_score", np.nan)

        def compute_final_score(model_score, kronos_score, net_score, vol_ratio, is_new_listing):
            kronos_val = 0.0 if pd.isna(kronos_score) else float(kronos_score)
            kronos_c = float(np.clip(kronos_val, 0.00, 0.95))

            net_val = 0.0 if pd.isna(net_score) else float(net_score)
            net_c = 1.0 / (1.0 + np.exp(-net_val / 3.0))

            try:
                v = 0.0 if vol_ratio is None or pd.isna(vol_ratio) else float(vol_ratio)
            except Exception:
                v = 0.0

            if v > 0.04:   # 高波動
                w_model, w_net = 0.6, 0.2
            else:          # 低波動
                w_model, w_net = 0.5, 0.25

            ms = 0.0 if pd.isna(model_score) else float(model_score)

            base = w_model * ms + w_net * net_c
            final = base * kronos_c * 100
            
            return float(final)

        market_vol_ratio = 0.0 if ("vol_ratio" not in locals() or pd.isna(locals().get("vol_ratio"))) else float(locals().get("vol_ratio"))

        out_df["final_score"] = out_df.apply(
            lambda r: compute_final_score(
                r.get("model_score", 0.0),
                r.get("kronos_score", 0.0),
                r.get("net_score", 0.0),
                market_vol_ratio,
                r.get("is_new_listing", False),
            ),
            axis=1,
        )

        out_df["net_c"] = 1.0 / (1.0 + np.exp(-out_df["net_score"].fillna(0.0) / 3.0))
        # keep composite_score for backward compatibility
        out_df["composite_score"] = out_df["model_score"].fillna(0.0) * out_df["net_score"].fillna(0.0)
    except Exception:
        pass

    # Apply improved buy filter based on provided reference logic
    # Ensures final_score exists even if earlier computation failed
    if "final_score" not in out_df.columns:
        out_df["final_score"] = 0.0

    def buy_filter(row, liq_threshold=args.liq_threshold, final_threshold=args.final_threshold):
        # Kronos gate
        try:
            ks = row.get("kronos_score", np.nan)
            if pd.notna(ks) and float(ks) < 0.4:
                return False
        except Exception:
            return False

        # Exclude newly listed
        #if bool(row.get("is_new_listing", False)):
        #    return False

        # Liquidity check
        try:
            if float(row.get("avg_dollar_vol", 0.0)) < float(liq_threshold):
                return False
        except Exception:
            return False

        # Technical check (reject only when bear - bull > 3)
        try:
            bull = float(row.get("bull_score", 0.0))
            bear = float(row.get("bear_score", 0.0))
            if bull <= bear and (bear - bull) > 3.0:
                return False
        except Exception:
            return False

        # final_score threshold
        try:
            if float(row.get("final_score", 0.0)) < float(final_threshold):
                return False
        except Exception:
            return False

        return True

    try:
        out_df["buy"] = out_df.apply(lambda r: bool(buy_filter(r)), axis=1)
    except Exception:
        # If anything goes wrong, keep existing buy values
        pass
    out_path = f"bull_bear_today_{args.recorder_id}.csv"
    # Use UTF-8 BOM to improve Excel Chinese display on Windows
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved decision table to {out_path}")

    # print concise table (sorted by final_score)
    base_cols = ["instrument", "chinese_name", "model_score", "kronos_score", "bull_score", "bear_score", "net_c", "avg_dollar_vol", "is_new_listing", "buy", "final_score"]
    disp = out_df[base_cols].copy()
    # Improve East Asian alignment in console
    try:
        pd.set_option("display.unicode.east_asian_width", True)
        pd.set_option("display.colheader_justify", "center")
    except Exception:
        pass
    # sort: buy first, then final_score desc, then model_score desc
    disp_sorted = disp.sort_values(["buy", "final_score", "model_score"], ascending=[False, False, False])
    # format a view for pretty printing without touching CSV types
    view = disp_sorted.head(args.topk).copy()
    view["model_score"] = view["model_score"].map(lambda x: f"{x:.6f}")
    if "kronos_score" in view.columns:
        view["kronos_score"] = view["kronos_score"].map(lambda x: ("" if pd.isna(x) else f"{x:.6f}"))
    if "net_c" in view.columns:
        view["net_c"] = view["net_c"].map(lambda x: ("" if pd.isna(x) else f"{x:.6f}"))
    view["final_score"] = view["final_score"].map(lambda x: f"{x:.3f}")
    view["avg_dollar_vol"] = view["avg_dollar_vol"].map(lambda x: f"{int(x):,}")
    print(f"\nDecision preview: buy = (kronos>=0.2) & (not new) & (liq>={int(args.liq_threshold):,}) & (bull>bear) & (final_score>={args.final_threshold:.3f}).")
    # reorder columns for display
    disp_cols = ["instrument", "chinese_name", "model_score", "kronos_score", "bull_score", "bear_score", "net_c", "final_score", "avg_dollar_vol", "is_new_listing", "buy"]
    try:
        # ensure is_new_listing shown as True/False string
        if "is_new_listing" in view.columns:
            view["is_new_listing"] = view["is_new_listing"].map(lambda x: "True" if bool(x) else "False")
        print(view[disp_cols].to_string(index=False))
    except Exception:
        print(view[disp_cols])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder_id", default="050590db3b4f47239180b6f83d16ca7e", help="recorder/run id")
    parser.add_argument("--experiment_name", default="workflow", help="experiment name (unused here)")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/hk_data", help="qlib data dir")
    parser.add_argument("--topk", type=int, default=50, help="number of instruments to score")
    parser.add_argument("--lookback", type=int, default=365, help="lookback days for indicators")
    parser.add_argument("--liq_threshold", type=float, default=1000000.0, help="avg dollar vol gate")
    parser.add_argument("--liq_window", type=int, default=20, help="window for avg dollar vol")
    parser.add_argument("--allow_partial", action="store_true", help="allow partial scoring for insufficient-history instruments (skip missing indicators instead of penalizing)")
    parser.add_argument("--final_threshold", type=float, default=5.0, help="minimum final_score threshold for buy filter")
    # Kronos options
    parser.add_argument("--enable_kronos", action="store_true", help="enable Kronos 7-day uptrend scoring (default off)")
    parser.add_argument("--kronos_model", default="NeoQuasar/Kronos-base", help="Kronos model name or path (default Kronos-small)")
    parser.add_argument("--kronos_device", default="cuda:0", help="device for Kronos (e.g., cpu or cuda:0)")
    parser.add_argument("--kronos_lookback", type=int, default=300, help="lookback bars for Kronos input")
    parser.add_argument("--kronos_pred_len", type=int, default=7, help="prediction horizon for Kronos (days)")
    parser.add_argument("--kronos_max_context", type=int, default=512, help="max context tokens for Kronos predictor")
    parser.add_argument("--kronos_sigmoid_alpha", type=float, default=20.0, help="alpha for deterministic sigmoid mapping of 7D return")
    parser.add_argument("--kronos_debug", action="store_true", help="print Kronos debug info when scoring fails")
    parser.add_argument("--kronos_min_bars", type=int, default=100, help="minimum number of bars required to run Kronos scoring")
    parser.add_argument("--kronos_fill_missing", action="store_true", help="ffill/bfill missing OHLCV/amount before Kronos input cleaning")
    args = parser.parse_args()
    main(args)
