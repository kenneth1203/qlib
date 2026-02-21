#!/usr/bin/env python3
"""Predict today using a saved model from a recorder/run.

Usage:
  python -m qlib.tests.predict_today --recorder_id c10cdaaf600b4c1e85c47c74a966e3e4

This script loads `params.pkl` from the recorder, builds a dataset for today
based on the HK example config, runs prediction, and prints top-k results.
"""
import argparse
import copy
import datetime
import os
import pickle
from typing import Optional
import sys
import pandas as pd

import qlib
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.constant import REG_HK
from wcwidth import wcswidth

from qlib.utils.notify import TelegramNotifier, resolve_notify_params
from qlib.utils.func import (
    to_qlib_inst,
    load_chinese_name_map,
    resolve_chinese,
    calendar_last_day,
    next_trading_day_from_future,
)


def _escape_markdown_v2(text: str) -> str:
    # Telegram MarkdownV2 reserved characters: _ * [ ] ( ) ~ ` > # + - = | { } . !
    if text is None:
        return ""
    s = str(text)
    for ch in "_[]()~`>#+-=|{}.!":
        s = s.replace(ch, f"\\{ch}")
    return s

# Ensure stdout is UTF-8 on Windows to avoid GBK-related mojibake when printing Chinese
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _disp_width(x):
    try:
        return wcswidth(str(x)) if x is not None else 0
    except Exception:
        return len(str(x)) if x is not None else 0


def _pad_right(s, width):
    s = "" if s is None else str(s)
    cur = _disp_width(s)
    if cur >= width:
        return s
    return s + " " * (width - cur)


def _pad_left(s, width):
    s = "" if s is None else str(s)
    cur = _disp_width(s)
    if cur >= width:
        return s
    return " " * (width - cur) + s


def _load_issued_shares(provider_uri: str) -> pd.Series:
    try:
        path = os.path.join(os.path.expanduser(str(provider_uri)), "boardlot", "issued_shares.txt")
        if not os.path.exists(path):
            return pd.Series(dtype=float)
        df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["instrument", "shares"]
        elif df.shape[1] == 1:
            df.columns = ["instrument"]
            df["shares"] = pd.NA
        else:
            return pd.Series(dtype=float)
        df["instrument"] = df["instrument"].astype(str)
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
        df = df.dropna(subset=["shares"])
        return df.set_index("instrument")["shares"].astype(float)
    except Exception:
        return pd.Series(dtype=float)


def _get_turnover_mask(instruments, trade_date, df_all, shares, min_turnover, allow_missing_shares):
    if not instruments:
        return pd.Series(dtype=bool)
    if shares.empty:
        return pd.Series(allow_missing_shares, index=pd.Index(instruments, name="instrument"))
    trade_ts = pd.Timestamp(trade_date)
    if df_all is None or df_all.empty or "$volume" not in df_all.columns:
        return pd.Series(allow_missing_shares, index=pd.Index(instruments, name="instrument"))
    mask = pd.Series(False, index=pd.Index(instruments, name="instrument"))
    for inst in instruments:
        try:
            sub = df_all.xs(inst, level="instrument")
        except Exception:
            continue
        try:
            sub = sub[sub.index <= trade_ts]
            if sub.empty:
                continue
            vol_series = pd.to_numeric(sub["$volume"], errors="coerce")
            med_vol = vol_series.tail(20).median()
            share = shares.get(inst)
            if share is None or pd.isna(share):
                continue
            turnover = med_vol / share
            mask.loc[inst] = turnover >= float(min_turnover)
        except Exception:
            continue
    if allow_missing_shares:
        mask = mask.fillna(True)
    else:
        mask = mask.fillna(False)
    return mask


def main(recorder_id, experiment_name, provider_uri, topk, min_listing_days=120, notifier: Optional[TelegramNotifier] = None):
    provider_uri = os.path.expanduser(provider_uri)
    month_provider_uri = os.path.expanduser("~/.qlib/qlib_data/hk_data_1mo")
    week_provider_uri = os.path.expanduser("~/.qlib/qlib_data/hk_data_1w")
    year_provider_uri = os.path.expanduser("~/.qlib/qlib_data/hk_data_1y")

    active_provider = None

    def _ensure_provider(uri: str):
        nonlocal active_provider
        if active_provider != uri:
            qlib.init(provider_uri=uri, region=REG_HK)
            active_provider = uri

    _ensure_provider(provider_uri)

    # MACD buy + turnover filter settings (aligned with MACDTopkDropoutStrategy_v2 defaults)
    min_turnover = 0.001
    allow_missing_shares = False
    shares = _load_issued_shares(provider_uri)

    # import the HK example dataset config
    try:
        # try normal import first
        from qlib.tests import T0_workflow_by_code_HK as hkmod  # type: ignore
    except Exception:
        # fallback: load module from the tests directory file path
        import importlib.util
        test_dir = os.path.dirname(__file__)
        wk_path = os.path.join(test_dir, "T0_workflow_by_code_HK.py")
        if not os.path.exists(wk_path):
            raise RuntimeError(f"Cannot find T0_workflow_by_code_HK.py at {wk_path}")
        spec = importlib.util.spec_from_file_location("qlib.tests.T0_workflow_by_code_HK", wk_path)
        hkmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hkmod)  # type: ignore

    # get recorder and load model
    recorder = None
    model = None
    # 1) try with experiment name
    try:
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
        print(f"Loaded recorder via experiment name: {recorder}")
        model = recorder.load_object("params.pkl")
        print("Model loaded from recorder (by name).")
    except Exception:
        # 2) try with experiment id if provided
        # note: caller can pass experiment id via environment or later we attempt mlruns
        pass

    # build a dataset for the latest available trading day in qlib calendar (<= today)
    from qlib.data import D

    today_dt = datetime.date.today()
    target_day = calendar_last_day(today_dt)

    msg_day = next_trading_day_from_future(provider_uri, target_day) or target_day

    print("target_day (from qlib calendar):", target_day)

    ds_cfg = copy.deepcopy(hkmod.HK_GBDT_TASK["dataset"])

    # Apply the same liquidity filter used in backtest to keep universes aligned
    try:
        hkw = ds_cfg["kwargs"]["handler"]["kwargs"]
        # Ensure handler end_time is the latest available trading day from qlib calendar
        try:
            hkw["end_time"] = target_day
        except Exception:
            pass
        liq_window = 60
        try:
            csv_path = os.path.abspath(os.path.join(os.getcwd(), "instrument_filtered_bt.csv"))
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"instrument_filtered.csv not found at {csv_path}")

            df_csv = pd.read_csv(csv_path)
            if df_csv.empty:
                raise RuntimeError(f"instrument_filtered.csv is empty at {csv_path}")

            inst_col = "instrument" if "instrument" in df_csv.columns else df_csv.columns[0]
            keep_insts = df_csv[inst_col].astype(str).tolist()
            info = {
                "orig_count": len(keep_insts),
                "kept_count": len(keep_insts),
                "pct": 100.0,
                "sample": keep_insts[:20],
                "csv_path": csv_path,
            }
            for meta in ["min_avg_amount", "avg_amount_window", "min_turnover", "turnover_window", "target_day", "allow_missing_shares"]:
                if meta in df_csv.columns:
                    info[meta] = df_csv[meta].iloc[0]
            print(
                f"Liquidity filter (predict, CSV): kept {info['kept_count']} / {info['orig_count']} instruments ({info['pct']:.2f}%)"
            )
            print("Sample kept instruments:", info.get("sample", []))
            print("Using precomputed CSV:", csv_path)
        except Exception as e:
            print(f"Loading liquidity filter from CSV failed: {e}, falling back to dynamic computation.")
            keep_insts, info = hkmod.compute_liquid_instruments(
                liq_threshold=30_000_000,
                liq_window=liq_window,
                handler_end_time=hkw.get("end_time", None),
            )
            print(
                f"Liquidity filter (predict, fallback): kept {info['kept_count']} / {info['orig_count']} instruments ({info['pct']:.2f}%)"
            )
            print("Sample kept instruments:", info.get("sample", []))

        if len(keep_insts) > 0:
            hkw["instruments"] = keep_insts

        # New listing filter after liquidity filter, before handler uses instruments
        if min_listing_days and min_listing_days > 0:
            insts = list(hkw.get("instruments", []))
            if insts:
                try:
                    #import pandas as pd

                    _ensure_provider(provider_uri)

                    listing_df = D.features(
                        insts,
                        ["$close"],
                        start_time="2005-01-01",
                        end_time=target_day,
                        disk_cache=True,
                    )
                    listed_days_map = {}
                    if isinstance(listing_df, pd.DataFrame):
                        close_s = listing_df[listing_df.columns[0]]
                        if "instrument" in close_s.index.names:
                            listed_days_map = (
                                close_s.groupby(level="instrument")
                                .apply(lambda s: int(s.dropna().shape[0]))
                                .to_dict()
                            )
                    filtered = []
                    filtered_out = []
                    for inst in insts:
                        days = int(listed_days_map.get(inst, 0))
                        if days >= min_listing_days:
                            filtered.append(inst)
                        else:
                            filtered_out.append((inst, days))
                    hkw["instruments"] = filtered
                    kept = len(filtered)
                    if filtered_out:
                        sample = ", ".join([f"{i}({d})" for i, d in filtered_out[:5]])
                        print(
                            f"Listing-day filter >= {min_listing_days}: kept {kept} / {len(insts)}; dropped sample: {sample}"
                        )
                    else:
                        print(f"Listing-day filter >= {min_listing_days}: kept {kept} / {len(insts)}")
                except Exception as e:
                    print("Listing-day filter skipped due to error:", e)
    except Exception as e:
        print("Liquidity filter for predict failed:", e)
    
    # Do NOT overwrite handler start/end (features require historical window).
    # Only set the 'test' segment to target_day so prepare('test') returns features.
    ds_cfg.setdefault("kwargs", {})["segments"] = {"test": (target_day, target_day)}
    #print(ds_cfg)
    dataset = init_instance_by_config(ds_cfg)
    print("Dataset initialized for date:", target_day)

    # Diagnostic: prepare test features and report basic info to help debug empty predictions
    try:
        X_test = dataset.prepare("test")
        try:
            #import pandas as pd

            if hasattr(X_test, "empty") and X_test.empty:
                print("Prepared test features: EMPTY")
            else:
                if isinstance(X_test, (pd.DataFrame, pd.Series)):
                    # infer instrument level robustly
                    n_idx = None
                    if isinstance(X_test.index, pd.MultiIndex):
                        names = list(X_test.index.names)
                        if "instrument" in names:
                            inst_level = names.index("instrument")
                        elif "datetime" in names and len(names) == 2:
                            inst_level = 1 - names.index("datetime")
                        else:
                            inst_level = 0
                        n_idx = len(X_test.index.get_level_values(inst_level).unique())
                    else:
                        n_idx = len(getattr(X_test.index, "unique", lambda: [])())
                    print(f"Prepared test features shape: {getattr(X_test, 'shape', 'unknown')}, instruments: {n_idx}")
                else:
                    print(f"Prepared test features type: {type(X_test)}")
        except Exception:
            print("Prepared test features type/shape:", type(X_test))
    except Exception as e:
        print("dataset.prepare('test') failed:", e)

    # try predict via model API
    pred = None
    try:
        # many qlib models support model.predict(dataset)
        pred = model.predict(dataset)
    except Exception:
        # fallback: prepare test features for today and predict
        try:
            X_test = dataset.prepare("test")
            pred = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    # pred expected as pandas Series/DataFrame with MultiIndex containing instrument & datetime
    try:
        #import pandas as pd

        # normalize to Series of scores
        if isinstance(pred, pd.DataFrame):
            s = pred.iloc[:, 0]
        else:
            s = pred

        if not isinstance(s, pd.Series):
            print("Prediction returned type:", type(pred))
        else:
            names = list(getattr(s.index, "names", []))
            # Try to filter by target_day if datetime level exists
            dt_level = names.index("datetime") if "datetime" in names else None
            inst_level = None
            if "instrument" in names:
                inst_level = names.index("instrument")
            elif dt_level is not None and len(names) == 2:
                inst_level = 1 - dt_level
            # filter to the target date slice if possible
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
            # reduce to instrument-only series and rank
            if isinstance(ss.index, pd.MultiIndex) and inst_level is not None:
                if len(ss.index.names) > 1:
                    # drop datetime by grouping or taking last
                    try:
                        ss = ss.groupby(level=inst_level).last()
                    except Exception:
                        ss = ss.droplevel([n for i, n in enumerate(ss.index.names) if i != inst_level])
            # keep full ranked list (do not cap at 500); selection later truncates to `topk`
            top = ss.sort_values(ascending=False)

            # --- Select top-k (liquidity + listing pre-filters already applied via handler) ---
            cand_insts = [to_qlib_inst(i) for i in top.index]
            print(f"Total candidate instruments after pre-filters: {len(cand_insts)}")

            # apply turnover filter for candidates
            try:
                if cand_insts:
                    start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=liq_window * 3)).strftime("%Y-%m-%d")
                    _ensure_provider(provider_uri)
                    feat_turnover = D.features(
                        cand_insts,
                        ["$volume"],
                        start_time=start_dt,
                        end_time=target_day,
                    )
                    turnover_mask = _get_turnover_mask(
                        cand_insts,
                        target_day,
                        feat_turnover,
                        shares,
                        min_turnover,
                        allow_missing_shares,
                    )
                    cand_insts = [i for i in cand_insts if bool(turnover_mask.get(i, allow_missing_shares))]
                    print(f"Candidates after turnover filter: {len(cand_insts)}")
            except Exception as e:
                print(f"Turnover filter skipped due to error: {e}")
            """
            # apply multi-condition trend filter
            bullish_set = set()
            golden_set = set()

            try:
                feat = None
                week_feat = None
                month_feat = None
                year_feat = None
                if cand_insts:
                    start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=240)).strftime("%Y-%m-%d")
                    week_start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
                    month_start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
                    year_start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=365 * 15)).strftime("%Y-%m-%d")

                    _ensure_provider(provider_uri)
                    feat = D.features(
                        cand_insts,
                        ["$DIF", "$DEA", "$MACD", "$volume", "$close", "$MFI"],
                        start_time=start_dt,
                        end_time=target_day,
                    )

                    _ensure_provider(week_provider_uri)
                    week_feat = D.features(
                        cand_insts,
                        ["$MFI", "$close"],
                        start_time=week_start_dt,
                        end_time=target_day,
                    )

                    _ensure_provider(month_provider_uri)
                    month_feat = D.features(
                        cand_insts,
                        ["$EMA5", "$EMA10", "$EMA20", "$MACD", "$MFI"],
                        start_time=month_start_dt,
                        end_time=target_day,
                    )

                    _ensure_provider(year_provider_uri)
                    year_feat = D.features(
                        cand_insts,
                        ["$MFI"],
                        start_time=year_start_dt,
                        end_time=target_day,
                    )

                    _ensure_provider(provider_uri)
                    if isinstance(feat, pd.DataFrame) and not feat.empty and isinstance(feat.index, pd.MultiIndex):
                        def _get_sub(df, inst):
                            if df is None or df.empty:
                                return pd.DataFrame()
                            try:
                                sub_df = df.xs(inst, level="instrument")
                            except Exception:
                                return pd.DataFrame()
                            sub_df = sub_df[sub_df.index <= pd.Timestamp(target_day)]
                            return sub_df.sort_index()

                        def _get_float(row, key):
                            if row is None or key not in row:
                                return None
                            val = row.get(key)
                            if val is None or pd.isna(val):
                                return None
                            try:
                                return float(val)
                            except Exception:
                                return None

                        def _mfi_gt_ma10(sub_df, row, mfi_val):
                            if sub_df.empty or "$MFI" not in sub_df.columns or mfi_val is None:
                                return False
                            if "$MFI_MA10" in sub_df.columns:
                                mfi_ma10 = row.get("$MFI_MA10")
                                if mfi_ma10 is None or pd.isna(mfi_ma10):
                                    return False
                                return mfi_val > float(mfi_ma10)
                            mfi_series = pd.to_numeric(sub_df["$MFI"], errors="coerce").tail(10)
                            if len(mfi_series) < 2:
                                return False
                            mfi_ma10 = mfi_series.mean()
                            if pd.isna(mfi_ma10):
                                return False
                            return mfi_val > float(mfi_ma10)

                        for inst, sub in feat.groupby(level=0):
                            sub_inst = sub.droplevel(0).sort_index()
                            if sub_inst.empty:
                                continue
                            row = sub_inst.iloc[-1]
                            prev_row = sub_inst.iloc[-2] if len(sub_inst) >= 2 else None
                            prev_row2 = sub_inst.iloc[-3] if len(sub_inst) >= 3 else None

                            dif = _get_float(row, "$DIF")
                            dea = _get_float(row, "$DEA")
                            macd = _get_float(row, "$MACD")
                            macd_prev = _get_float(prev_row, "$MACD")
                            macd_prev2 = _get_float(prev_row2, "$MACD")
                            mfi = _get_float(row, "$MFI")

                            macd_down_2 = (
                                macd is not None
                                and macd_prev is not None
                                and macd_prev2 is not None
                                and macd < macd_prev < macd_prev2
                            )

                            monthly_ok = False
                            sub_m = _get_sub(month_feat, inst)
                            if not sub_m.empty:
                                row_m = sub_m.iloc[-1]
                                prev_row_m = sub_m.iloc[-2] if len(sub_m) >= 2 else None
                                prev_row_m2 = sub_m.iloc[-3] if len(sub_m) >= 3 else None
                                ema5_m = _get_float(row_m, "$EMA5")
                                ema10_m = _get_float(row_m, "$EMA10")
                                ema20_m = _get_float(row_m, "$EMA20")
                                macd_m = _get_float(row_m, "$MACD")
                                mfi_m = _get_float(row_m, "$MFI")
                                macd_m_prev = _get_float(prev_row_m, "$MACD")
                                macd_m_prev2 = _get_float(prev_row_m2, "$MACD")
                                mfi_ok = _mfi_gt_ma10(sub_m, row_m, mfi_m)
                                if ema5_m is not None and ema10_m is not None and ema20_m is not None and mfi_ok:
                                    monthly_ok = ema5_m > ema10_m > ema20_m
                                    if macd_m is not None and macd_m_prev is not None and macd_m_prev2 is not None:
                                        monthly_ok = not (macd_m < macd_m_prev < macd_m_prev2)

                            weekly_ok = False
                            sub_w = _get_sub(week_feat, inst)
                            if not sub_w.empty:
                                row_w = sub_w.iloc[-1]
                                mfi_w = _get_float(row_w, "$MFI")
                                weekly_ok = _mfi_gt_ma10(sub_w, row_w, mfi_w)
                                if weekly_ok and "$close" in sub_w.columns and "$MFI" in sub_w.columns:
                                    close_recent = pd.to_numeric(sub_w["$close"], errors="coerce").tail(10)
                                    mfi_recent = pd.to_numeric(sub_w["$MFI"], errors="coerce").tail(10)
                                    if not close_recent.empty and not mfi_recent.empty:
                                        close_w = close_recent.iloc[-1]
                                        mfi_w_recent = mfi_recent.iloc[-1]
                                        if not pd.isna(close_w) and not pd.isna(mfi_w_recent):
                                            if close_w == close_recent.max() and mfi_w_recent != mfi_recent.max():
                                                weekly_ok = False

                            yearly_ok = False
                            sub_y = _get_sub(year_feat, inst)
                            if not sub_y.empty:
                                row_y = sub_y.iloc[-1]
                                mfi_y = _get_float(row_y, "$MFI")
                                yearly_ok = _mfi_gt_ma10(sub_y, row_y, mfi_y)

                            vol_ok = False
                            if "$volume" in sub_inst.columns:
                                vol_series = pd.to_numeric(sub_inst["$volume"], errors="coerce").tail(10)
                                if len(vol_series) >= 9:
                                    good_days = (vol_series.fillna(0) > 0).sum()
                                    vol_ok = int(good_days) >= 9

                            mfi_day_ok = _mfi_gt_ma10(sub_inst, row, mfi)

                            daily_peak_ok = True
                            if "$close" in sub_inst.columns and "$MFI" in sub_inst.columns:
                                close_series = pd.to_numeric(sub_inst["$close"], errors="coerce").tail(10)
                                mfi_series = pd.to_numeric(sub_inst["$MFI"], errors="coerce").tail(10)
                                if not close_series.empty and not mfi_series.empty:
                                    close_last = close_series.iloc[-1]
                                    mfi_last = mfi_series.iloc[-1]
                                    if not pd.isna(close_last) and not pd.isna(mfi_last):
                                        if close_last == close_series.max() and mfi_last != mfi_series.max():
                                            daily_peak_ok = False

                            daily_ok = (
                                dif is not None
                                and dea is not None
                                and dif > dea
                                and dif > 0
                                and not macd_down_2
                                and vol_ok
                                and mfi_day_ok
                                and daily_peak_ok
                            )

                            macd_buy = daily_ok and monthly_ok and weekly_ok and yearly_ok

                            if macd_buy:
                                bullish_set.add(inst)
            except Exception:
                bullish_set = set()
                golden_set = set()

            if bullish_set:
                turnover_mask = _get_turnover_mask(
                    list(bullish_set),
                    target_day,
                    feat,
                    shares,
                    min_turnover,
                    allow_missing_shares,
                )
                bullish_set = {k for k in bullish_set if bool(turnover_mask.get(k, allow_missing_shares))}
                cand_insts = [i for i in cand_insts if i in bullish_set]
            # select top-k from bullish candidates ranked by model score
            golden_ranked = [i for i in cand_insts if i in golden_set]
            bullish_ranked = [i for i in cand_insts if i in bullish_set and i not in golden_set]
            print("sample of golden cross instruments:", golden_ranked[:5], "sample of bullish instruments:", bullish_ranked[:5])
            selected = (golden_ranked + bullish_ranked)[:topk]
            """
            selected = cand_insts[:topk]
            mapping = load_chinese_name_map()

            # Print selected top instruments with Chinese names and last-day volume
            score_df = top.reindex(cand_insts).to_frame("score")
            score_df.index.name = "instrument"

            feat = None
            vol_map = {}
            turnover_map = {}
            # compute avg dollar volume over last `liq_window` trading bars for selected instruments
            try:
                if selected:
                    # Reuse cached day features when available to avoid extra data fetch.
                    if feat is not None and isinstance(feat, pd.DataFrame) and not feat.empty:
                        base = feat.loc[feat.index.get_level_values("instrument").isin(selected), ["$close", "$volume"]]
                        base = base.rename(columns={"$close": "$close", "$volume": "$volume"})
                    else:
                        start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=liq_window * 3)).strftime("%Y-%m-%d")
                        _ensure_provider(provider_uri)
                        base = D.features(selected, ["$close", "$volume"], start_time=start_dt, end_time=target_day)
                        base.columns = ["$close", "$volume"]

                    # Use 20-day median dollar volume (turnover) instead of long-window mean
                    turnover_window = 20

                    def _tail_median_dollar(df):
                        df2 = df.dropna()
                        if df2.empty:
                            return 0.0
                        dv = (df2["$close"] * df2["$volume"]).tail(turnover_window)
                        return float(dv.median()) if len(dv) > 0 else 0.0

                    def _tail_median_volume(df):
                        df2 = df.dropna()
                        if df2.empty:
                            return 0.0
                        vv = df2["$volume"].tail(turnover_window)
                        return float(vv.median()) if len(vv) > 0 else 0.0

                    grouped = base.groupby(level="instrument")
                    vol_map = grouped.apply(_tail_median_dollar).to_dict()
                    vol_med = grouped.apply(_tail_median_volume).to_dict()
                    for inst, med_vol in vol_med.items():
                        try:
                            share = shares.get(inst)
                            if share is None or pd.isna(share) or float(share) == 0.0:
                                continue
                            turnover_map[inst] = float(med_vol) / float(share)
                        except Exception:
                            continue
            except Exception:
                vol_map = {}
                turnover_map = {}

            # Build a tidy view and print as an aligned table
            rows = []
            for inst in selected:
                score = float(score_df.loc[inst, "score"]) if inst in score_df.index else 0.0
                mk = inst.split(".", 1)[0].zfill(5) + ".hk"
                name = resolve_chinese(inst, mapping)
                avg_dollar = int(round(vol_map.get(inst, 0)))
                turnover = turnover_map.get(inst)
                rows.append(
                    {"id": mk, "name": name, "score": score, "avg_dollar": avg_dollar, "turnover": turnover}
                )

            view_df = pd.DataFrame(rows)
            if not view_df.empty:
                # prepare formatted string columns for aligned printing
                view_df["_score_s"] = view_df["score"].map(lambda x: f"{x:.6f}")
                view_df["_avg_s"] = view_df["avg_dollar"].map(lambda x: f"{int(x):,}")
                view_df["_turn_s"] = view_df["turnover"].map(
                    lambda x: f"{x * 100:.2f}%" if pd.notna(x) else ""
                )
                # determine column widths (display width-aware)
                cols = ["id", "name", "_score_s", "_avg_s", "_turn_s"]
                widths = {}
                for c in cols:
                    header_name = "score" if c == "_score_s" else ("avg_dollar" if c == "_avg_s" else ("turnover" if c == "_turn_s" else c))
                    max_cell = 0
                    if not view_df.empty:
                        max_cell = int(view_df[c].map(lambda v: _disp_width(v)).max())
                    widths[c] = max(_disp_width(header_name), max_cell)
                # header (use display-aware padding)
                hdr = (
                    f"{_pad_right('id', widths['id'])}  {_pad_right('name', widths['name'])}  "
                    f"{_pad_left('score', widths['_score_s'])}  {_pad_left('avg_dollar', widths['_avg_s'])}  "
                    f"{_pad_left('turnover', widths['_turn_s'])}"
                )
                print(f"\nTop {topk} instruments for {target_day}:")
                print(hdr)
                # rows (display-aware padding)
                lines = [hdr]
                for _, r in view_df.iterrows():
                    id_s = r['id'] if pd.notna(r['id']) else ''
                    name_s = r['name'] if pd.notna(r['name']) else ''
                    score_s = r['_score_s'] if pd.notna(r['_score_s']) else ''
                    avg_s = r['_avg_s'] if pd.notna(r['_avg_s']) else ''
                    turn_s = r['_turn_s'] if pd.notna(r['_turn_s']) else ''
                    row_s = (
                        f"{_pad_right(id_s, widths['id'])}  {_pad_right(name_s, widths['name'])}  "
                        f"{_pad_left(score_s, widths['_score_s'])}  {_pad_left(avg_s, widths['_avg_s'])}  "
                        f"{_pad_left(turn_s, widths['_turn_s'])}"
                    )
                    print(row_s)
                    lines.append(row_s)
    except Exception:
        print("Prediction result:", pred)

    # save model prediction in original format (do not overwrite with selected list)
    pred_orig = pred
    day_str = pd.to_datetime(target_day).strftime("%Y%m%d")
    out_dated = f"pred_{day_str}_{recorder_id}.pkl"
    with open(out_dated, "wb") as f:
        pickle.dump(pred_orig, f)
    print(f"Saved prediction to {out_dated}")

    # save selected list to CSV with id, name, score
    try:
        if "selected" in locals() and "score_df" in locals():
            rows_csv = []
            for inst in selected:
                try:
                    score_val = float(score_df.loc[inst, "score"]) if inst in score_df.index else 0.0
                except Exception:
                    score_val = 0.0
                mk = inst.split(".", 1)[0].zfill(5) + ".hk"
                name = resolve_chinese(inst, mapping) if "mapping" in locals() else ""
                rows_csv.append({"id": mk, "name": name, "score": score_val})
            sel_df = pd.DataFrame(rows_csv)
            sel_path = f"selection_{day_str}_{recorder_id}.csv"
            sel_df.to_csv(sel_path, index=False, encoding="utf-8-sig")
            print(f"Saved selection CSV to {sel_path}")
    except Exception as e:
        print(f"Warning: failed to write selection CSV: {e}")

    if notifier:
        preview = ", ".join(selected) if 'selected' in locals() else ""
        table_lines = []
        if 'lines' in locals():
            table_lines = lines
        if table_lines:
            payload_lines = ["```", f"T1 selection for {msg_day}", f"recorder_id: {recorder_id}"] + table_lines + ["```"]
        else:
            payload_lines = [f"T1 selection for {msg_day}", f"recorder_id: {recorder_id}"]
        #safe_lines = [_escape_markdown_v2(line) for line in payload_lines]
        notifier.send("\n".join(payload_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder_id", default=None, help="recorder/run id (run id). If omitted, use latest run folder under ./mlruns")
    parser.add_argument("--experiment_name", default="workflow", help="experiment name")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/hk_data", help="qlib data dir")
    parser.add_argument("--topk", type=int, default=20, help="print top-k instruments")
    parser.add_argument("--min_listing_days", type=int, default=0, help="minimum trading days since listing; set 0 to disable filter")
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
    main(args.recorder_id, args.experiment_name, args.provider_uri, args.topk, args.min_listing_days, notifier)