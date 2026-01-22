#!/usr/bin/env python3
"""Indicator-based stock selection built on top of T1 predictions.

- Applies liquidity and listing filters (defaults: liq_threshold=30_000_000 HKD, liq_window=60, min_listing_days=120).
- Loads a trained model from a recorder, predicts for the latest trading day, ranks by score.
- Applies top-k (default 200) BEFORE indicator filters.
- Computes two technical indicators (translated from provided Futu custom formulas):
  * indicator1 (EMA momentum cross >0 within last N bars)
  * indicator2 (Guanding composite buy triggers within last N bars)
- Keeps stocks where indicator1 or indicator2 is true in the last lookback window (default 3 bars) and exports a CSV.

Output CSV columns: code, chinese_name, indicator1, indicator2, score
"""
import argparse
import copy
import datetime
import os
import pickle
from typing import Optional, Tuple, List
import sys

import numpy as np
import pandas as pd

import qlib
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.constant import REG_HK
from wcwidth import wcswidth
from qlib.data import D

from qlib.utils.notify import TelegramNotifier, resolve_notify_params
from qlib.utils.func import (
    to_qlib_inst,
    load_chinese_name_map,
    resolve_chinese,
    calendar_last_day,
    next_trading_day_from_future,
)

# Ensure stdout is UTF-8 on Windows
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


def _load_hk_module():
    # import the HK example dataset config
    try:
        from qlib.tests import T0_workflow_by_code_HK as hkmod  # type: ignore
    except Exception:
        import importlib.util
        test_dir = os.path.dirname(__file__)
        wk_path = os.path.join(test_dir, "T0_workflow_by_code_HK.py")
        if not os.path.exists(wk_path):
            raise RuntimeError(f"Cannot find T0_workflow_by_code_HK.py at {wk_path}")
        spec = importlib.util.spec_from_file_location("qlib.tests.T0_workflow_by_code_HK", wk_path)
        hkmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hkmod)  # type: ignore
    return hkmod


def _liquidity_filter(hkmod, target_day, liq_threshold, liq_window, handler_kwargs):
    try:
        hkw = handler_kwargs
        liq_window_use = liq_window
        keep_insts, info = hkmod.compute_liquid_instruments(
            liq_threshold=liq_threshold,
            liq_window=liq_window_use,
            handler_end_time=hkw.get("end_time", None),
        )
        print(f"Liquidity filter: kept {info['kept_count']} / {info['orig_count']} instruments ({info['pct']:.2f}%)")
        print("Sample kept instruments:", info.get("sample", []))
        if len(keep_insts) > 0:
            hkw["instruments"] = keep_insts
    except Exception as e:
        print("Liquidity filter failed:", e)


def _listing_filter(target_day, handler_kwargs, min_listing_days):
    if not (min_listing_days and min_listing_days > 0):
        return
    try:
        insts = list(handler_kwargs.get("instruments", []))
        if not insts:
            return
        listing_df = D.features(
            insts,
            ["$close"],
            start_time="2005-01-01",
            end_time=target_day,
            freq="day",
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
        handler_kwargs["instruments"] = filtered
        kept = len(filtered)
        if filtered_out:
            sample = ", ".join([f"{i}({d})" for i, d in filtered_out[:5]])
            print(f"Listing-day filter >= {min_listing_days}: kept {kept} / {len(insts)}; dropped sample: {sample}")
        else:
            print(f"Listing-day filter >= {min_listing_days}: kept {kept} / {len(insts)}")
    except Exception as e:
        print("Listing-day filter skipped due to error:", e)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def indicator1_signal(df: pd.DataFrame, lookback: int = 3) -> bool:
    """庄家抬轿翻譯: VAR1=EMA(EMA(close,9),9); VAR2=(VAR1-REF(VAR1,1))/REF(VAR1,1)*1000; signal if VAR2 crosses above 0 within lookback bars."""
    close = df["close"].astype(float)
    var1 = _ema(_ema(close, 9), 9)
    var2 = (var1 - var1.shift(1)) / var1.shift(1) * 1000.0
    cross = (var2 > 0) & (var2.shift(1) <= 0)
    return bool(cross.tail(lookback).any())


def indicator2_signal(df: pd.DataFrame, lookback: int = 3) -> bool:
    """冠鼎指标買點：嚴格以 BBUY（DRAWICON(BBUY,30,1)）為準。
    BBUY 定義：D1=(C+L+H)/3; D2=EMA(D1,6); D3=EMA(D2,5); CROSS(D2, D3).
    在最近 lookback 根K內任一觸發即為 True。
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    tp = (close + low + high) / 3.0
    d2 = ema(tp, 6)
    d3 = ema(d2, 5)
    bbuy = (d2 > d3) & (d2.shift(1) <= d3.shift(1))

    return bool(bbuy.tail(lookback).any())


def indicator1_triggers(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series of EMA momentum cross events for indicator1."""
    close = df["close"].astype(float)
    var1 = _ema(_ema(close, 9), 9)
    var2 = (var1 - var1.shift(1)) / var1.shift(1) * 1000.0
    # Strict CROSS(VAR2, 0): require previous value to be valid and <= 0, current > 0
    prev = var2.shift(1)
    cross = (var2 > 0) & (prev <= 0) & var2.notna() & prev.notna()
    return cross


def indicator2_triggers(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series of BBUY events only (CROSS of D2 over D3)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    tp = (close + low + high) / 3.0
    d2 = ema(tp, 6)
    d3 = ema(d2, 5)
    bbuy = (d2 > d3) & (d2.shift(1) <= d3.shift(1))
    return bbuy


def latest_trigger_date(triggers: pd.Series, lookback: int) -> Optional[pd.Timestamp]:
    """Return the latest date within the lookback window when trigger is True."""
    try:
        recent = triggers.tail(lookback)
        idx_true = recent[recent].index
        if len(idx_true) == 0:
            return None
        return pd.to_datetime(idx_true.max())
    except Exception:
        return None


def fetch_ohlcv(insts: List[str], start_day: datetime.date, end_day: datetime.date) -> pd.DataFrame:
    feats = ["$close", "$high", "$low", "$volume"]
    df = D.features(insts, feats, start_time=start_day, end_time=end_day, freq="day")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"$close": "close", "$high": "high", "$low": "low", "$volume": "volume"})
    return df


def main(recorder_id, experiment_name, provider_uri, topk, min_listing_days, liq_threshold, liq_window, lookback, indicator_lookback, output_csv, notifier: Optional[TelegramNotifier] = None):
    provider_uri = os.path.expanduser(provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_HK)

    hkmod = _load_hk_module()

    # load recorder/model
    recorder = None
    model = None
    try:
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
        print(f"Loaded recorder via experiment name: {recorder}")
        model = recorder.load_object("params.pkl")
        print("Model loaded from recorder (by name).")
    except Exception:
        pass
    if model is None:
        raise RuntimeError("Failed to load model from recorder")

    today_dt = datetime.date.today()
    target_day = calendar_last_day(today_dt)
    msg_day = next_trading_day_from_future(provider_uri, target_day) or target_day
    print("target_day (from qlib calendar):", target_day)

    ds_cfg = copy.deepcopy(hkmod.HK_GBDT_TASK["dataset"])
    hkw = ds_cfg["kwargs"]["handler"]["kwargs"]
    try:
        hkw["end_time"] = target_day
    except Exception:
        pass

    # liquidity and listing filters
    _liquidity_filter(hkmod, target_day, liq_threshold, liq_window, hkw)
    _listing_filter(target_day, hkw, min_listing_days)

    # dataset/test segment
    ds_cfg.setdefault("kwargs", {})["segments"] = {"test": (target_day, target_day)}
    dataset = init_instance_by_config(ds_cfg)
    print("Dataset initialized for date:", target_day)

    # predict
    try:
        pred = model.predict(dataset)
    except Exception:
        try:
            X_test = dataset.prepare("test")
            pred = model.predict(X_test)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    # normalize prediction to Series and slice target_day
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
    if isinstance(ss.index, pd.MultiIndex) and inst_level is not None:
        if len(ss.index.names) > 1:
            try:
                ss = ss.groupby(level=inst_level).last()
            except Exception:
                ss = ss.droplevel([n for i, n in enumerate(ss.index.names) if i != inst_level])
    ss = ss.sort_values(ascending=False)

    # apply top-k BEFORE indicator filters
    ss_top = ss.head(topk)
    cand_insts = [to_qlib_inst(i) for i in ss_top.index]

    # fetch OHLCV for indicator computation
    lookback_days = max(120, indicator_lookback * 5)  # ensure enough history
    start_day = target_day - datetime.timedelta(days=lookback_days)
    ohlcv = fetch_ohlcv(cand_insts, start_day, target_day)
    if ohlcv.empty:
        raise RuntimeError("OHLCV fetch empty for candidates")

    mapping = load_chinese_name_map()

    indicator_pass = []
    for inst, score in ss_top.items():
        inst_str = to_qlib_inst(inst)
        try:
            sub = ohlcv.xs(inst_str, level="instrument") if isinstance(ohlcv.index, pd.MultiIndex) else ohlcv
        except Exception:
            sub = ohlcv[ohlcv.get("instrument") == inst_str] if "instrument" in ohlcv.columns else pd.DataFrame()
        if sub.empty:
            ind1 = ind2 = False
            d1 = d2 = None
        else:
            # ensure sorted by date
            sub = sub.sort_index()
            tr1 = indicator1_triggers(sub)
            tr2 = indicator2_triggers(sub)
            ind1 = bool(tr1.tail(indicator_lookback).any())
            ind2 = bool(tr2.tail(indicator_lookback).any())
            d1 = latest_trigger_date(tr1, indicator_lookback)
            d2 = latest_trigger_date(tr2, indicator_lookback)
        latest_dt = None
        if d1 is not None and d2 is not None:
            latest_dt = max(d1, d2)
        elif d1 is not None:
            latest_dt = d1
        elif d2 is not None:
            latest_dt = d2
        indicator_pass.append((inst_str, ind1, ind2, float(score), d1, d2, latest_dt))

    # keep those passing either indicator
    selected = [(c, s, i1, i2, d1, d2, ldt) for (c, i1, i2, s, d1, d2, ldt) in indicator_pass if i1 or i2]

    # sort: first prioritize both indicators true, then by model score; ignore date
    def sort_key(item):
        code, score, i1, i2, d1, d2, ldt = item
        both = 1 if (i1 and i2) else 0
        return (both, float(score))
    selected = sorted(selected, key=sort_key, reverse=True)

    # build table for stdout
    if selected:
        table_rows = []
        for code, score, i1, i2, d1, d2, ldt in selected:
            mk = code.split(".", 1)[0].zfill(5) + ".hk"
            name = resolve_chinese(code, mapping)
            table_rows.append({
                "id": mk,
                "name": name,
                "score": score,
                "indicator1": i1,
                "indicator2": i2,
                "latest_date": (ldt.strftime("%Y-%m-%d") if isinstance(ldt, pd.Timestamp) else "")
            })
        df_view = pd.DataFrame(table_rows)
        df_view["_score_s"] = df_view["score"].map(lambda x: f"{x:.6f}")
        cols = ["id", "name", "_score_s", "indicator1", "indicator2", "latest_date"]
        widths = {c: max(_disp_width(c), int(df_view[c].map(lambda v: _disp_width(v)).max())) for c in cols}
        hdr = f"{_pad_right('id', widths['id'])}  {_pad_right('name', widths['name'])}  {_pad_left('score', widths['_score_s'])}  {_pad_left('indicator1', widths['indicator1'])}  {_pad_left('indicator2', widths['indicator2'])}  {_pad_right('latest_date', widths['latest_date'])}"
        print("\nSelected (indicator OR):")
        print(hdr)
        for _, r in df_view.iterrows():
            row_s = f"{_pad_right(r['id'], widths['id'])}  {_pad_right(r['name'], widths['name'])}  {_pad_left(r['_score_s'], widths['_score_s'])}  {_pad_left(str(r['indicator1']), widths['indicator1'])}  {_pad_left(str(r['indicator2']), widths['indicator2'])}  {_pad_right(r['latest_date'], widths['latest_date'])}"
            print(row_s)

    # export CSV
    day_str = pd.to_datetime(target_day).strftime("%Y%m%d")
    out_csv = output_csv or f"t1b_selection_{day_str}_{recorder_id}.csv"
    export_rows = []
    for code, ind1, ind2, score, d1, d2, ldt in indicator_pass:
        if not (ind1 or ind2):
            continue
        mk = code.split(".", 1)[0].zfill(5) + ".hk"
        name = resolve_chinese(code, mapping)
        export_rows.append({
            "code": mk,
            "chinese_name": name,
            "indicator1": bool(ind1),
            "indicator2": bool(ind2),
            "indicator1_date": (d1.strftime("%Y-%m-%d") if isinstance(d1, pd.Timestamp) else ""),
            "indicator2_date": (d2.strftime("%Y-%m-%d") if isinstance(d2, pd.Timestamp) else ""),
            "latest_indicator_date": (ldt.strftime("%Y-%m-%d") if isinstance(ldt, pd.Timestamp) else ""),
            "score": float(score)
        })
    # sort export rows: prioritize both indicators true, then by score
    if export_rows:
        export_rows = sorted(
            export_rows,
            key=lambda r: (
                1 if (bool(r.get("indicator1")) and bool(r.get("indicator2"))) else 0,
                float(r["score"]),
            ),
            reverse=True,
        )
    if export_rows:
        # Write CSV with UTF-8 BOM to avoid garbled Chinese in Excel on Windows
        pd.DataFrame(export_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Saved selection CSV: {out_csv} ({len(export_rows)} rows)")
    else:
        print("No instruments passed indicator filters; CSV not written.")

    # save prediction pickles (aligned with T1)
    #out_dated = f"pred_{day_str}_{recorder_id}.pkl"
    #with open(out_dated, "wb") as f:
    #    pickle.dump(pred, f)
    #out_legacy = f"pred_today_{recorder_id}.pkl"
    #try:
    #    with open(out_legacy, "wb") as f:
    #        pickle.dump(pred, f)
    #except Exception as e:
    #    print(f"Warning: failed to write legacy pred file {out_legacy}: {e}")

    # notify via Telegram if configured
    #if notifier:
    #    lines = [f"T1b selection for {msg_day}", f"recorder_id: {recorder_id}"]
    #    for row in export_rows[:20]:
    #        lines.append(f"{row['code']} {row['chinese_name']} ind1={row['indicator1']} ind2={row['indicator2']} score={row['score']:.4f}")
    #    notifier.send("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder_id", default=None, help="recorder/run id (run id). If omitted, use latest run folder under ./mlruns")
    parser.add_argument("--experiment_name", default="workflow", help="experiment name")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/hk_data", help="qlib data dir")
    parser.add_argument("--topk", type=int, default=300, help="apply top-k BEFORE indicator filters")
    parser.add_argument("--min_listing_days", type=int, default=120, help="minimum trading days since listing; set 0 to disable filter")
    parser.add_argument("--liq_threshold", type=float, default=30_000_000, help="HKD liquidity threshold")
    parser.add_argument("--liq_window", type=int, default=60, help="liquidity window (days)")
    parser.add_argument("--indicator_lookback", type=int, default=3, help="bars to look back for indicator triggers")
    parser.add_argument("--lookback_days", type=int, default=250, help="override OHLCV history length (days)")
    parser.add_argument("--output_csv", help="output CSV path; default t1b_selection_<date>_<recorder_id>.csv")
    parser.add_argument("--telegram_token", help="telegram bot token (optional)")
    parser.add_argument("--telegram_chat_id", help="telegram chat id (optional)")
    parser.add_argument("--notify_config", help="path to JSON config with telegram_token/chat_id")
    args = parser.parse_args()

    # Auto-detect recorder_id if missing
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
    notifier = TelegramNotifier(tok, chat, parse_mode="MarkdownV2") if tok and chat else None

    # override lookback days if provided
    lookback_days = args.lookback_days if args.lookback_days and args.lookback_days > 0 else 250

    main(
        recorder_id=args.recorder_id,
        experiment_name=args.experiment_name,
        provider_uri=args.provider_uri,
        topk=args.topk,
        min_listing_days=args.min_listing_days,
        liq_threshold=args.liq_threshold,
        liq_window=args.liq_window,
        lookback=lookback_days,
        indicator_lookback=args.indicator_lookback,
        output_csv=args.output_csv,
        notifier=notifier,
    )
