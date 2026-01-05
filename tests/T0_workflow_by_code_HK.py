#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
"""
Qlib provides two kinds of interfaces.
(1) Users could define the Quant research workflow by a simple configuration.
(2) Qlib is designed in a modularized way and supports creating research workflow by code just like building blocks.

The interface of (1) is `qrun XXX.yaml`.  The interface of (2) is script like this, which nearly does the same thing as `qrun XXX.yaml`
"""
import qlib
from qlib.constant import REG_HK
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
import pandas as pd
import numpy as np
import os

HK_BENCH = "800000.HK"  # ^HSI 被存成 800000.HK

# 簡易的 HK Alpha158+LGB 配置（可以依需要調整）
HK_GBDT_TASK = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": "2005-01-01",        # ✅ 你的資料起點
                    "end_time":   "2025-12-17",        # ✅ 你的資料終點
                    "fit_start_time": "2005-01-01",    # ✅ normalize 用的資料
                    "fit_end_time":   "2021-12-31",    # ✅ 避免洩漏 test 資料
                    "instruments": "all",  # 若要限定標的可改成 ["800000.HK"] 或自訂清單
                },
            },
            "segments": {
                "train": ("2005-01-01", "2019-12-31"),   # ✅ 15 年訓練
                "valid": ("2020-01-01", "2021-12-31"),   # ✅ 2 年驗證
                "test":  ("2022-01-01", "2025-12-17")    # ✅ 4 年測試
            },
        },
    },
}


def compute_liquid_instruments(liq_threshold=1_000_000, liq_window=20, handler_end_time=None):
    """Return instruments that pass rolling dollar-volume filter.

    Parameters
    ----------
    liq_threshold : float
        Minimum average dollar volume (HKD).
    liq_window : int
        Rolling window (trading days) for the average.
    handler_end_time : str or pd.Timestamp, optional
        End time for the feature pull; falls back to today if None.
    """

    all_insts = D.instruments("all")

    end_time = handler_end_time or pd.Timestamp.today().strftime("%Y-%m-%d")
    start_time = (pd.to_datetime(end_time) - pd.Timedelta(days=liq_window * 3)).strftime("%Y-%m-%d")

    feat = D.features(all_insts, ["$close", "$volume"], start_time=start_time, end_time=end_time, freq="day", disk_cache=True)
    feat.columns = ["$close", "$volume"]
    feat_df = feat.reset_index()
    feat_df["dollar_vol"] = feat_df["$close"] * feat_df["$volume"]

    try:
        _minp = max(10, liq_window // 2)
        dv_mean = feat_df.groupby("instrument")["dollar_vol"].rolling(window=liq_window, min_periods=_minp).mean().reset_index()
        last_mean = dv_mean.groupby("instrument").tail(1).set_index("instrument")["dollar_vol"]
        keep_insts = last_mean[last_mean >= float(liq_threshold)].index.astype(str).tolist()
    except Exception:
        dv = feat_df[["instrument", "dollar_vol"]].dropna()

        def _last_mean(g):
            vals = g["dollar_vol"].tail(liq_window)
            return vals.mean() if len(vals) >= max(10, liq_window // 2) else float("nan")

        inst_mean = dv.groupby("instrument").apply(_last_mean)
        keep_insts = inst_mean[inst_mean >= float(liq_threshold)].index.astype(str).tolist()

    orig_count = int(feat_df["instrument"].nunique())
    keep_insts = list(dict.fromkeys(keep_insts))
    kept_count = len(keep_insts)
    pct = (kept_count / orig_count) * 100 if orig_count > 0 else float("nan")

    info = {
        "orig_count": orig_count,
        "kept_count": kept_count,
        "pct": pct,
        "sample": keep_insts[:20],
    }
    return keep_insts, info


if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/hk_data"  # HK 資料目錄
    GetData().qlib_data(target_dir=provider_uri, region=REG_HK, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_HK)

    # Determine last trading day from qlib calendar and update task/backtest dates
    try:
        # Try explicit call; some qlib versions accept None arguments
        cal = D.calendar(start_time=None, end_time=None, freq="day")
        last_day = pd.to_datetime(cal[-1]).strftime("%Y-%m-%d") if len(cal) > 0 else pd.Timestamp.today().strftime("%Y-%m-%d")
    except Exception:
        try:
            cal = D.calendar(freq="day")
            last_day = pd.to_datetime(cal[-1]).strftime("%Y-%m-%d") if len(cal) > 0 else pd.Timestamp.today().strftime("%Y-%m-%d")
        except Exception:
            last_day = pd.Timestamp.today().strftime("%Y-%m-%d")
    print("Last trading day determined from qlib calendar:", last_day)
    # Update handler end_time and test segment end to last trading day (do NOT change fit_end_time)
    HK_GBDT_TASK["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = last_day
    HK_GBDT_TASK["dataset"]["kwargs"]["segments"]["test"] = (
        HK_GBDT_TASK["dataset"]["kwargs"]["segments"]["test"][0],
        last_day,
    )

    # keep a module-scoped name for later use (extra_quote / backtest override)
    _last_trading_day = last_day

    # --- Liquidity filter: reuse helper for consistent universe ---
    handler_kwargs = HK_GBDT_TASK["dataset"]["kwargs"]["handler"]["kwargs"]
    keep_insts, info = compute_liquid_instruments(
        liq_threshold=60_000_000,
        liq_window=20,
        handler_end_time=handler_kwargs.get("end_time", None),
    )
    print(f"Liquidity filter: kept {info['kept_count']} / {info['orig_count']} instruments ({info['pct']:.2f}%)")
    print("Sample kept instruments:", info["sample"])

    if len(keep_insts) == 0:
        import warnings
        warnings.warn("No instruments pass liquidity threshold; continuing with full universe.")
    else:
        HK_GBDT_TASK["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = keep_insts
    # --- end liquidity filter ---

    model = init_instance_by_config(HK_GBDT_TASK["model"])
    dataset = init_instance_by_config(HK_GBDT_TASK["dataset"])

    # ---- Load boardlot mapping and build extra_quote for correct lot rounding ----
    boardlot_path = os.path.join(os.path.expanduser(provider_uri), "boardlot", "boardlot.txt")
    if os.path.exists(boardlot_path):
        print("Loading boardlot from", boardlot_path)
        bl = pd.read_csv(
            boardlot_path,
            sep=r"\s+",
            header=None,
            names=["instrument", "board_lot"],
            dtype={"instrument": str},
        )
        bl = bl.set_index("instrument")["board_lot"].astype(float)
        factor_map = (qlib.config.C.trade_unit / bl).rename("$factor")

        # Build extra_quote with $close, $volume, and $factor for backtest date range
        start, end = "2022-01-01", _last_trading_day
        base = D.features(
            factor_map.index.tolist(),
            ["$close", "$volume"],
            start_time=start,
            end_time=end,
            freq="day",
        )
        base.columns = ["$close", "$volume"]
        extra_quote = base.copy()
        extra_quote["$factor"] = extra_quote.index.get_level_values("instrument").map(factor_map)
        # ensure column set is exactly { "$close", "$volume", "$factor" }
        extra_quote = extra_quote[["$close", "$volume", "$factor"]]
        # Add volume expression column required by exchange volume_threshold
        # Keep expression name in sync with exchange_kwargs below
        vol_expr = "0.2 * $volume"
        if vol_expr not in extra_quote.columns:
            extra_quote[vol_expr] = extra_quote["$volume"] * 0.2
    else:
        extra_quote = None

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 12,
                "n_drop": 2,
                "only_tradable": True,
                "forbid_all_trade_at_limit": True
            },
        },
        "backtest": {
            "start_time": "2022-01-01",   # ✅ 與 test 對齊
            "end_time":   "2025-12-17",   # ✅ 與 test 對齊
            "account": 10000000,
            "benchmark": HK_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "volume_threshold": {
                    # limit per-step traded volume to 10% of daily volume
                    "all": ("current", "0.2 * $volume")
                },
                # Inject per-instrument board-lot via factor if available
                "extra_quote": extra_quote,
                "trade_unit": qlib.config.C.trade_unit,
            },
        },
    }

    # If we computed a calendar last trading day, override backtest end_time for consistency
    try:
        port_analysis_config["backtest"]["end_time"] = _last_trading_day
    except NameError:
        pass

        # NOTE: This line is optional
    # It demonstrates that the dataset can be used standalone.
    example_df = dataset.prepare("train")
    print(example_df.head())

    # start exp
    with R.start(experiment_name="workflow"):
        import copy
        # create a safe copy for logging to avoid MLflow param-length truncation
        _log_task = copy.deepcopy(HK_GBDT_TASK)
        try:
            hkwargs = _log_task["dataset"]["kwargs"]["handler"]["kwargs"]
            if "instruments" in hkwargs:
                inst_val = hkwargs["instruments"]
                if isinstance(inst_val, (list, tuple)):
                    hkwargs["instruments"] = f"{len(inst_val)} instruments"
                else:
                    hkwargs["instruments"] = str(inst_val)[:200]
        except Exception:
            pass
        R.log_params(**flatten_dict(_log_task))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # prediction
        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # --- Diagnostics: inspect saved predictions ---
        try:
            import pandas as _pd
            pred_obj = recorder.load_object("pred.pkl")
            print("\n--- Prediction diagnostics ---")
            if isinstance(pred_obj, _pd.DataFrame):
                p = pred_obj
            elif isinstance(pred_obj, _pd.Series):
                p = pred_obj.to_frame(name="score")
            else:
                p = _pd.DataFrame(pred_obj)

            print("pred shape:", getattr(p, "shape", None))
            try:
                insts = p.index.get_level_values("instrument").unique()
                dts = p.index.get_level_values("datetime").unique()
                print("unique instruments:", len(insts))
                print("unique dates:", len(dts))
                print("per-date counts:\n", p.groupby(level="datetime").size().describe())
                print(p.head(20))
                # example top-n on most recent date
                try:
                    last_dt = _pd.to_datetime(dts).max()
                    mask = p.index.get_level_values("datetime") == last_dt
                    last_slice = p[mask]
                    print("\nTop candidates on last date:")
                    if not last_slice.empty:
                        if "score" in last_slice.columns:
                            print(last_slice["score"].nlargest(20))
                        else:
                            print(last_slice.iloc[:, 0].nlargest(20))
                except Exception:
                    pass

                # check $close/$volume NaNs for a sample of instruments on last date
                try:
                    sample_insts = list(insts)[:200]
                    last_date = _pd.to_datetime(dts).max().strftime("%Y-%m-%d")
                    q = D.features(sample_insts, ["$close", "$volume"], start_time=last_date, end_time=last_date, freq="day")
                    print("\n$close/$volume NaN counts (sample):\n", q.isna().sum())
                except Exception as e:
                    print("Failed to fetch quote sample:", e)
            except Exception as e:
                print("Failed to analyze pred index layout:", e)
        except Exception as e:
            print("Failed to load/inspect pred.pkl:", e)
        # --- end diagnostics ---

        # --- Use explicit signal DataFrame for strategy ---
        import pandas as _pd
        
        pred = recorder.load_object("pred.pkl")
        # to DataFrame with single column 'score'
        if isinstance(pred, _pd.Series):
            pred = pred.to_frame("score")
        elif isinstance(pred, _pd.DataFrame):
            if "score" not in pred.columns:
                pred = pred.rename(columns={pred.columns[0]: "score"})
        
        # ensure MultiIndex names are correct
        try:
            if isinstance(pred.index, _pd.MultiIndex):
                pred.index.set_names(["datetime", "instrument"], inplace=True)
        except Exception:
            pass
        
        # hand over explicit signal to strategy (will trigger real portfolio construction)
        port_analysis_config["strategy"]["kwargs"]["signal"] = pred
        # --- end explicit signal ---

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest with (model, dataset) signal (universe already restricted in handler)
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
