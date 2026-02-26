import copy
import os
import re
import pandas as pd

import qlib
from qlib.constant import REG_HK
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord

HK_BENCH = "800000.HK"

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
                    "start_time": "2005-01-01",
                    "end_time": "2025-12-17",
                    "fit_start_time": "2005-01-01",
                    "fit_end_time": "2021-12-31",
                    "instruments": "all",
                },
            },
            "segments": {
                "train": ("2005-01-01", "2018-12-31"),
                "valid": ("2019-01-01", "2020-12-31"),
                "test": ("2021-01-01", "2025-12-17"),
            },
        },
    },
}


def compute_liquid_instruments(liq_threshold=1_000_000, liq_window=20, handler_end_time=None):
    try:
        from qlib.tests import T0b_stock_filter as stock_filter  # type: ignore

        try:
            from qlib.config import C

            _provider_uri = getattr(C, "provider_uri", None)
        except Exception:
            _provider_uri = None

        all_insts = D.instruments("all")
        keep_insts, info = stock_filter.filter_instruments_by_conditions(
            instruments=all_insts,
            target_day=handler_end_time,
            provider_uri=_provider_uri or "~/.qlib/qlib_data/hk_data",
            min_avg_amount=liq_threshold,
            avg_amount_window=liq_window,
            auto_init=_provider_uri is None,
        )
        if "kept_count" not in info:
            info = {
                **info,
                "orig_count": len(all_insts),
                "kept_count": len(keep_insts),
                "pct": (len(keep_insts) / max(1, len(all_insts))) * 100,
            }
        return keep_insts, info
    except Exception:
        pass

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


def load_precomputed_indicators(
    instruments,
    start_time: str,
    end_time: str,
    fields=None,
    required_fields=None,
    freq: str = "day",
) -> pd.DataFrame:
    if fields is None:
        fields = ["DIF", "DEA", "MACD", "RSI", "KDJ_K", "KDJ_D", "KDJ_J", "MFI", "ROC", "EMA5", "EMA10", "EMA20", "EMA60", "EMA120"]
    if required_fields is None:
        required_fields = []

    if isinstance(instruments, (list, tuple, pd.Index, pd.Series)):
        instruments = [str(x).strip() for x in instruments if str(x).strip()]

    collected = []
    loaded_fields = []
    for f in fields:
        loaded = None
        expr_candidates = [f"${f}", f"${f.lower()}", f"${f.upper()}"]
        dedup_expr = []
        for expr in expr_candidates:
            if expr not in dedup_expr:
                dedup_expr.append(expr)
        for expr in dedup_expr:
            try:
                tmp = D.features(instruments, [expr], start_time=start_time, end_time=end_time, freq=freq, disk_cache=True)
                tmp.columns = [f]
                loaded = tmp
                break
            except Exception:
                continue
        if loaded is not None:
            collected.append(loaded)
            loaded_fields.append(f)

    if not collected:
        raise RuntimeError("No indicator field can be loaded from qlib. Check whether source fields exist (must use $-style fields).")

    ind_df = pd.concat(collected, axis=1)
    if isinstance(ind_df.index, pd.MultiIndex):
        names = list(ind_df.index.names)
        if "instrument" in names and "datetime" in names:
            ind_df = ind_df.reorder_levels(["datetime", "instrument"]).sort_index()
        elif len(names) == 2:
            ind_df.index = ind_df.index.set_names(["datetime", "instrument"])
            ind_df = ind_df.sort_index()

    missing_required = [f for f in required_fields if f not in ind_df.columns]
    if missing_required:
        raise RuntimeError(
            f"Required indicator fields missing: {missing_required}. Loaded fields: {loaded_fields}"
        )

    print(f"Loaded indicator fields: {loaded_fields}")
    if required_fields:
        non_na_stats = {f: int(ind_df[f].notna().sum()) for f in required_fields if f in ind_df.columns}
        print(f"Required indicator non-NaN counts: {non_na_stats}")

    return ind_df


def add_derived_indicators(ind_df: pd.DataFrame) -> pd.DataFrame:
    if ind_df is None or ind_df.empty:
        return ind_df

    if "MFI" in ind_df.columns:
        mfi = pd.to_numeric(ind_df["MFI"], errors="coerce")
        ind_df["MFI_MA10"] = mfi.groupby(level="instrument").transform(
            lambda s: s.rolling(window=10, min_periods=10).mean()
        )

    return ind_df


if __name__ == "__main__":
    task = copy.deepcopy(HK_GBDT_TASK)

    provider_uri = "~/.qlib/qlib_data/hk_data"
    GetData().qlib_data(target_dir=provider_uri, region=REG_HK, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_HK)

    try:
        cal = D.calendar(start_time=None, end_time=None, freq="day")
        last_day = pd.to_datetime(cal[-1]).strftime("%Y-%m-%d") if len(cal) > 0 else pd.Timestamp.today().strftime("%Y-%m-%d")
    except Exception:
        try:
            cal = D.calendar(freq="day")
            last_day = pd.to_datetime(cal[-1]).strftime("%Y-%m-%d") if len(cal) > 0 else pd.Timestamp.today().strftime("%Y-%m-%d")
        except Exception:
            last_day = pd.Timestamp.today().strftime("%Y-%m-%d")

    task["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = last_day
    task["dataset"]["kwargs"]["segments"]["test"] = (
        task["dataset"]["kwargs"]["segments"]["test"][0],
        last_day,
    )
    _last_trading_day = last_day

    handler_kwargs = task["dataset"]["kwargs"]["handler"]["kwargs"]
    keep_insts = []
    try:
        csv_path = os.path.abspath(os.path.join(os.getcwd(), "instrument_filtered_bt.csv"))
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"instrument_filtered_bt.csv not found at {csv_path}")
        df = pd.read_csv(csv_path)
        if df.empty:
            raise RuntimeError(f"instrument_filtered_bt.csv is empty at {csv_path}")
        inst_col = "instrument" if "instrument" in df.columns else df.columns[0]
        keep_insts = df[inst_col].astype(str).str.strip().tolist()
        print(f"Liquidity filter (CSV): kept {len(keep_insts)} instruments")
    except Exception as e:
        print("CSV liquidity load failed, switching to fallback:", e)
        keep_insts, info = compute_liquid_instruments(
            liq_threshold=30_000_000,
            liq_window=60,
            handler_end_time=handler_kwargs.get("end_time", None),
        )
        print(
            f"Liquidity filter (fallback): kept {info['kept_count']} / {info['orig_count']} instruments ({info['pct']:.2f}%)"
        )

    if len(keep_insts) > 0:
        task["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = keep_insts

    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    boardlot_path = os.path.join(os.path.expanduser(provider_uri), "boardlot", "boardlot.txt")
    if os.path.exists(boardlot_path):
        bl = pd.read_csv(
            boardlot_path,
            sep=r"\s+",
            header=None,
            names=["instrument", "board_lot"],
            dtype={"instrument": str},
        )
        bl = bl.set_index("instrument")["board_lot"].astype(float)
        factor_map = (bl / qlib.config.C.trade_unit).rename("$factor")
        start, end = "2021-01-01", _last_trading_day
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
        extra_quote = extra_quote[["$close", "$volume", "$factor"]]
        vol_expr = "0.2 * $volume"
        if vol_expr not in extra_quote.columns:
            extra_quote[vol_expr] = extra_quote["$volume"] * 0.2
    else:
        extra_quote = None

    macd_universe = keep_insts if len(keep_insts) > 0 else D.instruments("all")
    ind_start = task["dataset"]["kwargs"]["handler"]["kwargs"].get("start_time", "2005-01-01")
    filter_expr = "(DIF > DEA) & (DIF > 0) & (MFI > MFI_MA10)"
    derived_fields = {"MFI_MA10"}
    required_fields = sorted(
        {
            token
            for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", filter_expr)
            if token not in {"and", "or", "not", "True", "False"} and token not in derived_fields
        }
    )
    indicator_data = load_precomputed_indicators(
        instruments=macd_universe,
        start_time=ind_start,
        end_time=_last_trading_day,
        fields=["DIF", "DEA", "MACD", "RSI", "KDJ_K", "KDJ_D", "KDJ_J", "MFI", "ROC", "EMA5", "EMA10", "EMA20", "EMA60", "EMA120"],
        required_fields=required_fields,
        freq="day",
    )
    indicator_data = add_derived_indicators(indicator_data)
    missing_derived = [c for c in derived_fields if c not in indicator_data.columns]
    if missing_derived:
        raise RuntimeError(f"Derived indicator fields missing: {missing_derived}")
    print(
        "Derived indicator non-NaN counts:",
        {c: int(indicator_data[c].notna().sum()) for c in sorted(derived_fields)},
    )
    print(f"Prepared precomputed indicator data: {indicator_data.shape}")

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
            "class": "MACDTopkDropoutStrategy_v3",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 6,
                "n_drop": 1,
                "only_tradable": True,
                "forbid_all_trade_at_limit": True,
                "indicator_data": indicator_data,
                "filter_expr": filter_expr,
                "allow_no_filter_date": False,
            },
        },
        "backtest": {
            "start_time": "2021-01-01",
            "end_time": _last_trading_day,
            "account": 1000000,
            "benchmark": HK_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "volume_threshold": {
                    "all": ("current", "0.2 * $volume")
                },
                "extra_quote": extra_quote,
                "trade_unit": qlib.config.C.trade_unit,
            },
        },
    }

    with R.start(experiment_name="workflow"):
        _log_task = copy.deepcopy(task)
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

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        pred = recorder.load_object("pred.pkl")
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        elif isinstance(pred, pd.DataFrame) and "score" not in pred.columns:
            pred = pred.rename(columns={pred.columns[0]: "score"})
        if isinstance(pred.index, pd.MultiIndex):
            pred.index.set_names(["datetime", "instrument"], inplace=True)

        port_analysis_config["strategy"]["kwargs"]["signal"] = pred

        sar = SigAnaRecord(recorder)
        sar.generate()

        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
