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


def main(recorder_id, experiment_name, provider_uri, topk, min_listing_days=120, notifier: Optional[TelegramNotifier] = None):
    provider_uri = os.path.expanduser(provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_HK)

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
        liq_window = 20
        keep_insts, info = hkmod.compute_liquid_instruments(
            liq_threshold=5_000_000,
            liq_window=liq_window,
            handler_end_time=hkw.get("end_time", None),
        )
        print(f"Liquidity filter (predict): kept {info['kept_count']} / {info['orig_count']} instruments ({info['pct']:.2f}%)")
        print("Sample kept instruments:", info["sample"])
        if len(keep_insts) > 0:
            hkw["instruments"] = keep_insts

        # New listing filter after liquidity filter, before handler uses instruments
        if min_listing_days and min_listing_days > 0:
            insts = list(hkw.get("instruments", []))
            if insts:
                try:
                    import pandas as pd

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
            import pandas as pd

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
        import pandas as pd

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
            top = ss.sort_values(ascending=False).head(max(topk, 500))

            # --- Select top-k (liquidity + listing pre-filters already applied via handler) ---
            cand_insts = [to_qlib_inst(i) for i in top.index]
            selected = cand_insts[:topk]

            mapping = load_chinese_name_map()

            # Print selected top instruments with Chinese names and last-day volume
            score_df = pd.DataFrame({"score": list(top.values)}, index=cand_insts)
            score_df.index.name = "instrument"

            vol_map = {}
            # compute avg dollar volume over last `liq_window` trading bars for selected instruments
            try:
                if selected:
                    start_dt = (pd.to_datetime(target_day) - pd.Timedelta(days=liq_window * 3)).strftime("%Y-%m-%d")
                    base = D.features(selected, ["$close", "$volume"], start_time=start_dt, end_time=target_day, freq="day")
                    base.columns = ["$close", "$volume"]

                    def _tail_mean_dollar(df):
                        df2 = df.dropna()
                        if df2.empty:
                            return 0.0
                        dv = (df2["$close"] * df2["$volume"]).tail(liq_window)
                        return float(dv.mean()) if len(dv) > 0 else 0.0

                    vol_map = (
                        base.groupby(level="instrument").apply(_tail_mean_dollar).to_dict()
                    )
            except Exception:
                vol_map = {}

            # Build a tidy view and print as an aligned table
            rows = []
            for inst in selected:
                score = float(score_df.loc[inst, "score"]) if inst in score_df.index else 0.0
                mk = inst.split(".", 1)[0].zfill(5) + ".hk"
                name = resolve_chinese(inst, mapping)
                avg_dollar = int(round(vol_map.get(inst, 0)))
                rows.append({"id": mk, "name": name, "score": score, "avg_dollar": avg_dollar})

            view_df = pd.DataFrame(rows)
            if not view_df.empty:
                # prepare formatted string columns for aligned printing
                view_df["_score_s"] = view_df["score"].map(lambda x: f"{x:.6f}")
                view_df["_avg_s"] = view_df["avg_dollar"].map(lambda x: f"{int(x):,}")
                # determine column widths (display width-aware)
                cols = ["id", "name", "_score_s", "_avg_s"]
                widths = {}
                for c in cols:
                    header_name = "score" if c == "_score_s" else ("avg_dollar" if c == "_avg_s" else c)
                    max_cell = 0
                    if not view_df.empty:
                        max_cell = int(view_df[c].map(lambda v: _disp_width(v)).max())
                    widths[c] = max(_disp_width(header_name), max_cell)
                # header (use display-aware padding)
                hdr = f"{_pad_right('id', widths['id'])}  {_pad_right('name', widths['name'])}  {_pad_left('score', widths['_score_s'])}  {_pad_left('avg_dollar', widths['_avg_s'])}"
                print(f"\nTop {topk} instruments for {target_day}:")
                print(hdr)
                # rows (display-aware padding)
                lines = [hdr]
                for _, r in view_df.iterrows():
                    id_s = r['id'] if pd.notna(r['id']) else ''
                    name_s = r['name'] if pd.notna(r['name']) else ''
                    score_s = r['_score_s'] if pd.notna(r['_score_s']) else ''
                    avg_s = r['_avg_s'] if pd.notna(r['_avg_s']) else ''
                    row_s = f"{_pad_right(id_s, widths['id'])}  {_pad_right(name_s, widths['name'])}  {_pad_left(score_s, widths['_score_s'])}  {_pad_left(avg_s, widths['_avg_s'])}"
                    print(row_s)
                    lines.append(row_s)
    except Exception:
        print("Prediction result:", pred)

    # save prediction locally (dated file) and keep legacy name for compatibility
    day_str = pd.to_datetime(target_day).strftime("%Y%m%d")
    out_dated = f"pred_{day_str}_{recorder_id}.pkl"
    with open(out_dated, "wb") as f:
        pickle.dump(pred, f)
    print(f"Saved prediction to {out_dated}")

    out_legacy = f"pred_today_{recorder_id}.pkl"
    try:
        with open(out_legacy, "wb") as f:
            pickle.dump(pred, f)
        print(f"Saved legacy prediction to {out_legacy}")
    except Exception as e:
        print(f"Warning: failed to write legacy pred file {out_legacy}: {e}")

    if notifier:
        preview = ", ".join(selected) if 'selected' in locals() else ""
        table_lines = []
        if 'lines' in locals():
            table_lines = lines
        if table_lines:
            payload_lines = ["```", f"T1 selection for {msg_day}", f"recorder_id: {recorder_id}"] + table_lines + ["```"]
        else:
            payload_lines = [f"T1 selection for {msg_day}", f"recorder_id: {recorder_id}"]
        notifier.send("\n".join(payload_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder_id", default=None, help="recorder/run id (run id). If omitted, use latest run folder under ./mlruns")
    parser.add_argument("--experiment_name", default="workflow", help="experiment name")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/hk_data", help="qlib data dir")
    parser.add_argument("--topk", type=int, default=20, help="print top-k instruments")
    parser.add_argument("--min_listing_days", type=int, default=120, help="minimum trading days since listing; set 0 to disable filter")
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