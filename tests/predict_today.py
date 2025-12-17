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

import qlib
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.constant import REG_HK


def main(recorder_id, experiment_name, provider_uri, topk):
    provider_uri = os.path.expanduser(provider_uri)
    qlib.init(provider_uri=provider_uri, region=REG_HK)

    # import the HK example dataset config
    try:
        # try normal import first
        from qlib.tests import workflow_by_code_HK as hkmod  # type: ignore
    except Exception:
        # fallback: load module from the tests directory file path
        import importlib.util
        test_dir = os.path.dirname(__file__)
        wk_path = os.path.join(test_dir, "workflow_by_code_HK.py")
        if not os.path.exists(wk_path):
            raise RuntimeError(f"Cannot find workflow_by_code_HK.py at {wk_path}")
        spec = importlib.util.spec_from_file_location("qlib.tests.workflow_by_code_HK", wk_path)
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
    cal = D.calendar(start_time=(today_dt - datetime.timedelta(days=14)).strftime("%Y-%m-%d"),
                     end_time=today_dt.strftime("%Y-%m-%d"), freq="day")
    if len(cal) == 0:
        target_day = today_dt.strftime("%Y-%m-%d")
    else:
        target_day = cal[-1]

    print("target_day (from qlib calendar):", target_day)

    ds_cfg = copy.deepcopy(hkmod.HK_GBDT_TASK["dataset"])

    # Apply the same liquidity filter used in backtest to keep universes aligned
    try:
        hkw = ds_cfg["kwargs"]["handler"]["kwargs"]
        liq_window = 20
        keep_insts, info = hkmod.compute_liquid_instruments(
            liq_threshold=1_000_000,
            liq_window=liq_window,
            handler_end_time=hkw.get("end_time", None),
        )
        print(f"Liquidity filter (predict): kept {info['kept_count']} / {info['orig_count']} instruments ({info['pct']:.2f}%)")
        print("Sample kept instruments:", info["sample"])
        if len(keep_insts) > 0:
            hkw["instruments"] = keep_insts
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

            # --- Select top-k (liquidity pre-filter already applied via handler) ---
            def to_qlib_inst(x: str) -> str:
                s = str(x).lower()
                if s.startswith("hk."):
                    s = s.split(".", 1)[1]
                if "." in s:
                    s = s.split(".", 1)[0]
                if s.isdigit():
                    s = s.zfill(5)
                return s.upper() + ".HK"

            cand_insts = [to_qlib_inst(i) for i in top.index]
            selected = cand_insts[:topk]

            # Load Chinese name mapping (optional display)
            mapping = {}
            cand_paths = [
                r"C:\\Users\\kennethlao\\.qlib\\qlib_data\\hk_data\\boardlot\\chinese_name.txt",
                os.path.join(os.path.expanduser("~"), ".qlib", "qlib_data", "hk_data", "boardlot", "chinese_name.txt"),
            ]
            lines = []
            for p in cand_paths:
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

            for ln in lines:
                parts = ln.strip().split()
                if len(parts) >= 2:
                    key = parts[0]
                    name = " ".join(parts[1:])
                    k = key.lower()
                    if k.startswith("hk."):
                        k = k.split(".", 1)[1] + ".hk"
                    elif "." not in k and k.isdigit():
                        k = k.zfill(5) + ".hk"
                    mapping[k] = name

            # Print selected top instruments with Chinese names and last-day volume
            score_df = pd.DataFrame({"score": list(top.values)}, index=cand_insts)
            score_df.index.name = "instrument"

            # compute avg dollar volume over last `liq_window` trading bars for selected instruments
            try:
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

            print("\nTop instruments (id, name, score, avg_dollar_volume):")
            for inst in selected:
                score = float(score_df.loc[inst, "score"]) if inst in score_df.index else 0.0
                mk = inst.split(".", 1)[0].zfill(5) + ".hk"
                name = mapping.get(mk, "")
                avg_dollar = int(round(vol_map.get(inst, 0)))
                print(f"{mk}\t{name}\t{score:.6f}\t{avg_dollar:,}")
    except Exception:
        print("Prediction result:", pred)

    # save prediction locally
    out_path = f"pred_today_{recorder_id}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(pred, f)
    print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recorder_id",default="84171777c98d42dbabe60bd7369198b6", help="recorder/run id (run id)")
    parser.add_argument("--experiment_name", default="workflow", help="experiment name")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/hk_data", help="qlib data dir")
    parser.add_argument("--topk", type=int, default=20, help="print top-k instruments")
    args = parser.parse_args()
    main(args.recorder_id, args.experiment_name, args.provider_uri, args.topk)
    
