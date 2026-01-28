from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
from qlib.workflow import R
from qlib.utils import init_instance_by_config
import pandas as pd
import warnings
from qlib.tests.data import GetData
import plotly.io as pio
import plotly.graph_objs as go
import os

import qlib
from qlib.constant import REG_HK

HK_BENCH = "800000.HK"  # ^HSI 被存成 800000.HK

# 簡易的 HK Alpha158+LGB 配置（可以依需要調整）
HK_GBDT_TASK = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "max_depth": 6,
            "num_leaves": 64,
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
                    "start_time": "2015-01-01",
                    "end_time": "2025-12-31",
                    "fit_start_time": "2015-01-01",
                    "fit_end_time": "2023-12-31",
                    "instruments": "all",  # 若要限定標的可改成 ["800000.HK"] 或自訂清單
                },
            },
            "segments": {
                "train": ("2015-01-01", "2021-12-31"),
                "valid": ("2022-01-01", "2022-12-31"),
                "test": ("2023-01-01", "2025-12-03"),
            },
        },
    },
}

if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/hk_data"  # HK 資料目錄
    GetData().qlib_data(target_dir=provider_uri, region=REG_HK, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_HK)

    dataset = init_instance_by_config(HK_GBDT_TASK["dataset"])

    # Auto-detect latest recorder_id from ./mlruns and use it
    try:
        mlruns_dir = os.path.join(".", "mlruns")
        rec_id = None
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
                rec_id = os.path.basename(os.path.normpath(latest))
                print(f"Auto-detected recorder_id from mlruns: {rec_id}")
        if rec_id:
            recorder = R.get_recorder(recorder_id=rec_id, experiment_name="workflow")
        else:
            recorder = R.get_recorder(experiment_name="workflow")
    except Exception:
        recorder = R.get_recorder(experiment_name="workflow")
    print(recorder)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    figs = analysis_position.report_graph(report_normal_df, show_notebook=False) or []
    # risk analysis graphs
    risk_figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False) or []

    # Top50 most profitable stocks (approx by position value change)
    pos_figs = []
    try:
        pos_obj = positions
        # normalize positions into a MultiIndex DataFrame
        if isinstance(pos_obj, dict):
            rows = []
            for dt, pos in pos_obj.items():
                try:
                    dt_ts = pd.to_datetime(dt)
                except Exception:
                    dt_ts = dt
                if hasattr(pos, "position") and isinstance(pos.position, dict):
                    iter_dict = pos.position
                elif isinstance(pos, dict):
                    iter_dict = pos
                else:
                    continue
                for inst, info in iter_dict.items():
                    if inst in ("cash", "now_account_value", "cash_delay"):
                        continue
                    amt = None
                    w = None
                    if isinstance(info, dict):
                        amt = info.get("amount")
                        w = info.get("weight")
                    else:
                        try:
                            amt = float(info)
                        except Exception:
                            amt = None
                    rows.append({"datetime": dt_ts, "instrument": inst, "amount": amt, "weight": w})
            pos_df = pd.DataFrame(rows)
            if not pos_df.empty:
                pos_df = pos_df.set_index(["datetime", "instrument"]).sort_index()
        elif isinstance(pos_obj, pd.DataFrame):
            pos_df = pos_obj.copy()
            if not isinstance(pos_df.index, pd.MultiIndex) and "datetime" in pos_df.columns and "instrument" in pos_df.columns:
                pos_df = pos_df.set_index(["datetime", "instrument"])
        else:
            pos_df = pd.DataFrame()

        if not pos_df.empty and isinstance(pos_df.index, pd.MultiIndex) and pos_df.index.nlevels == 2:
            pos_df.index = pos_df.index.set_names(["datetime", "instrument"])

            amount_series = None
            value_df = None
            close_wide = None
            if "amount" in pos_df.columns:
                amount_series = pos_df["amount"].fillna(0)
            elif "position" in pos_df.columns:
                amount_series = pos_df["position"].fillna(0)

            if amount_series is not None:
                amt_wide = amount_series.unstack(level="instrument").fillna(0)
                insts = amt_wide.columns.tolist()
                dt_start = amt_wide.index.min()
                dt_end = amt_wide.index.max()
                close_wide = D.features(insts, ["$close"], start_time=dt_start, end_time=dt_end, freq="day", disk_cache=True)
                close_wide = close_wide["$close"].unstack(level="instrument").reindex(amt_wide.index).fillna(method="ffill")
                value_df = (amt_wide * close_wide).fillna(0)
            elif "weight" in pos_df.columns and isinstance(report_normal_df, pd.DataFrame) and "account" in report_normal_df:
                w_wide = pos_df["weight"].fillna(0).unstack(level="instrument").fillna(0)
                acct = report_normal_df.get("account", pd.Series(dtype=float))
                try:
                    acct.index = pd.to_datetime(acct.index)
                except Exception:
                    pass
                acct = acct.reindex(w_wide.index).fillna(method="ffill")
                value_df = w_wide.mul(acct, axis=0)
                insts = w_wide.columns.tolist()
                dt_start = w_wide.index.min()
                dt_end = w_wide.index.max()
                try:
                    close_wide = D.features(insts, ["$close"], start_time=dt_start, end_time=dt_end, freq="day", disk_cache=True)
                    close_wide = close_wide["$close"].unstack(level="instrument").reindex(w_wide.index).fillna(method="ffill")
                except Exception:
                    close_wide = None

            results = []
            trade_rows = []
            if value_df is not None and not value_df.empty:
                for inst in value_df.columns:
                    series = value_df[inst].fillna(0)
                    # use absolute position value to handle shorts
                    hold_mask = series.abs() > 0
                    if not hold_mask.any():
                        continue
                    first_idx = hold_mask.idxmax()
                    last_idx = hold_mask[::-1].idxmax()
                    enters = (hold_mask.astype(int).diff().fillna(hold_mask.astype(int)) == 1).sum()
                    start_val = float(series.loc[first_idx]) if first_idx in series.index else float("nan")
                    end_val = float(series.loc[last_idx]) if last_idx in series.index else float("nan")
                    profit_amt = end_val - start_val if pd.notna(start_val) and pd.notna(end_val) else float("nan")
                    profit_rate = (profit_amt / start_val) if (pd.notna(profit_amt) and start_val not in (0, float("nan"))) else float("nan")
                    results.append(
                        {
                            "instrument": inst,
                            "first_buy": first_idx,
                            "last_sell": last_idx,
                            "trades": int(enters),
                            "profit_rate": profit_rate,
                            "profit_amt": profit_amt,
                        }
                    )

                    # reconstruct simple round-trip trades by holding intervals
                    diff = hold_mask.astype(int).diff().fillna(hold_mask.astype(int))
                    starts = diff[diff == 1].index.tolist()
                    ends = diff[diff == -1].index.tolist()
                    if hold_mask.iloc[0]:
                        starts = [hold_mask.index[0]] + starts
                    if len(ends) < len(starts):
                        ends = ends + [hold_mask.index[-1]]
                    for s_dt, e_dt in zip(starts, ends):
                        try:
                            s_val = float(series.loc[s_dt])
                            e_val = float(series.loc[e_dt])
                        except Exception:
                            continue
                        buy_px = None
                        sell_px = None
                        vol_amt = None
                        pnl = e_val - s_val
                        rate = float("nan")
                        try:
                            if close_wide is not None and s_dt in close_wide.index and inst in close_wide.columns:
                                buy_px = float(close_wide.loc[s_dt, inst])
                            if close_wide is not None and e_dt in close_wide.index and inst in close_wide.columns:
                                sell_px = float(close_wide.loc[e_dt, inst])
                        except Exception:
                            pass
                        try:
                            if amount_series is not None:
                                vol_amt = float(amount_series.unstack(level="instrument").reindex(series.index).loc[s_dt, inst])
                        except Exception:
                            vol_amt = None
                        # Prefer price * volume for PnL when available
                        if vol_amt is not None and buy_px is not None and sell_px is not None:
                            pnl = (sell_px - buy_px) * vol_amt
                            cost = buy_px * vol_amt
                            rate = (pnl / cost) if cost else float("nan")
                        else:
                            if s_val not in (0, float("nan")):
                                rate = (pnl / s_val) if s_val else float("nan")
                        trade_rows.append(
                            {
                                "instrument": inst,
                                "buy_time": s_dt,
                                "sell_time": e_dt,
                                "profit_rate": rate,
                                "profit_amt": pnl,
                                "buy_price": buy_px,
                                "sell_price": sell_px,
                                "volume": vol_amt,
                            }
                        )

            if results:
                res_df = pd.DataFrame(results)
                res_df = res_df.sort_values("profit_amt", ascending=False).head(50)
                fig_table = go.Figure(
                    data=[
                        go.Table(
                            header=dict(values=["Rank", "股票", "最初買入時間", "最後賣出時間", "中間交易次數", "最終獲利率", "最終獲利金額"], fill_color="lightgrey"),
                            cells=dict(
                                values=[
                                    (res_df.index + 1).tolist(),
                                    res_df["instrument"].tolist(),
                                    [str(v) for v in res_df["first_buy"]],
                                    [str(v) for v in res_df["last_sell"]],
                                    res_df["trades"].tolist(),
                                    [f"{x:.2%}" if pd.notna(x) else "" for x in res_df["profit_rate"]],
                                    [f"{x:,.0f}" for x in res_df["profit_amt"]],
                                ]
                            ),
                        )
                    ]
                )
                fig_table.update_layout(title="Top50 最賺錢股票", height=720)
                pos_figs = [fig_table]

            if trade_rows:
                trade_df = pd.DataFrame(trade_rows)

                # Full trade history for the single most profitable stock
                try:
                    if results:
                        top_stock = res_df.iloc[0]["instrument"]
                        top_trades = trade_df[trade_df["instrument"] == top_stock].copy()
                        if not top_trades.empty:
                            top_trades = top_trades.sort_values("buy_time")
                            fig_top_stock = go.Figure(
                                data=[
                                    go.Table(
                                        header=dict(
                                            values=["Rank", "股票", "買入時間", "賣出時間", "買入價", "賣出價", "成交量", "獲利率", "獲利金額"],
                                            fill_color="lightgrey",
                                        ),
                                        cells=dict(
                                            values=[
                                                (top_trades.index + 1).tolist(),
                                                top_trades["instrument"].tolist(),
                                                [str(v) for v in top_trades["buy_time"]],
                                                [str(v) for v in top_trades["sell_time"]],
                                                ["" if pd.isna(x) else f"{x:.3f}" for x in top_trades.get("buy_price", [])],
                                                ["" if pd.isna(x) else f"{x:.3f}" for x in top_trades.get("sell_price", [])],
                                                ["" if pd.isna(x) else f"{x:,.0f}" for x in top_trades.get("volume", [])],
                                                [f"{x:.2%}" if pd.notna(x) else "" for x in top_trades["profit_rate"]],
                                                [f"{x:,.0f}" for x in top_trades["profit_amt"]],
                                            ]
                                        ),
                                    )
                                ]
                            )
                            fig_top_stock.update_layout(title="Top1 股票交易明細", height=720)
                            pos_figs = list(pos_figs) + [fig_top_stock]
                except Exception:
                    pass

                best_df = trade_df.sort_values("profit_amt", ascending=False).head(50)
                fig_trades_best = go.Figure(
                    data=[
                        go.Table(
                            header=dict(values=["Rank", "股票", "買入時間", "賣出時間", "買入價", "賣出價", "成交量", "獲利率", "獲利金額"], fill_color="lightgrey"),
                            cells=dict(
                                values=[
                                    (best_df.index + 1).tolist(),
                                    best_df["instrument"].tolist(),
                                    [str(v) for v in best_df["buy_time"]],
                                    [str(v) for v in best_df["sell_time"]],
                                    ["" if pd.isna(x) else f"{x:.3f}" for x in best_df.get("buy_price", [])],
                                    ["" if pd.isna(x) else f"{x:.3f}" for x in best_df.get("sell_price", [])],
                                    ["" if pd.isna(x) else f"{x:,.0f}" for x in best_df.get("volume", [])],
                                    [f"{x:.2%}" if pd.notna(x) else "" for x in best_df["profit_rate"]],
                                    [f"{x:,.0f}" for x in best_df["profit_amt"]],
                                ]
                            ),
                        )
                    ]
                )
                fig_trades_best.update_layout(title="Top50 最賺錢交易", height=720)

                worst_df = trade_df.sort_values("profit_amt", ascending=True).head(50)
                fig_trades_worst = go.Figure(
                    data=[
                        go.Table(
                            header=dict(values=["Rank", "股票", "買入時間", "賣出時間", "買入價", "賣出價", "成交量", "獲利率", "獲利金額"], fill_color="lightgrey"),
                            cells=dict(
                                values=[
                                    (worst_df.index + 1).tolist(),
                                    worst_df["instrument"].tolist(),
                                    [str(v) for v in worst_df["buy_time"]],
                                    [str(v) for v in worst_df["sell_time"]],
                                    ["" if pd.isna(x) else f"{x:.3f}" for x in worst_df.get("buy_price", [])],
                                    ["" if pd.isna(x) else f"{x:.3f}" for x in worst_df.get("sell_price", [])],
                                    ["" if pd.isna(x) else f"{x:,.0f}" for x in worst_df.get("volume", [])],
                                    [f"{x:.2%}" if pd.notna(x) else "" for x in worst_df["profit_rate"]],
                                    [f"{x:,.0f}" for x in worst_df["profit_amt"]],
                                ]
                            ),
                        )
                    ]
                )
                fig_trades_worst.update_layout(title="Top50 最虧錢交易", height=720)

                pos_figs = list(pos_figs) + [fig_trades_best, fig_trades_worst]
    except Exception:
        pos_figs = []

    # Top50 longest holding time (days) table
    top_hold_figs = []
    try:
        hold_map = {}
        # positions can be dict(datetime -> Position) or DataFrame
        if isinstance(positions, dict):
            for dt, pos in positions.items():
                # normalize datetime
                try:
                    dt_ts = pd.to_datetime(dt)
                except Exception:
                    dt_ts = dt
                # Position-like object with .position dict
                if hasattr(pos, "position") and isinstance(pos.position, dict):
                    for inst, info in pos.position.items():
                        if inst in ("cash", "now_account_value", "cash_delay"):
                            continue
                        cnt = 0
                        if isinstance(info, dict):
                            # typical count key: 'count_day' (bar == "day"), or others like 'count_{bar}'
                            cnt = info.get("count_day", None)
                            if cnt is None:
                                for k, v in info.items():
                                    if isinstance(k, str) and k.startswith("count_"):
                                        cnt = v
                                        break
                            if cnt is None:
                                # fallback to stored 'amount' as proxy (rare)
                                cnt = info.get("amount", 0) or 0
                        else:
                            # if info itself is numeric, treat as amount
                            try:
                                cnt = int(info)
                            except Exception:
                                cnt = 0
                        hold_map[inst] = max(hold_map.get(inst, 0), int(cnt or 0))
        elif isinstance(positions, pd.DataFrame):
            df = positions.copy()
            # if counts stored as column 'count_day' or 'count_*'
            count_cols = [c for c in df.columns if c.startswith("count_")]
            if len(count_cols) > 0:
                # use max across count columns per instrument
                if isinstance(df.index, pd.MultiIndex) and "instrument" in df.index.names:
                    inst_idx = df.index.get_level_values("instrument")
                    df2 = df.reset_index()
                    df2["max_count"] = df2[count_cols].max(axis=1)
                    gm = df2.groupby("instrument")["max_count"].max()
                    for inst, cnt in gm.items():
                        hold_map[inst] = int(cnt or 0)
            elif "count_day" in df.columns:
                gm = df["count_day"].groupby(level="instrument").max()
                for inst, cnt in gm.items():
                    hold_map[inst] = int(cnt or 0)
    
        if len(hold_map) > 0:
            hold_df = (
                pd.Series(hold_map, name="max_hold_days")
                .rename_axis("instrument")
                .reset_index()
                .sort_values("max_hold_days", ascending=False)
                .reset_index(drop=True)
            )
            top50 = hold_df.head(50)
            # Build Plotly table
            fig_table = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=["Rank", "Instrument", "Max Hold Days"], fill_color="lightgrey"),
                        cells=dict(
                            values=[
                                (top50.index + 1).tolist(),
                                top50["instrument"].tolist(),
                                top50["max_hold_days"].tolist(),
                            ]
                        ),
                    )
                ]
            )
            fig_table.update_layout(title="Top50 最長持股時間 (天)", height=700)
            top_hold_figs = [fig_table]
    except Exception:
        top_hold_figs = []

    # account (principal) curve
    account_figs = []
    try:
        if "account" in report_normal_df.columns:
            acct = report_normal_df["account"].copy()
            # ensure datetime index
            try:
                idx = pd.to_datetime(acct.index)
            except Exception:
                idx = acct.index
            fig_account = go.Figure(
                data=go.Scatter(x=idx, y=acct.values, mode="lines", name="Account Value")
            )
            fig_account.update_layout(title="本金曲線 (Account Value)", xaxis_title="Date", yaxis_title="Account", height=480)
            account_figs = [fig_account]
    except Exception:
        account_figs = []


    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    ic_figs = analysis_position.score_ic_graph(pred_label, show_notebook=False) or []
    perf_figs = analysis_model.model_performance_graph(pred_label, show_notebook=False) or []

    # Combine all generated figures into a single HTML file (account first)
    all_figs = list(account_figs) + list(figs) + list(pos_figs) + list(top_hold_figs) + list(risk_figs) + list(ic_figs) + list(perf_figs)
    fragments = []
    out_dir = os.getcwd()
    for i, fig in enumerate(all_figs):
        # fragment for combined file; include plotlyjs only once
        if i == 0:
            frag = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
        else:
            frag = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        fragments.append(f"<div id=\"figure_{i}\">{frag}</div>")

    combined_html = "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Combined Report</title></head><body>"
    combined_html += "\n<hr/>\n".join(fragments)
    combined_html += "</body></html>"

    combined_path = os.path.join(out_dir, "combined_report.html")
    with open(combined_path, "w", encoding="utf-8") as fp:
        fp.write(combined_html)

    print(f"Saved combined report to: {combined_path}")
    # cleanup old individual HTML files (keep combined_report.html)
    import glob

    patterns = ["report_*.html", "risk_report_*.html", "ic_report_*.html", "perf_report_*.html", "report_part_*.html"]
    for pat in patterns:
        for fpath in glob.glob(os.path.join(out_dir, pat)):
            if os.path.abspath(fpath) == os.path.abspath(combined_path):
                continue
            try:
                os.remove(fpath)
            except Exception:
                pass