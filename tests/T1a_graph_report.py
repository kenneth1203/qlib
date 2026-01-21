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

    recorder = R.get_recorder(recorder_id="3e3febba13ee43d4bf40cd27122b96cc", experiment_name="workflow")
    print(recorder)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    figs = analysis_position.report_graph(report_normal_df, show_notebook=False) or []
    # risk analysis graphs
    risk_figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False) or []

    # position concentration (top holdings share and HHI)
    pos_figs = []
    try:
        pos_obj = positions
        # case A: dict of Position objects {datetime: Position}
        if isinstance(pos_obj, dict):
            rows = []
            for dt, pos in pos_obj.items():
                try:
                    dt_ts = pd.to_datetime(dt)
                except Exception:
                    dt_ts = dt
                # if it's a Position-like object with get_stock_weight_dict
                if hasattr(pos, "get_stock_weight_dict"):
                    weight_dict = pos.get_stock_weight_dict()
                    for inst, w in weight_dict.items():
                        rows.append({"datetime": dt_ts, "instrument": inst, "weight": w})
                # if it's a plain dict mapping inst->info
                elif isinstance(pos, dict):
                    for inst, info in pos.items():
                        if inst in ("cash", "now_account_value", "cash_delay"):
                            continue
                        w = None
                        if isinstance(info, dict):
                            w = info.get("weight") or info.get("amount")
                        else:
                            w = info
                        rows.append({"datetime": dt_ts, "instrument": inst, "weight": w})
            if len(rows) > 0:
                pos_df = pd.DataFrame(rows).set_index(["datetime", "instrument"]).sort_index()
            else:
                pos_df = pd.DataFrame()
        # case B: already a DataFrame saved in recorder
        elif isinstance(pos_obj, pd.DataFrame):
            pos_df = pos_obj.copy()
            # ensure multi-index
            if not isinstance(pos_df.index, pd.MultiIndex) and "datetime" in pos_df.columns and "instrument" in pos_df.columns:
                pos_df = pos_df.set_index(["datetime", "instrument"])
        else:
            pos_df = pd.DataFrame()
    
        if not pos_df.empty and isinstance(pos_df.index, pd.MultiIndex) and pos_df.index.nlevels == 2:
            # ensure names
            names = list(pos_df.index.names)
            if names[0] != "datetime":
                pos_df.index = pos_df.index.set_names(["datetime", "instrument"])
            # prefer 'weight' column, fallback to 'amount'
            if "weight" in pos_df.columns:
                w_series = pos_df["weight"]
            elif "position" in pos_df.columns:
                w_series = pos_df["position"]
            elif "amount" in pos_df.columns:
                w_series = pos_df["amount"]
            else:
                # if weight stored as values in the index->dict, try to create from index level values
                w_series = None
    
            if w_series is not None:
                w = w_series.unstack(level=1).fillna(0)
                def topk_cum(row, k=10):
                    vals = row.abs().sort_values(ascending=False)
                    if len(vals) == 0:
                        return 0.0
                    return vals.cumsum().iloc[min(k, len(vals)) - 1]
                top3 = w.apply(topk_cum, k=3, axis=1)
                top5 = w.apply(topk_cum, k=5, axis=1)
                top10 = w.apply(topk_cum, k=10, axis=1)
                hhi = (w.pow(2)).sum(axis=1)
    
                fig_pos = go.Figure()
                fig_pos.add_trace(go.Scatter(x=top3.index, y=top3.values, mode="lines", name="Top3 CumWt"))
                fig_pos.add_trace(go.Scatter(x=top5.index, y=top5.values, mode="lines", name="Top5 CumWt"))
                fig_pos.add_trace(go.Scatter(x=top10.index, y=top10.values, mode="lines", name="Top10 CumWt"))
                fig_pos.add_trace(go.Scatter(x=hhi.index, y=hhi.values, mode="lines", name="HHI"))
                fig_pos.update_layout(
                    title="持股集中度 (Top-N 累計權重 / HHI)",
                    xaxis_title="Date",
                    yaxis_title="Weight / HHI",
                    height=520,
                )
                pos_figs = [fig_pos]
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