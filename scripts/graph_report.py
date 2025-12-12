from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
from qlib.workflow import R
from qlib.utils import init_instance_by_config
import pandas as pd
import warnings
from qlib.tests.data import GetData
import plotly.io as pio
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

    recorder = R.get_recorder(recorder_id="006df08a0658485aa1aff8d2da16e92a", experiment_name="workflow")
    print(recorder)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    figs = analysis_position.report_graph(report_normal_df, show_notebook=False) or []
    # risk analysis graphs
    risk_figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False) or []


    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    ic_figs = analysis_position.score_ic_graph(pred_label, show_notebook=False) or []
    perf_figs = analysis_model.model_performance_graph(pred_label, show_notebook=False) or []

    # Combine all generated figures into a single HTML file
    all_figs = list(figs) + list(risk_figs) + list(ic_figs) + list(perf_figs)
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