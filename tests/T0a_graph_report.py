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

    # Align instruments with backtest (instrument_filtered.csv)
    try:
        csv_path = os.path.abspath(os.path.join(os.getcwd(), "instrument_filtered.csv"))
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                inst_col = "instrument" if "instrument" in df.columns else df.columns[0]
                keep_insts = df[inst_col].astype(str).tolist()
                if keep_insts:
                    HK_GBDT_TASK["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = keep_insts
    except Exception:
        pass

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

    # Per-stock trade summary & per-trade details
    pos_figs = []
    html_blocks = []
    trade_df = pd.DataFrame()
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

            trade_rows = []
            if amount_series is not None and close_wide is not None:
                amt_wide = amount_series.unstack(level="instrument").reindex(close_wide.index).fillna(0)
                for inst in amt_wide.columns:
                    amt_s = amt_wide[inst].fillna(0)
                    if (amt_s.abs() > 0).sum() == 0:
                        continue
                    price_s = None
                    if close_wide is not None and inst in close_wide.columns:
                        price_s = close_wide[inst].reindex(amt_s.index)
                    lot_queue = []  # FIFO lots: (buy_time, buy_price, amount)
                    prev_amt = 0.0
                    for dt, amt in amt_s.items():
                        try:
                            amt = float(amt)
                        except Exception:
                            continue
                        diff = amt - prev_amt
                        if diff > 0:
                            buy_px = None
                            try:
                                if price_s is not None and dt in price_s.index:
                                    buy_px = float(price_s.loc[dt])
                            except Exception:
                                buy_px = None
                            lot_queue.append([dt, buy_px, diff])
                        elif diff < 0:
                            sell_amt = -diff
                            sell_px = None
                            try:
                                if price_s is not None and dt in price_s.index:
                                    sell_px = float(price_s.loc[dt])
                            except Exception:
                                sell_px = None
                            remain_amt = amt
                            while sell_amt > 0 and lot_queue:
                                lot_dt, lot_px, lot_amt = lot_queue[0]
                                take_amt = min(lot_amt, sell_amt)
                                pnl = float("nan")
                                cost = None
                                rate = float("nan")
                                if lot_px is not None and sell_px is not None:
                                    pnl = (sell_px - lot_px) * take_amt
                                    cost = lot_px * take_amt
                                    rate = (pnl / cost) if cost else float("nan")
                                hold_days = None
                                try:
                                    hold_days = (pd.to_datetime(dt) - pd.to_datetime(lot_dt)).days
                                except Exception:
                                    hold_days = None
                                trade_rows.append(
                                    {
                                        "instrument": inst,
                                        "buy_time": lot_dt,
                                        "sell_time": dt,
                                        "profit_rate": rate,
                                        "profit_amt": pnl,
                                        "buy_price": lot_px,
                                        "sell_price": sell_px,
                                        "volume": take_amt,
                                        "remain_volume": remain_amt,
                                        "cost": cost,
                                        "hold_days": hold_days,
                                    }
                                )
                                lot_amt -= take_amt
                                sell_amt -= take_amt
                                if lot_amt <= 0:
                                    lot_queue.pop(0)
                                else:
                                    lot_queue[0][2] = lot_amt
                        prev_amt = amt
            elif value_df is not None and not value_df.empty:
                for inst in value_df.columns:
                    series = value_df[inst].fillna(0)
                    # use absolute position value to handle shorts
                    hold_mask = series.abs() > 0
                    if not hold_mask.any():
                        continue
                    # reconstruct simple round-trip trades by holding intervals
                    diff = hold_mask.astype(int).diff().fillna(hold_mask.astype(int))
                    starts = diff[diff == 1].index.tolist()
                    ends = diff[diff == -1].index.tolist()
                    if hold_mask.iloc[0]:
                        if not starts or starts[0] != hold_mask.index[0]:
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
                        cost = None
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
                        remain_amt = None
                        try:
                            if amount_series is not None:
                                remain_amt = float(amount_series.unstack(level="instrument").reindex(series.index).loc[e_dt, inst])
                        except Exception:
                            remain_amt = None
                        # Prefer price * volume for PnL when available
                        if vol_amt is not None and buy_px is not None and sell_px is not None:
                            pnl = (sell_px - buy_px) * vol_amt
                            cost = buy_px * vol_amt
                            rate = (pnl / cost) if cost else float("nan")
                        else:
                            if s_val not in (0, float("nan")):
                                cost = abs(s_val)
                                rate = (pnl / s_val) if s_val else float("nan")
                        hold_days = None
                        try:
                            hold_days = (pd.to_datetime(e_dt) - pd.to_datetime(s_dt)).days
                        except Exception:
                            hold_days = None
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
                                "remain_volume": remain_amt,
                                "cost": cost,
                                "hold_days": hold_days,
                            }
                        )

            if trade_rows:
                trade_df = pd.DataFrame(trade_rows)

                # All traded stocks summary (same columns as old Top50, plus hold time)
                try:
                    grp = trade_df.groupby("instrument", dropna=True)
                    res_df = grp.agg(
                        first_buy=("buy_time", "min"),
                        last_sell=("sell_time", "max"),
                        trades=("instrument", "count"),
                        profit_amt=("profit_amt", "sum"),
                        total_cost=("cost", "sum"),
                        total_hold_days=("hold_days", "sum"),
                    ).reset_index()
                    res_df["profit_rate"] = res_df.apply(
                        lambda r: (r["profit_amt"] / r["total_cost"]) if r["total_cost"] else float("nan"),
                        axis=1,
                    )
                    res_df = res_df.sort_values("profit_amt", ascending=False).reset_index(drop=True)

                    # Build interactive HTML table; clicking trade count shows a modal with per-trade details
                    def _fmt_num(x, fmt):
                        try:
                            return fmt.format(x)
                        except Exception:
                            return ""

                    trade_detail_map = {}
                    for inst in res_df["instrument"].astype(str).tolist():
                        sub = trade_df[trade_df["instrument"].astype(str) == inst].copy()
                        sub = sub.sort_values("buy_time")
                        rows = []
                        for _, r in sub.iterrows():
                            rows.append(
                                {
                                    "buy_time": str(r.get("buy_time", "")),
                                    "sell_time": str(r.get("sell_time", "")),
                                    "buy_price": _fmt_num(r.get("buy_price"), "{:.3f}"),
                                    "sell_price": _fmt_num(r.get("sell_price"), "{:.3f}"),
                                    "volume": _fmt_num(r.get("volume"), "{:,.0f}"),
                                    "remain_volume": _fmt_num(r.get("remain_volume"), "{:,.0f}"),
                                    "profit_rate": _fmt_num(r.get("profit_rate"), "{:.2%}"),
                                    "profit_amt": _fmt_num(r.get("profit_amt"), "{:,.0f}"),
                                    "hold_days": str(int(r.get("hold_days") or 0)),
                                }
                            )
                        trade_detail_map[inst] = rows

                    summary_rows = []
                    for idx, r in res_df.iterrows():
                        inst = str(r["instrument"])
                        summary_rows.append(
                            "<tr>"
                            f"<td>{idx + 1}</td>"
                            f"<td>{inst}</td>"
                            f"<td>{r['first_buy']}</td>"
                            f"<td>{r['last_sell']}</td>"
                            f"<td><a href='#' class='trade-detail' data-inst='{inst}'>{int(r['trades'])}</a></td>"
                            f"<td>{_fmt_num(r['profit_rate'], '{:.2%}') if pd.notna(r['profit_rate']) else ''}</td>"
                            f"<td>{_fmt_num(r['profit_amt'], '{:,.0f}')}</td>"
                            f"<td>{int(r['total_hold_days'] or 0)}</td>"
                            "</tr>"
                        )

                    import json as _json

                    trade_detail_json = _json.dumps(trade_detail_map, ensure_ascii=False)

                    summary_html = """
                    <div id="trade-summary">
                        <h2>交易股票清單（回測期間所有交易過）</h2>
                        <div style="max-height:720px; overflow:auto; border:1px solid #ddd; border-radius:6px;">
                            <table border="1" cellspacing="0" cellpadding="6" style="width:100%; border-collapse:collapse; font-family:Arial, sans-serif; font-size:12px;">
                                <thead>
                                    <tr style="background:#eee;">
                                        <th>Rank</th>
                                        <th>股票</th>
                                        <th>最初買入時間</th>
                                        <th>最後賣出時間</th>
                                        <th>中間交易次數</th>
                                        <th>最終獲利率</th>
                                        <th>最終獲利金額</th>
                                        <th>總持股天數</th>
                                    </tr>
                                </thead>
                                <tbody>
                    """
                    summary_html += "".join(summary_rows)
                    summary_html += """
                            </tbody>
                        </table>
                    </div>
                    </div>

                    <div id="trade-modal" style="display:none; position:fixed; z-index:9999; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,0.5);">
                      <div style="background:#fff; margin:5% auto; padding:20px; width:90%; max-width:1200px; max-height:80%; overflow:auto; border-radius:6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                          <h3 id="trade-modal-title">交易明細</h3>
                          <button id="trade-modal-close" style="font-size:16px;">關閉</button>
                        </div>
                        <div id="trade-modal-content"></div>
                      </div>
                    </div>

                    <script>
                      window.tradeDetails = {trade_detail_json};
                      function renderTradeTable(inst) {
                        const rows = window.tradeDetails[inst] || [];
                                                let html = "<table border='1' cellspacing='0' cellpadding='6' style='width:100%; border-collapse:collapse;'>";
                                                html += "<thead><tr style='background:#eee;'><th>買入時間</th><th>賣出時間</th><th>買入價</th><th>賣出價</th><th>成交量</th><th>剩餘持倉</th><th>獲利率</th><th>獲利金額</th><th>持股天數</th></tr></thead><tbody>";
                        for (const r of rows) {
                                                    html += `<tr><td>${r.buy_time}</td><td>${r.sell_time}</td><td>${r.buy_price}</td><td>${r.sell_price}</td><td>${r.volume}</td><td>${r.remain_volume}</td><td>${r.profit_rate}</td><td>${r.profit_amt}</td><td>${r.hold_days}</td></tr>`;
                        }
                        html += "</tbody></table>";
                        return html;
                      }
                      document.addEventListener('DOMContentLoaded', function () {
                        document.querySelectorAll('.trade-detail').forEach(function (el) {
                          el.addEventListener('click', function (e) {
                            e.preventDefault();
                            const inst = el.getAttribute('data-inst');
                            document.getElementById('trade-modal-title').textContent = inst + " 交易明細";
                            document.getElementById('trade-modal-content').innerHTML = renderTradeTable(inst);
                            document.getElementById('trade-modal').style.display = 'block';
                          });
                        });
                        document.getElementById('trade-modal-close').addEventListener('click', function () {
                          document.getElementById('trade-modal').style.display = 'none';
                        });
                        document.getElementById('trade-modal').addEventListener('click', function (e) {
                          if (e.target.id === 'trade-modal') {
                            document.getElementById('trade-modal').style.display = 'none';
                          }
                        });
                      });
                    </script>
                    """.replace("{trade_detail_json}", trade_detail_json)

                    html_blocks.append(summary_html)
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

    # trade win/loss summary table
    winloss_figs = []
    try:
        total_trades = int(len(trade_df)) if isinstance(trade_df, pd.DataFrame) else 0
        if total_trades > 0 and "profit_amt" in trade_df.columns:
            win_cnt = int((trade_df["profit_amt"] > 0).sum())
            loss_cnt = int((trade_df["profit_amt"] < 0).sum())
        else:
            win_cnt = 0
            loss_cnt = 0
        win_rate = (win_cnt / total_trades) if total_trades else 0.0
        avg_daily_trades = 0.0
        try:
            if total_trades > 0 and "buy_time" in trade_df.columns:
                buy_dates = pd.to_datetime(trade_df["buy_time"], errors="coerce").dt.normalize()
                day_count = int(buy_dates.dropna().nunique())
                avg_daily_trades = (total_trades / day_count) if day_count > 0 else 0.0
        except Exception:
            avg_daily_trades = 0.0
        fig_winloss = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["總交易次數", "賺錢交易次數", "虧錢交易次數", "勝率", "平均每日交易次數"], fill_color="lightgrey"),
                    cells=dict(values=[[total_trades], [win_cnt], [loss_cnt], [f"{win_rate:.2%}"], [f"{avg_daily_trades:.2f}"]]),
                )
            ]
        )
        fig_winloss.update_layout(title="交易勝率摘要", height=240)
        winloss_figs = [fig_winloss]
    except Exception:
        winloss_figs = []


    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    ic_figs = analysis_position.score_ic_graph(pred_label, show_notebook=False) or []
    perf_figs = analysis_model.model_performance_graph(pred_label, show_notebook=False) or []

    # Combine all generated figures into a single HTML file (account first)
    all_figs = list(account_figs) + list(winloss_figs) + list(figs) + list(html_blocks) + list(pos_figs) + list(risk_figs) + list(ic_figs) + list(perf_figs)
    fragments = []
    out_dir = os.getcwd()
    for i, fig in enumerate(all_figs):
        # fragment for combined file; include plotlyjs only once
        if isinstance(fig, str):
            fragments.append(f"<div id=\"figure_{i}\">{fig}</div>")
            continue
        if i == 0:
            frag = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
        else:
            frag = pio.to_html(fig, include_plotlyjs=False, full_html=False)
        fragments.append(f"<div id=\"figure_{i}\">{frag}</div>")

    combined_html = "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Combined Report</title>"
    combined_html += "<style>body, table, th, td, div { user-select: text; -webkit-user-select: text; -ms-user-select: text; }</style>"
    combined_html += "</head><body>"
    combined_html += "\n<hr/>\n".join(fragments)
    combined_html += "</body></html>"

    rec_id_safe = rec_id or "unknown"
    combined_path = os.path.join(out_dir, f"combined_report_{rec_id_safe}.html")
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