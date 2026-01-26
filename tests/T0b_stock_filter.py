#!/usr/bin/env python3
"""
Stock filter for HK qlib data.
Supports combining multiple conditions: latest dollar amount, rolling average dollar amount,
turnover rate (using boardlot/issued_shares.txt), and price bounds. Can run standalone to
print the filtered instrument list.
"""
import argparse
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_HK
from qlib.data import D


def _ensure_qlib(provider_uri: str, region=REG_HK) -> None:
    try:
        from qlib.config import C
        cur = getattr(C, "provider_uri", None)
        # re-init if provider differs from requested path
        if cur:
            exp = os.path.expanduser(provider_uri)
            if os.path.abspath(str(cur)) == os.path.abspath(exp):
                return
    except Exception:
        pass
    qlib.init(provider_uri=os.path.expanduser(provider_uri), region=region)


def _last_trading_day() -> str:
    try:
        cal = D.calendar(start_time=None, end_time=None, freq="day")
    except Exception:
        try:
            cal = D.calendar(freq="day")
        except Exception:
            cal = []
    if len(cal) == 0:
        return pd.Timestamp.today().strftime("%Y-%m-%d")
    return pd.to_datetime(cal[-1]).strftime("%Y-%m-%d")


def _load_issued_shares(provider_uri: str) -> pd.Series:
    path = os.path.join(os.path.expanduser(provider_uri), "boardlot", "issued_shares.txt")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["instrument", "shares"]
        elif df.shape[1] == 1:
            df.columns = ["instrument"]
            df["shares"] = np.nan
        else:
            return pd.Series(dtype=float)
        df["instrument"] = df["instrument"].astype(str)
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
        df = df.dropna(subset=["shares"])
        return df.set_index("instrument")["shares"].astype(float)
    except Exception:
        return pd.Series(dtype=float)


def filter_instruments_by_conditions(
    instruments: Optional[Iterable[str]] = None,
    target_day: Optional[str] = None,
    provider_uri: str = "~/.qlib/qlib_data/hk_data",
    min_amount: Optional[float] = None,
    min_avg_amount: Optional[float] = None,
    avg_amount_window: int = 20,
    min_turnover: Optional[float] = None,
    min_avg_turnover: Optional[float] = None,
    turnover_window: int = 20,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    auto_init: bool = True,
    allow_missing_shares: bool = False,
) -> Tuple[List[str], dict]:
    if auto_init:
        _ensure_qlib(provider_uri)

    insts = list(instruments) if instruments is not None else D.instruments("all")
    if not insts:
        return [], {"orig_count": 0, "kept_count": 0, "pct": float("nan"), "sample": []}

    end_day = target_day or _last_trading_day()
    end_day = pd.to_datetime(end_day).strftime("%Y-%m-%d")

    max_window = max(
        1,
        avg_amount_window if min_avg_amount is not None else 1,
        turnover_window if min_avg_turnover is not None else 1,
    )
    start_day = (pd.to_datetime(end_day) - pd.Timedelta(days=max_window * 3)).strftime("%Y-%m-%d")

    feat = D.features(insts, ["$close", "$volume"], start_time=start_day, end_time=end_day, freq="day", disk_cache=True)
    if isinstance(feat, pd.DataFrame):
        feat = feat.copy()
    feat.columns = ["close", "volume"]

    if isinstance(feat.index, pd.MultiIndex):
        names = list(feat.index.names)
        if "datetime" in names and "instrument" in names:
            feat = feat.reorder_levels(["datetime", "instrument"]).sort_index()
    else:
        raise RuntimeError("D.features did not return a MultiIndex DataFrame")

    close_df = feat["close"].unstack(level="instrument")
    vol_df = feat["volume"].unstack(level="instrument")

    if close_df.empty or vol_df.empty:
        orig_count = close_df.columns.size if hasattr(close_df, "columns") else len(insts)
        return [], {"orig_count": orig_count, "kept_count": 0, "pct": float("nan"), "sample": []}

    close_df = close_df.sort_index()
    vol_df = vol_df.sort_index().fillna(0)

    amount_df = (close_df * vol_df).fillna(0)
    last_idx = close_df.index.max()
    minp_amount = max(5, avg_amount_window // 2)
    minp_turnover = max(5, turnover_window // 2)

    last_amount = amount_df.loc[last_idx]
    avg_amount = amount_df.rolling(window=avg_amount_window, min_periods=minp_amount).mean().loc[last_idx]

    shares = _load_issued_shares(provider_uri)
    shares = shares.reindex(vol_df.columns)
    turnover_last = None
    turnover_avg = None
    if min_turnover is not None or min_avg_turnover is not None:
        if shares.isna().all():
            raise RuntimeError("Turnover filter requested but issued_shares.txt is missing or empty")
        turnover_last = vol_df.loc[last_idx] / shares
        avg_vol = vol_df.rolling(window=turnover_window, min_periods=minp_turnover).mean().loc[last_idx]
        turnover_avg = avg_vol / shares

    price_last = close_df.loc[last_idx]

    mask = pd.Series(True, index=close_df.columns, dtype=bool)

    def _mask_ge(cur_mask: pd.Series, series: pd.Series, thresh: float, fill_missing_true: bool = False) -> pd.Series:
        cond = series.reindex(cur_mask.index)
        cond = cond >= float(thresh)
        cond = cond.fillna(True if fill_missing_true else False)
        return cur_mask & cond

    def _mask_le(cur_mask: pd.Series, series: pd.Series, thresh: float) -> pd.Series:
        cond = series.reindex(cur_mask.index)
        cond = cond <= float(thresh)
        cond = cond.fillna(False)
        return cur_mask & cond

    if min_amount is not None:
        mask = _mask_ge(mask, last_amount, min_amount)
    if min_avg_amount is not None:
        mask = _mask_ge(mask, avg_amount, min_avg_amount)
    if min_turnover is not None and turnover_last is not None:
        mask = _mask_ge(mask, turnover_last, min_turnover, fill_missing_true=allow_missing_shares)
    if min_avg_turnover is not None and turnover_avg is not None:
        mask = _mask_ge(mask, turnover_avg, min_avg_turnover, fill_missing_true=allow_missing_shares)
    if min_price is not None:
        mask = _mask_ge(mask, price_last, min_price)
    if max_price is not None:
        mask = _mask_le(mask, price_last, max_price)

    mask = mask.fillna(False)
    keep_insts = mask[mask].index.astype(str).tolist()

    orig_count = close_df.columns.size
    info = {
        "orig_count": orig_count,
        "kept_count": len(keep_insts),
        "pct": (len(keep_insts) / orig_count * 100.0) if orig_count > 0 else float("nan"),
        "sample": keep_insts[:20],
        "target_day": end_day,
        "min_amount": min_amount,
        "min_avg_amount": min_avg_amount,
        "min_turnover": min_turnover,
        "min_avg_turnover": min_avg_turnover,
        "min_price": min_price,
        "max_price": max_price,
    }

    metrics = pd.DataFrame(index=close_df.columns)
    metrics["amount"] = last_amount
    metrics["avg_amount"] = avg_amount
    metrics["price"] = price_last
    if turnover_last is not None:
        metrics["turnover"] = turnover_last
    if turnover_avg is not None:
        metrics["avg_turnover"] = turnover_avg
    info["metrics"] = metrics
    return keep_insts, info


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter HK instruments by liquidity/turnover/price conditions")
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/hk_data", help="qlib data path")
    parser.add_argument("--target-day", default=None, help="target trading day (YYYY-MM-DD); default last in calendar")
    parser.add_argument("--min-amount", type=float, default=None, help="minimum dollar amount on target day")
    parser.add_argument("--min-avg-amount", type=float, default=None, help="minimum rolling average dollar amount")
    parser.add_argument("--avg-amount-window", type=int, default=20, help="rolling window for average dollar amount")
    parser.add_argument("--min-turnover", type=float, default=None, help="minimum turnover on target day (volume / shares)")
    parser.add_argument(
        "--min-avg-turnover", type=float, default=None, help="minimum rolling average turnover (volume / shares)"
    )
    parser.add_argument("--turnover-window", type=int, default=20, help="rolling window for average turnover")
    parser.add_argument("--min-price", type=float, default=None, help="minimum last close price")
    parser.add_argument("--max-price", type=float, default=None, help="maximum last close price")
    parser.add_argument("--output", default=None, help="optional CSV path to save filtered instruments with metrics")
    parser.add_argument("--allow-missing-shares", action="store_true", help="keep stocks without shares data when using turnover filters")
    return parser.parse_args()


def main():
    args = _parse_args()
    keep_insts, info = filter_instruments_by_conditions(
        instruments=None,
        target_day=args.target_day,
        provider_uri=args.provider_uri,
        min_amount=args.min_amount,
        min_avg_amount=args.min_avg_amount,
        avg_amount_window=args.avg_amount_window,
        min_turnover=args.min_turnover,
        min_avg_turnover=args.min_avg_turnover,
        turnover_window=args.turnover_window,
        min_price=args.min_price,
        max_price=args.max_price,
        auto_init=True,
        allow_missing_shares=args.allow_missing_shares,
    )

    print(
        f"Filtered {info['kept_count']} / {info['orig_count']} instruments on {info['target_day']} "
        f"({info['pct']:.2f}% kept)"
    )
    print("Sample:", info.get("sample", []))
    print("Result list first 50 instruments:")
    print("\n".join(keep_insts[:50]))

    out_path = args.output
    if out_path and "metrics" in info:
        out_path = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        info["metrics"].loc[keep_insts].to_csv(out_path)
        print("Saved metrics to:", out_path)


if __name__ == "__main__":
    main()
