#!/usr/bin/env python3
"""
Pre-calculate daily indicators and store them into CSV files under hk_data_indicator.

Default flow: read CSVs in hk_data_indicator, compute indicators, and write back in-place.
"""
import argparse
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


FIELDS = [
    "DIF",
    "DEA",
    "MACD",
    "RSI",
    "KDJ_K",
    "KDJ_D",
    "KDJ_J",
    "MFI",
    "ROC",
    "EMA5",
    "EMA10",
    "EMA20",
    "EMA60",
    "EMA120",
]


def _compute_indicators_for_instrument(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_index()
    close = sub["$close"].astype(float)
    high = sub["$high"].astype(float)
    low = sub["$low"].astype(float)
    volume = sub["$volume"].astype(float)

    out = pd.DataFrame(index=sub.index)

    # MACD 12/26/9
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=9, adjust=False).mean()
    out["DIF"] = dif
    out["DEA"] = dea
    out["MACD"] = (dif - dea) * 2
    # RSI(6) using EMA-like smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 6, min_periods=6, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 6, min_periods=6, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI"] = 100 - (100 / (1 + rs))

    # KDJ(9)
    low_n = low.rolling(window=9, min_periods=4).min()
    high_n = high.rolling(window=9, min_periods=4).max()
    rsv = (close - low_n) / (high_n - low_n)
    rsv = rsv.replace([np.inf, -np.inf], np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    out["KDJ_K"] = k
    out["KDJ_D"] = d
    out["KDJ_J"] = j

    # MFI(14)
    tp = (high + low + close) / 3.0
    mf = tp * volume
    tp_prev = tp.shift(1)
    pos_mf = mf.where(tp > tp_prev, 0.0)
    neg_mf = mf.where(tp < tp_prev, 0.0)
    pos_sum = pos_mf.rolling(window=14, min_periods=7).sum()
    neg_sum = neg_mf.rolling(window=14, min_periods=7).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    out["MFI"] = 100 - (100 / (1 + mfr))

    # ROC(10)
    roc_n = 10
    roc = (close - close.shift(roc_n)) / close.shift(roc_n).replace(0, np.nan) * 100
    out["ROC"] = roc

    # EMAs
    for w in (5, 10, 20, 60, 120):
        out[f"EMA{w}"] = close.ewm(span=w, adjust=False).mean()

    return out.fillna(0.0)


def _compute_indicators_from_source_df(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.set_index("date")
    required = {"close", "high", "low", "volume"}
    if not required.issubset(set(work.columns)):
        missing = ", ".join(sorted(required - set(work.columns)))
        raise ValueError(f"missing required columns: {missing}")
    work = work.rename(
        columns={
            "close": "$close",
            "high": "$high",
            "low": "$low",
            "volume": "$volume",
        }
    )
    ind = _compute_indicators_for_instrument(work)
    ind = ind.reset_index().rename(columns={"index": "date"})
    return ind


def precalc_inplace(indicator_dir: str):
    ind_dir = Path(indicator_dir).expanduser().resolve()
    if not ind_dir.exists():
        print(f"Indicator dir not found: {ind_dir}")
        return
    files = list(ind_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {ind_dir}")
        return

    for src_file in files:
        try:
            base_df = pd.read_csv(src_file)
            if base_df.empty:
                continue
            if "date" not in base_df.columns:
                print(f"Skip {src_file.name}: missing date column")
                continue
            base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
            base_df = base_df.dropna(subset=["date"])
            if base_df.empty:
                continue
            if "symbol" not in base_df.columns:
                base_df.insert(0, "symbol", src_file.stem)

            base_df = base_df.drop(columns=[c for c in FIELDS if c in base_df.columns])
            ind_df = _compute_indicators_from_source_df(base_df)
            merge_cols = ["date"]
            ind_cols = [c for c in ind_df.columns if c not in merge_cols]
            merged_df = base_df.merge(ind_df[merge_cols + ind_cols], on=merge_cols, how="left")
            if "date" in merged_df.columns:
                merged_df["date"] = merged_df["date"].dt.strftime("%Y-%m-%d")

            ordered_cols = [
                col
                for col in [
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "adjclose",
                    "dividends",
                ]
                if col in merged_df.columns
            ] + [c for c in FIELDS if c in merged_df.columns]
            merged_df = merged_df[[c for c in ordered_cols if c in merged_df.columns]]

            merged_df.to_csv(src_file, index=False)
        except Exception as e:
            print(f"Failed to process {src_file.name}: {e}")
            continue

    print(f"Done. Indicators computed in-place for {ind_dir}")


def main():
    parser = argparse.ArgumentParser(description="Precompute indicators into CSVs in a folder")
    parser.add_argument(
        "--indicator-dir",
        default="C:/Users/kennethlao/.qlib/stock_data/source/hk_data_indicator",
        help="directory containing hk_data_indicator CSVs",
    )
    args = parser.parse_args()

    precalc_inplace(args.indicator_dir)


if __name__ == "__main__":
    main()
