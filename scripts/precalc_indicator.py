#!/usr/bin/env python3
"""
Pre-calculate daily indicators (MACD, KDJ, RSI, EMA5/10/20/60/120) and store them into qlib bin features
so they can be read directly via D.features(instruments, ["DIF", "DEA", "MACD", "RSI", "KDJ_K", "EMA20"], ...).

The script computes indicators for all instruments and uses DumpDataUpdate to write/update the feature bins.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_HK
from qlib.data import D
from dump_bin import DumpDataUpdate


FIELDS = [
    "DIF",
    "DEA",
    "MACD",
    "DIF_PREV",
    "DEA_PREV",
    "RSI",
    "KDJ_K",
    "KDJ_D",
    "KDJ_J",
    "EMA5",
    "EMA10",
    "EMA20",
    "EMA60",
    "EMA120",
]


def _ensure_qlib(provider_uri: str):
    qlib.init(provider_uri=os.path.expanduser(provider_uri), region=REG_HK)


def _compute_indicators_for_instrument(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.sort_index()
    close = sub["$close"].astype(float)
    high = sub["$high"].astype(float)
    low = sub["$low"].astype(float)

    out = pd.DataFrame(index=sub.index)

    # MACD 12/26/9
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=9, adjust=False).mean()
    out["DIF"] = dif
    out["DEA"] = dea
    out["MACD"] = dif - dea
    out["DIF_PREV"] = dif.shift(1)
    out["DEA_PREV"] = dea.shift(1)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=7).mean()
    avg_loss = loss.rolling(window=14, min_periods=7).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
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

    # EMAs
    for w in (5, 10, 20, 60, 120):
        out[f"EMA{w}"] = close.ewm(span=w, adjust=False).mean()

    return out


def _chunk_list(lst: Iterable[str], size: int):
    # Convert to list to ensure slicing works even if the input is an iterator or mapping view.
    seq = list(lst)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _merge_indicators_into_source(indicator_dir: Path, source_dir: Path):
    """Merge per-instrument indicator CSVs into existing source OHLCV CSVs."""
    indicator_dir = indicator_dir.expanduser().resolve()
    source_dir = source_dir.expanduser().resolve()
    merged = 0

    for ind_file in indicator_dir.glob("*.csv"):
        sym = ind_file.stem
        src_file = source_dir.joinpath(f"{sym}.csv")
        if not src_file.exists():
            print(f"Skip merge for {sym}: source file not found at {src_file}")
            continue

        ind_df = pd.read_csv(ind_file)
        base_df = pd.read_csv(src_file)

        # Normalize date for robust merge
        if "date" in base_df.columns:
            base_df["date"] = pd.to_datetime(base_df["date"])
        if "date" in ind_df.columns:
            ind_df["date"] = pd.to_datetime(ind_df["date"])

        merge_cols = ["symbol", "date"]
        ind_cols = [c for c in ind_df.columns if c not in merge_cols]
        merged_df = base_df.merge(ind_df[merge_cols + ind_cols], on=merge_cols, how="left")

        # Ensure column order: base first, then indicators
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
        merged += 1
    print(f"Merge complete. Files updated: {merged}")


def precalc_to_bin(
    provider_uri: str,
    start_time: str,
    end_time: str,
    chunk: int,
    tmp_dir: str,
    per_instr_dir: str = None,
    reuse_existing: bool = False,
    merge_into_source: bool = False,
    source_dir: str = None,
):
    _ensure_qlib(provider_uri)
    instruments_cfg = D.instruments("all")
    # D.instruments may return a config dict (market/filter_pipe) rather than a list; expand when needed.
    instruments = (
        list(instruments_cfg)
        if isinstance(instruments_cfg, (list, tuple))
        else D.list_instruments(instruments_cfg, as_list=True)
    )
    if not instruments:
        print("No instruments found.")
        return
    print(f"Found {len(instruments)} instruments; chunk={chunk}")

    csv_dir = os.path.abspath(os.path.expanduser(per_instr_dir or tmp_dir))
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = None if per_instr_dir else os.path.join(csv_dir, "indicators.csv")
    print(f"Writing CSVs to {csv_dir}")

    first_write = True
    has_written = False

    if reuse_existing:
        if per_instr_dir:
            has_written = any(Path(csv_dir).glob("*.csv"))
        else:
            has_written = os.path.exists(csv_path)
        if not has_written:
            print("No existing CSVs found to reuse; will compute.")
        else:
            print("Reusing existing indicator CSVs; skip recompute.")

    if not has_written:
        for batch in _chunk_list(instruments, chunk):
            df = D.features(batch, ["$close", "$high", "$low"], start_time=start_time, end_time=end_time, freq="day", disk_cache=True)
            if df is None or df.empty:
                print(f"Batch skipped: no data for instruments {batch[:3]}... (size {len(batch)})")
                continue
            if not isinstance(df.index, pd.MultiIndex):
                print("Unexpected index type, skipping batch")
                continue
            parts = []
            for inst, sub in df.groupby(level=0):
                try:
                    sub_idxed = sub.droplevel(0)
                    ind = _compute_indicators_for_instrument(sub_idxed)
                    ind = ind.reset_index().rename(columns={"datetime": "date"})
                    ind.insert(0, "symbol", inst)
                    parts.append(ind)
                except Exception as e:
                    print(f"Indicator computation failed for {inst}: {e}")
                    continue
            if not parts:
                continue
            out_df = pd.concat(parts, axis=0, ignore_index=True)
            if per_instr_dir:
                for inst, inst_df in out_df.groupby("symbol"):
                    file_path = os.path.join(csv_dir, f"{inst}.csv")
                    mode = "a" if os.path.exists(file_path) else "w"
                    header = mode == "w"
                    inst_df.to_csv(file_path, mode=mode, header=header, index=False)
                    has_written = True
                print(f"Processed batch of {len(batch)} instruments; per-instrument CSVs updated")
            else:
                mode = "w" if first_write else "a"
                header = first_write
                out_df.to_csv(csv_path, mode=mode, header=header, index=False)
                first_write = False
                has_written = True
                print(f"Processed batch of {len(batch)} instruments; rows written: {len(out_df)}")

    if not has_written:
        print("No indicator data generated.")
        return

    if merge_into_source:
        if not per_instr_dir:
            print("merge_into_source requires --per-instrument-dir to be set.")
            return
        target_dir = Path(source_dir or "~/.qlib/stock_data/source/hk_data")
        _merge_indicators_into_source(Path(per_instr_dir), target_dir)
        return

    print("Writing indicators into qlib bins via DumpDataUpdate...")
    dumper = DumpDataUpdate(
        data_path=csv_dir,
        qlib_dir=os.path.expanduser(provider_uri),
        freq="day",
        date_field_name="date",
        symbol_field_name="symbol",
        file_suffix=".csv",
        exclude_fields="",
        include_fields=";".join(FIELDS),
    )
    dumper.dump()
    print("Done. Indicators stored in qlib features.")


def main():
    parser = argparse.ArgumentParser(description="Precompute indicators into qlib bin features")
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/hk_data", help="qlib data path")
    parser.add_argument("--start-time", default=None, help="start date (YYYY-MM-DD); default: all")
    parser.add_argument("--end-time", default=None, help="end date (YYYY-MM-DD); default: all")
    parser.add_argument("--chunk", type=int, default=200, help="instrument batch size per fetch")
    parser.add_argument("--tmp-dir", default=None, help="temp dir for intermediate CSV (default: system temp)")
    parser.add_argument(
        "--per-instrument-dir",
        default=None,
        help="directory to store per-instrument indicator CSVs (if set, files will be named <symbol>.csv)",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse existing indicator CSVs in the output dir and skip recomputation",
    )
    parser.add_argument(
        "--merge-into-source",
        action="store_true",
        help="merge indicator CSVs into existing source OHLCV CSVs instead of dumping bins",
    )
    parser.add_argument(
        "--source-dir",
        default="~/.qlib/stock_data/source/hk_data",
        help="source OHLCV CSV directory (used with --merge-into-source)",
    )
    args = parser.parse_args()

    tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="t0c_ind_")
    precalc_to_bin(
        args.provider_uri,
        args.start_time,
        args.end_time,
        args.chunk,
        tmp_dir,
        args.per_instrument_dir,
        args.reuse_existing,
        args.merge_into_source,
        args.source_dir,
    )


if __name__ == "__main__":
    main()
