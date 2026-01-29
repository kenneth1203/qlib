#!/usr/bin/env python3
r"""Resample Yahoo 1d CSV to weekly/monthly/yearly CSV.

Defaults:
	- src: C:\Users\kennethlao\.qlib\stock_data\source\hk_data
	- 1w dst: C:\Users\kennethlao\.qlib\stock_data\source\hk_data_1w
	- 1mo dst: C:\Users\kennethlao\.qlib\stock_data\source\hk_data_1mo
	- 1y dst: C:\Users\kennethlao\.qlib\stock_data\source\hk_data_1y

Rules:
	- Weekly: open=first day open, close=last day close, high=max, low=min, volume=sum.
	- Monthly/Yearly: same aggregation rule.
	- Date represents period start (Monday / 1st day of month / Jan 1).
	- Columns kept: symbol,date,open,high,low,close,volume
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd

SRC_DEFAULT = r"C:\Users\kennethlao\.qlib\stock_data\source\hk_data"
DST_1W_DEFAULT = r"C:\Users\kennethlao\.qlib\stock_data\source\hk_data_1w"
DST_1M_DEFAULT = r"C:\Users\kennethlao\.qlib\stock_data\source\hk_data_1mo"
DST_1Y_DEFAULT = r"C:\Users\kennethlao\.qlib\stock_data\source\hk_data_1y"

KEEP_COLS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def _ensure_dir(path: str) -> Path:
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
	try:
		df = pd.read_csv(path)
	except Exception:
		return None
	if df is None or df.empty:
		return None
	cols = [c for c in KEEP_COLS if c in df.columns]
	if not cols or "date" not in cols or "symbol" not in cols:
		return None
	df = df[cols].copy()
	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df = df.dropna(subset=["date"])
	if df.empty:
		return None
	return df.sort_values("date")


def _aggregate_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
	df = df.sort_values("date").set_index("date")
	agg_map = {
		"open": "first",
		"high": "max",
		"low": "min",
		"close": "last",
		"volume": "sum",
	}
	res = df.resample(freq, label="left", closed="left").agg(agg_map)
	# attach symbol
	if "symbol" in df.columns:
		res["symbol"] = df["symbol"].resample(freq, label="left", closed="left").first()
	res = res.dropna(subset=["open", "high", "low", "close"], how="any")
	res = res.reset_index().rename(columns={"index": "date"})
	for col in KEEP_COLS:
		if col not in res.columns:
			res[col] = pd.NA
	return res[KEEP_COLS]


def _periodic_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
	if freq == "M":
		return _aggregate_ohlcv(df, "MS")
	if freq == "Y":
		return _aggregate_ohlcv(df, "YS-JAN")
	return pd.DataFrame(columns=KEEP_COLS)


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(out_path, index=False)


def resample_dir(src_dir: str, freq: str, dst_dir: str) -> None:
	src = Path(src_dir)
	if not src.exists():
		raise FileNotFoundError(f"Source dir not found: {src}")
	_ensure_dir(dst_dir)

	for csv_path in src.glob("*.csv"):
		df = _load_csv(csv_path)
		if df is None or df.empty:
			continue

		if freq == "1w":
			out_df = _aggregate_ohlcv(df, "W-MON")
		elif freq == "1mo":
			out_df = _periodic_ohlcv(df, "M")
		elif freq == "1y":
			out_df = _periodic_ohlcv(df, "Y")
		else:
			raise ValueError("freq must be one of: 1w, 1mo, 1y")

		if out_df is None or out_df.empty:
			continue

		_write_csv(out_df, Path(dst_dir) / csv_path.name)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--src", default=SRC_DEFAULT, help="source dir for 1d yahoo csv")
	parser.add_argument("--freq", choices=["1w", "1mo", "1y", "all"], default="all")
	parser.add_argument("--dst", default=None, help="output dir for given freq (ignored if freq=all)")
	args = parser.parse_args()

	if args.freq == "all":
		resample_dir(args.src, "1w", DST_1W_DEFAULT)
		resample_dir(args.c, "1mo", DST_1M_DEFAULT)
		resample_dir(args.src, "1y", DST_1Y_DEFAULT)
	else:
		if not args.dst:
			if args.freq == "1w":
				dst = DST_1W_DEFAULT
			elif args.freq == "1mo":
				dst = DST_1M_DEFAULT
			else:
				dst = DST_1Y_DEFAULT
		else:
			dst = args.dst
		resample_dir(args.src, args.freq, dst)


if __name__ == "__main__":
	main()
