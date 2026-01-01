#!/usr/bin/env python3
"""Shared utility functions for qlib scripts."""
import os
import datetime
from typing import Dict, List, Optional

import pandas as pd

from qlib.data import D


def to_qlib_inst(x: str) -> str:
    s = str(x).lower()
    if s.startswith("hk."):
        s = s.split(".", 1)[1]
    if "." in s:
        s = s.split(".", 1)[0]
    if s.isdigit():
        s = s.zfill(5)
    return s.upper() + ".HK"


def calendar_last_day(today: datetime.date) -> str:
    cal = D.calendar(
        start_time=(today - datetime.timedelta(days=14)).strftime("%Y-%m-%d"),
        end_time=today.strftime("%Y-%m-%d"),
        freq="day",
    )
    return today.strftime("%Y-%m-%d") if len(cal) == 0 else cal[-1]


def load_chinese_name_map() -> Dict[str, str]:
    """Return mapping of instrument code -> Chinese name with robust key variants."""
    chinese_name: Dict[str, str] = {}
    try:
        name_paths = [
            os.path.join(os.path.expanduser("~"), ".qlib", "qlib_data", "hk_data", "boardlot", "chinese_name.txt"),
            r"C:\\Users\\kennethlao\\.qlib\\qlib_data\\hk_data\\boardlot\\chinese_name.txt",
        ]
        lines = []
        for p in name_paths:
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

        def add_name_keys(code: str, name: str):
            c = code.strip()
            c_low = c.lower()
            if c_low.startswith("hk."):
                num = c_low.split(".", 1)[1]
            elif c_low.endswith(".hk"):
                num = c_low.split(".", 1)[0]
            else:
                num = c_low
            num5 = num.zfill(5) if num.isdigit() else num
            keys = {
                num5 + ".hk",
                num5.upper() + ".HK",
                "hk." + num5,
                "HK." + num5,
                num5,
                num5.upper(),
            }
            for k in keys:
                chinese_name[k] = name

        for ln in lines:
            parts = ln.strip().split()
            if len(parts) >= 2:
                code = parts[0]
                name = " ".join(parts[1:])
                add_name_keys(code, name)
    except Exception:
        chinese_name = {}
    return chinese_name


def resolve_chinese(inst_code: str, chinese_map: Dict[str, str]) -> str:
    base = inst_code.split(".", 1)[0]
    candidates = [
        inst_code,
        inst_code.lower(),
        base,
        base.lower(),
        base.upper(),
        base.zfill(5),
        base.zfill(5).lower(),
        base.zfill(5).upper(),
        f"{base.zfill(5)}.hk",
        f"{base.zfill(5)}.HK",
        f"hk.{base.zfill(5)}",
        f"HK.{base.zfill(5)}",
    ]
    for k in candidates:
        if k in chinese_map:
            return chinese_map[k]
    return ""


def fetch_ohlcv(instruments: List[str], start_dt: str, end_dt: str):
    # try to include $open, $vwap, and $amount if available for Kronos input
    fields = ["$open", "$close", "$high", "$low", "$volume", "$vwap", "$amount"]
    try:
        df = D.features(instruments, fields, start_time=start_dt, end_time=end_dt, freq="day")
        # some providers may lack $vwap; align columns safely
        cols = list(df.columns)
        rename = {}
        # Expect at least 5: $open,$close,$high,$low,$volume,(optional)$vwap
        if len(cols) >= 5:
            # map first occurrences conservatively
            for c in cols:
                lc = str(c).lower()
                if "$open" in lc or lc.endswith("$open") or lc.endswith("open"):
                    rename[c] = "$open"
                elif "$close" in lc or lc.endswith("$close") or lc.endswith("close"):
                    rename[c] = "$close"
                elif "$high" in lc or lc.endswith("$high") or lc.endswith("high"):
                    rename[c] = "$high"
                elif "$low" in lc or lc.endswith("$low") or lc.endswith("low"):
                    rename[c] = "$low"
                elif "$volume" in lc or lc.endswith("$volume") or lc.endswith("volume"):
                    rename[c] = "$volume"
                elif "$vwap" in lc or lc.endswith("$vwap") or lc.endswith("vwap"):
                    rename[c] = "$vwap"
                elif "$amount" in lc or lc.endswith("$amount") or lc.endswith("amount"):
                    rename[c] = "$amount"
        df.columns = [rename.get(c, c) for c in cols]
    except Exception:
        df = D.features(instruments, ["$open", "$close", "$high", "$low", "$volume"], start_time=start_dt, end_time=end_dt, freq="day")
        df.columns = ["$open", "$close", "$high", "$low", "$volume"]
    return df


def fetch_base_close_vol(instruments: List[str], start_dt: str, end_dt: str):
    try:
        base = D.features(instruments, ["$close", "$volume"], start_time=start_dt, end_time=end_dt, freq="day")
        base.columns = ["$close", "$volume"]
        return base
    except Exception:
        return pd.DataFrame()


def compute_avg_dollar_volume(base, instruments: List[str], liq_window: int) -> Dict[str, float]:
    if base is None or getattr(base, "empty", False):
        return {inst: 0.0 for inst in instruments}

    def _tail_mean_dollar(df):
        df2 = df.dropna()
        if df2.empty:
            return 0.0
        dv = (df2["$close"] * df2["$volume"]).tail(liq_window)
        return float(dv.mean()) if len(dv) > 0 else 0.0

    try:
        return base.groupby(level="instrument").apply(_tail_mean_dollar).to_dict()
    except Exception:
        return {inst: 0.0 for inst in instruments}


def next_trading_day_from_future(provider_uri: str, current_day: str) -> Optional[str]:
    """Return the next trading day after current_day using day_future.txt; fallback to D.calendar."""
    try:
        base = os.path.expanduser(provider_uri)
        candidates = [
            os.path.join(base, "calendars", "day_future.txt"),
            os.path.join(base, "calendar", "day_future.txt"),
        ]
        lines: List[str] = []
        for p in candidates:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        lines = fh.read().splitlines()
                except Exception:
                    try:
                        with open(p, "r") as fh:
                            lines = fh.read().splitlines()
                    except Exception:
                        lines = []
                break
        if lines:
            cur_ts = pd.to_datetime(current_day)
            for ln in lines:
                try:
                    dt = pd.to_datetime(ln.strip())
                except Exception:
                    continue
                if dt > cur_ts:
                    return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    try:
        start = (pd.to_datetime(current_day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end = (pd.to_datetime(current_day) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        cal = D.calendar(start_time=start, end_time=end, freq="day")
        for d in cal:
            if pd.to_datetime(d) > pd.to_datetime(current_day):
                return pd.to_datetime(d).strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


__all__ = [
    "to_qlib_inst",
    "calendar_last_day",
    "load_chinese_name_map",
    "resolve_chinese",
    "fetch_ohlcv",
    "fetch_base_close_vol",
    "compute_avg_dollar_volume",
    "next_trading_day_from_future",
]
