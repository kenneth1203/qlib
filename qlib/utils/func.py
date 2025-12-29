#!/usr/bin/env python3
"""Shared utility functions for qlib scripts."""
import os
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
    "load_chinese_name_map",
    "resolve_chinese",
    "next_trading_day_from_future",
]
