# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import importlib
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import fire
import requests
import numpy as np
import pandas as pd
from loguru import logger
from yahooquery import Ticker
from dateutil.tz import tzlocal

import qlib
from qlib.data import D
from qlib.tests.data import GetData
from qlib.utils import code_to_fname, fname_to_code, exists_qlib_data
from qlib.utils.func import next_trading_day_from_future
from qlib.constant import REG_CN as REGION_CN

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from dump_bin import DumpDataUpdate
from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize
from data_collector.utils import (
    deco_retry,
    get_calendar_list,
    get_hs_stock_symbols,
    get_hk_stock_symbols,
    yahoo_hk_symbol_candidates,
    get_us_stock_symbols,
    get_in_stock_symbols,
    get_br_stock_symbols,
    generate_minutes_calendar_from_daily,
    calc_adjusted_price,
)

INDEX_BENCH_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get?secid=1.{index_code}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58&klt=101&fqt=0&beg={begin}&end={end}"


class YahooCollector(BaseCollector):
    retry = 1  # Configuration attribute.  How many times will it try to re-request the data if the network fails.

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
        symbols: [str, Iterable[str]] = None,
        update_mode: str = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [1min, 1d], default 1min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        super(YahooCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
            symbols=symbols,
            update_mode=update_mode,
            **kwargs,
        )

        self.init_datetime()

    def init_datetime(self):
        if self.interval == self.INTERVAL_1min:
            self.start_datetime = max(self.start_datetime, self.DEFAULT_START_DATETIME_1MIN)
        elif self.interval == self.INTERVAL_1d:
            pass
        else:
            raise ValueError(f"interval error: {self.interval}")

        self.start_datetime = self.convert_datetime(self.start_datetime, self._timezone)
        self.end_datetime = self.convert_datetime(self.end_datetime, self._timezone)

    @staticmethod
    def convert_datetime(dt: [pd.Timestamp, datetime.date, str], timezone):
        try:
            dt = pd.Timestamp(dt, tz=timezone).timestamp()
            dt = pd.Timestamp(dt, tz=tzlocal(), unit="s")
        except ValueError as e:
            pass
        return dt

    @property
    @abc.abstractmethod
    def _timezone(self):
        raise NotImplementedError("rewrite get_timezone")

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end, show_1min_logging: bool = False):
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        def _show_logging_func():
            if interval == YahooCollector.INTERVAL_1min and show_1min_logging:
                logger.warning(f"{error_msg}:{_resp}")

        interval = "1m" if interval in ["1m", "1min"] else interval
        try:
            _resp = Ticker(symbol, asynchronous=False).history(interval=interval, start=start, end=end)
            if isinstance(_resp, pd.DataFrame):
                return _resp.reset_index()
            elif isinstance(_resp, dict):
                _temp_data = _resp.get(symbol, {})
                if isinstance(_temp_data, str) or (
                    isinstance(_resp, dict) and _temp_data.get("indicators", {}).get("quote", None) is None
                ):
                    _show_logging_func()
            else:
                _show_logging_func()
        except Exception as e:
            logger.warning(
                f"get data error: {symbol}--{start}--{end}"
                + "Your data request fails. This may be caused by your firewall (e.g. GFW). Please switch your network if you want to access Yahoo! data"
            )

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        @deco_retry(retry_sleep=self.delay, retry=self.retry)
        def _get_simple(start_, end_):
            self.sleep()
            _remote_interval = "1m" if interval == self.INTERVAL_1min else interval
            resp = self.get_data_from_remote(
                symbol,
                interval=_remote_interval,
                start=start_,
                end=end_,
            )
            if resp is None or resp.empty:
                raise ValueError(
                    f"get data error: {symbol}--{start_}--{end_}" + "The stock may be delisted, please check"
                )
            return resp

        _result = None
        if interval == self.INTERVAL_1d:
            try:
                _result = _get_simple(start_datetime, end_datetime)
            except ValueError as e:
                pass
        elif interval == self.INTERVAL_1min:
            _res = []
            _start = self.start_datetime
            while _start < self.end_datetime:
                _tmp_end = min(_start + pd.Timedelta(days=7), self.end_datetime)
                try:
                    _resp = _get_simple(_start, _tmp_end)
                    _res.append(_resp)
                except ValueError as e:
                    pass
                _start = _tmp_end
            if _res:
                _result = pd.concat(_res, sort=False).sort_values(["symbol", "date"])
        else:
            raise ValueError(f"cannot support {self.interval}")
        return pd.DataFrame() if _result is None else _result

    def collector_data(self):
        """collector data"""
        super(YahooCollector, self).collector_data()
        self.download_index_data()

    @abc.abstractmethod
    def download_index_data(self):
        """download index data"""
        raise NotImplementedError("rewrite download_index_data")


class YahooCollectorCN(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get HS stock symbols......")
        symbols = get_hs_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol):
        symbol_s = symbol.split(".")
        symbol = f"sh{symbol_s[0]}" if symbol_s[-1] == "ss" else f"sz{symbol_s[0]}"
        return symbol

    @property
    def _timezone(self):
        return "Asia/Shanghai"


class YahooCollectorCN1d(YahooCollectorCN):
    def download_index_data(self):
        # TODO: from MSN
        _format = "%Y%m%d"
        _begin = self.start_datetime.strftime(_format)
        _end = self.end_datetime.strftime(_format)
        for _index_name, _index_code in {"csi300": "000300", "csi100": "000903", "csi500": "000905"}.items():
            logger.info(f"get bench data: {_index_name}({_index_code})......")
            try:
                df = pd.DataFrame(
                    map(
                        lambda x: x.split(","),
                        requests.get(
                            INDEX_BENCH_URL.format(index_code=_index_code, begin=_begin, end=_end), timeout=None
                        ).json()["data"]["klines"],
                    )
                )
            except Exception as e:
                logger.warning(f"get {_index_name} error: {e}")
                continue
            df.columns = ["date", "open", "close", "high", "low", "volume", "money", "change"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.astype(float, errors="ignore")
            df["adjclose"] = df["close"]
            df["symbol"] = f"sh{_index_code}"
            _path = self.save_dir.joinpath(f"sh{_index_code}.csv")
            if _path.exists():
                _old_df = pd.read_csv(_path)
                df = pd.concat([_old_df, df], sort=False)
            df.to_csv(_path, index=False)
            time.sleep(5)


class YahooCollectorCN1min(YahooCollectorCN):
    def get_instrument_list(self):
        symbols = super(YahooCollectorCN1min, self).get_instrument_list()
        return symbols + ["000300.ss", "000905.ss", "000903.ss"]

    def download_index_data(self):
        pass


class YahooCollectorUS(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get US stock symbols......")
        symbols = get_us_stock_symbols() + [
            "^GSPC",
            "^NDX",
            "^DJI",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "America/New_York"


class YahooCollectorUS1d(YahooCollectorUS):
    pass


class YahooCollectorUS1min(YahooCollectorUS):
    pass


class YahooCollectorHK(YahooCollector, ABC):
    def __init__(
        self,
        *args,
        update_mode: str = None,
        futu_host: str = "127.0.0.1",
        futu_port: int = 11111,
        **kwargs,
    ):
        # keep host/port lightweight to avoid pickle issues when running in parallel
        self.futu_host = futu_host or "127.0.0.1"
        try:
            self.futu_port = int(futu_port) if futu_port is not None else 11111
        except Exception:
            self.futu_port = 11111
        # detect whether we are running via update_data_to_bin flow
        self.use_futu_snapshot = str(update_mode).lower() == "update_data_to_bin"
        self._futu_snapshot_cache = None
        super().__init__(*args, update_mode=update_mode, **kwargs)
        if self.use_futu_snapshot:
            # futu snapshot is single-request; no need for inter-symbol sleep
            self.delay = 0
        if self.use_futu_snapshot and self.interval == self.INTERVAL_1d:
            self._prefetch_futu_snapshot()

    def get_instrument_list(self):
        logger.info("get HK stock symbols......")
        symbols = get_hk_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        # Prefer futu snapshot when running update_data_to_bin; fall back to Yahoo query otherwise
        if self.use_futu_snapshot and self.interval == self.INTERVAL_1d:
            df = self._get_futu_row("800000.HK", fetch_if_missing=True)
            if df is not None and not df.empty:
                _path = self.save_dir.joinpath("800000.HK.csv")
                try:
                    if _path.exists():
                        _old_df = pd.read_csv(_path)
                        df = pd.concat([_old_df, df], sort=False)
                except Exception:
                    logger.warning(f"Failed to read existing HSI CSV at {_path}; will overwrite")
                df.to_csv(_path, index=False)
                logger.info(f"Saved HSI snapshot from Futu to {_path}")
                return
            logger.warning("Futu snapshot unavailable for HSI; fallback to Yahoo query")
        self._download_index_data_via_yahoo()

    def normalize_symbol(self, symbol):
        s = str(symbol).strip()
        if "." in s:
            parts = s.split(".")
            if parts[-1].lower() == "hk":
                core = parts[0]
                if core.isdigit():
                    core = core.zfill(5)
                return f"{core}.HK"
            return s
        if s.isdigit():
            return f"{s.zfill(5)}.HK"
        return s

    @staticmethod
    def get_data_from_remote(symbol, interval, start, end, show_1min_logging: bool = False):
        """Override to try Yahoo candidate symbols for HK (one-leading-zero removed).

        Returns the first successful DataFrame (with an extra column
        `used_symbol_for_yahoo` indicating which Yahoo symbol was used),
        or None if all candidates failed.
        """
        error_msg = f"{symbol}-{interval}-{start}-{end}"

        def _show_logging_func(_s, _resp):
            if interval == YahooCollector.INTERVAL_1min and show_1min_logging:
                logger.warning(f"{error_msg}:{_resp}")

        # normalize interval for yahooquery
        _interval = "1m" if interval in ["1m", "1min"] else interval

        candidates = yahoo_hk_symbol_candidates(symbol)
        print(candidates)
        for cand in candidates:
            try:
                _resp = Ticker(cand, asynchronous=False).history(interval=_interval, start=start, end=end)
                print (_resp.head())
                if isinstance(_resp, pd.DataFrame):
                    df = _resp.reset_index()
                    #df["used_symbol_for_yahoo"] = cand
                    if interval.lower() == "1d" and "date" in df.columns:
                        df["date"] = df["date"].astype(str).str[:10]
                        #df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    return df
                elif isinstance(_resp, dict):
                    _temp_data = _resp.get(cand, {})
                    if isinstance(_temp_data, str) or (
                        isinstance(_resp, dict) and _temp_data.get("indicators", {}).get("quote", None) is None
                    ):
                        _show_logging_func(cand, _resp)
                        # try next candidate
                        continue
                    # If dict contains data but not DataFrame, we skip conversion here
                    _show_logging_func(cand, _resp)
                else:
                    _show_logging_func(cand, _resp)
            except Exception:
                logger.warning(
                    f"get data error: {cand}--{start}--{end}"
                    + "Your data request fails. This may be caused by your firewall (e.g. GFW). Please switch your network if you want to access Yahoo! data"
                )

        # none succeeded
        return None

    def _download_index_data_via_yahoo(self):
        # Download HSI (^HSI) and save as 800000.HK.csv so normalize/dump can treat it as benchmark
        try:
            logger.info("get HSI (^HSI) data......")
            _start = self.start_datetime.strftime("%Y-%m-%d")
            _end = self.end_datetime.strftime("%Y-%m-%d")
            resp = Ticker("^HSI", asynchronous=False).history(interval="1d", start=_start, end=_end)
            if isinstance(resp, pd.DataFrame) and not resp.empty:
                df = resp.reset_index()
                if "date" in df.columns:
                    df["date"] = df["date"].astype(str).str[:10]
                if "adjclose" not in df.columns and "close" in df.columns:
                    df["adjclose"] = df["close"]
                df["symbol"] = "800000.HK"
                _path = self.save_dir.joinpath("800000.HK.csv")
                try:
                    if _path.exists():
                        _old_df = pd.read_csv(_path)
                        df = pd.concat([_old_df, df], sort=False)
                except Exception:
                    logger.warning(f"Failed to read existing HSI CSV at {_path}; will overwrite")
                df.to_csv(_path, index=False)
                logger.info(f"Saved HSI to {_path}")
            else:
                logger.warning("No HSI data fetched from Yahoo for given date range")
        except Exception as e:
            logger.warning(f"Failed to fetch HSI (^HSI): {e}")

    @staticmethod
    def _to_futu_code(symbol: str) -> str:
        s = str(symbol).replace("HK.", "").replace(".HK", "").strip()
        if not s:
            return None
        if not s.isdigit():
            return None
        return f"HK.{s.zfill(5)}"

    @staticmethod
    def _from_futu_code(code: str) -> str:
        c = str(code).upper().replace("HK.", "").replace(".HK", "").strip()
        if not c:
            return None
        return f"{c.zfill(5)}.HK"

    def _fetch_futu_snapshot(self, symbols):
        try:
            from futu import OpenQuoteContext, RET_OK
        except Exception as e:
            logger.warning(f"Futu API not available: {e}")
            return None
        futu_codes = [self._to_futu_code(s) for s in symbols if self._to_futu_code(s)]
        if not futu_codes:
            return None
        frames = []
        for i in range(0, len(futu_codes), 400):
            batch = futu_codes[i : i + 400]
            try:
                ctx = OpenQuoteContext(host=self.futu_host, port=self.futu_port)
            except Exception as e:
                logger.warning(f"Failed to open Futu OpenD connection: {e}")
                return None
            try:
                ret, data = ctx.get_market_snapshot(batch)
            finally:
                try:
                    ctx.close()
                except Exception:
                    pass
            if ret != RET_OK:
                logger.warning(f"Futu snapshot request failed for batch starting {batch[0]}: {data}")
                return None
            frames.append(data)
        if not frames:
            return None
        raw_df = pd.concat(frames, ignore_index=True)
        if raw_df.empty:
            return None
        df = pd.DataFrame()
        df["symbol"] = raw_df["code"].apply(self._from_futu_code)
        df["date"] = pd.to_datetime(raw_df["update_time"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["open"] = raw_df.get("open_price")
        df["close"] = raw_df.get("last_price")
        df["high"] = raw_df.get("high_price")
        df["low"] = raw_df.get("low_price")
        df["volume"] = pd.to_numeric(raw_df.get("volume"), errors="coerce")
        df["adjclose"] = df["close"]
        df = df.dropna(subset=["symbol"]).reset_index(drop=True)
        return df

    def _prefetch_futu_snapshot(self):
        try:
            symbols = list(set(self.instrument_list + ["800000.HK"]))
        except Exception:
            symbols = ["800000.HK"]
        df = self._fetch_futu_snapshot(symbols)
        if df is not None and not df.empty:
            self._futu_snapshot_cache = df
            logger.info(f"Loaded {len(df)} symbols from Futu snapshot for update_data_to_bin")
        else:
            logger.warning("Futu snapshot prefetch failed; will fallback to Yahoo per-symbol")

    def _get_futu_row(self, symbol: str, fetch_if_missing: bool = False):
        if self._futu_snapshot_cache is not None:
            _mask = self._futu_snapshot_cache["symbol"] == self.normalize_symbol(symbol)
            if _mask.any():
                return self._futu_snapshot_cache[_mask].copy()
        if fetch_if_missing:
            df = self._fetch_futu_snapshot([symbol])
            if df is not None and not df.empty:
                return df
        return None

    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        if self.use_futu_snapshot and interval == self.INTERVAL_1d:
            df = self._get_futu_row(symbol, fetch_if_missing=True)
            if df is not None and not df.empty:
                try:
                    df_ts = pd.to_datetime(df["date"], errors="coerce")
                    df = df[(df_ts >= pd.to_datetime(start_datetime)) & (df_ts < pd.to_datetime(end_datetime))]
                except Exception:
                    pass
                if not df.empty:
                    return df.reset_index(drop=True)
        return super(YahooCollectorHK, self).get_data(symbol, interval, start_datetime, end_datetime)

    @property
    def _timezone(self):
        return "Asia/Hong_Kong"


class YahooCollectorHK1d(YahooCollectorHK):
    pass


class YahooCollectorIN(YahooCollector, ABC):
    def get_instrument_list(self):
        logger.info("get INDIA stock symbols......")
        symbols = get_in_stock_symbols()
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "Asia/Kolkata"


class YahooCollectorIN1d(YahooCollectorIN):
    pass


class YahooCollectorIN1min(YahooCollectorIN):
    pass


class YahooCollectorBR(YahooCollector, ABC):
    def retry(cls):  # pylint: disable=E0213
        """
        The reason to use retry=2 is due to the fact that
        Yahoo Finance unfortunately does not keep track of some
        Brazilian stocks.

        Therefore, the decorator deco_retry with retry argument
        set to 5 will keep trying to get the stock data up to 5 times,
        which makes the code to download Brazilians stocks very slow.

        In future, this may change, but for now
        I suggest to leave retry argument to 1 or 2 in
        order to improve download speed.

        To achieve this goal an abstract attribute (retry)
        was added into YahooCollectorBR base class
        """
        raise NotImplementedError

    def get_instrument_list(self):
        logger.info("get BR stock symbols......")
        symbols = get_br_stock_symbols() + [
            "^BVSP",
        ]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def download_index_data(self):
        pass

    def normalize_symbol(self, symbol):
        return code_to_fname(symbol).upper()

    @property
    def _timezone(self):
        return "Brazil/East"


class YahooCollectorBR1d(YahooCollectorBR):
    retry = 2


class YahooCollectorBR1min(YahooCollectorBR):
    retry = 2


class YahooNormalize(BaseNormalize):
    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        # Protect against missing or entirely-NaN close series
        if df is None or df.empty or "close" not in df.columns or df["close"].dropna().empty:
            # return a Series of NaNs with same index shape
            return pd.Series([np.nan] * (0 if df is None else len(df.index)), index=([] if df is None else df.index))
        df = df.copy()
        _tmp_series = df["close"].ffill()
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None and len(_tmp_shift_series) > 0:
            try:
                _tmp_shift_series.iloc[0] = float(last_close)
            except Exception:
                pass
        # avoid divide-by-zero by treating zero previous close as NaN
        _tmp_shift_series = _tmp_shift_series.replace(0, np.nan)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    @staticmethod
    def normalize_yahoo(
        df: pd.DataFrame,
        calendar_list: list = None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        last_close: float = None,
    ):
        if df.empty:
            return df
        symbol = df.loc[df[symbol_field_name].first_valid_index(), symbol_field_name]
        columns = copy.deepcopy(YahooNormalize.COLUMNS)
        df = df.copy()
        df.set_index(date_field_name, inplace=True)

        # Robustly convert index to timezone-naive DatetimeIndex (handle mixed formats)
        tmp_idx = []
        for _v in df.index:
            try:
                ts = pd.Timestamp(_v)
                # If tz-aware, drop tzinfo while preserving wall-clock time
                if getattr(ts, "tz", None) is not None and ts.tz is not None:
                    py_dt = ts.to_pydatetime().replace(tzinfo=None)
                    tmp_idx.append(pd.Timestamp(py_dt))
                else:
                    tmp_idx.append(ts)
            except Exception:
                tmp_idx.append(pd.NaT)

        df.index = pd.DatetimeIndex(tmp_idx)
        # drop rows with unparseable index entries and log how many were removed
        if df.index.isna().any():
            n_drop = int(df.index.isna().sum())
            logger.warning(f"{symbol}: dropped {n_drop} rows due to unparseable dates")
            df = df[~df.index.isna()]
        # remove duplicate timestamps (keep first)
        df = df[~df.index.duplicated(keep="first")]
        if calendar_list is not None:
            # Build timezone-naive DatetimeIndex from calendar_list robustly.
            tmp_list = []
            for _d in list(calendar_list):
                ts = pd.Timestamp(_d)
                # If ts has tz info, drop tz while preserving wall-time
                if getattr(ts, "tz", None) is not None and ts.tz is not None:
                    py_dt = ts.to_pydatetime().replace(tzinfo=None)
                    tmp_list.append(pd.Timestamp(py_dt))
                else:
                    tmp_list.append(ts)
            cal_index = pd.DatetimeIndex(tmp_list)
            cal_index = cal_index.sort_values().unique()
            # determine start/end bounds as timezone-naive timestamps
            start_ts = pd.Timestamp(df.index.min()).normalize()
            end_ts = pd.Timestamp(df.index.max()).normalize() + pd.Timedelta(hours=23, minutes=59)
            # filter calendar index within bounds
            cal_index = cal_index[(cal_index >= start_ts) & (cal_index <= end_ts)]
            # reindex by the filtered DatetimeIndex
            df = df.reindex(cal_index)
        df.sort_index(inplace=True)
        # When volume is 0 or missing it represents no trade that day; keep price columns intact
        # only nullify the volume column here. The `change` column will be handled after computation.
        if "volume" in df.columns:
            # Preserve zero volume (no trades) as 0.0; only treat negative or invalid volumes as missing.
            neg_mask = df["volume"] < 0
            if neg_mask.any():
                df.loc[neg_mask, "volume"] = np.nan

        change_series = YahooNormalize.calc_change(df, last_close)
        # NOTE: The data obtained by Yahoo finance sometimes has exceptions
        # WARNING: If it is normal for a `symbol(exchange)` to differ by a factor of *89* to *111* for consecutive trading days,
        # WARNING: the logic in the following line needs to be modified
        _count = 0
        while True:
            # NOTE: may appear unusual for many days in a row
            change_series = YahooNormalize.calc_change(df, last_close)
            _mask = (change_series >= 89) & (change_series <= 111)
            if not _mask.any():
                break
            _tmp_cols = ["high", "close", "low", "open", "adjclose"]
            df.loc[_mask, _tmp_cols] = df.loc[_mask, _tmp_cols] / 100
            _count += 1
            if _count >= 10:
                _symbol = df.loc[df[symbol_field_name].first_valid_index()]["symbol"]
                logger.warning(
                    f"{_symbol} `change` is abnormal for {_count} consecutive days, please check the specific data file carefully"
                )

        df["change"] = YahooNormalize.calc_change(df, last_close)

        columns += ["change"]
        # For rows with no trading volume (volume == 0) or missing volume, keep price columns,
        # set `change` to NaN. Keep volume==0 as 0.0; only missing volumes remain NaN.
        if "volume" in df.columns:
            zero_mask = df["volume"] == 0
            missing_mask = df["volume"].isna()
            neg_mask = df["volume"] < 0
            if "change" in df.columns:
                # Treat explicit zero-volume (no trades) as no price change
                if zero_mask.any():
                    df.loc[zero_mask, "change"] = 0.0
                # Keep missing or negative volumes as unknown change
                if (missing_mask | neg_mask).any():
                    df.loc[missing_mask | neg_mask, "change"] = np.nan

        df[symbol_field_name] = symbol
        df.index.names = [date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # normalize
        df = self.normalize_yahoo(df, self._calendar_list, self._date_field_name, self._symbol_field_name)
        # adjusted price
        df = self.adjusted_price(df)
        return df

    @abc.abstractmethod
    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """adjusted price"""
        raise NotImplementedError("rewrite adjusted_price")


class YahooNormalize1d(YahooNormalize, ABC):
    DAILY_FORMAT = "%Y-%m-%d"
    def __init__(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        skip_manual_adjust: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        skip_manual_adjust: bool
            If True, skip the manual adjustment step that rescales prices
            so the first valid close equals 1. Default False.
        """
        super(YahooNormalize1d, self).__init__(date_field_name=date_field_name, symbol_field_name=symbol_field_name, **kwargs)
        self.skip_manual_adjust = bool(skip_manual_adjust)

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        if "adjclose" in df:
            df["factor"] = df["adjclose"] / df["close"]
            df["factor"] = df["factor"].ffill()
        else:
            df["factor"] = 1
        for _col in self.COLUMNS:
            if _col not in df.columns:
                continue
            if _col == "volume":
                df[_col] = df[_col] / df["factor"]
            else:
                df[_col] = df[_col] * df["factor"]
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(YahooNormalize1d, self).normalize(df)
        df = self._manual_adj_data(df)
        return df

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """get first close value

        Notes
        -----
            For incremental updates(append) to Yahoo 1D data, user need to use a close that is not 0 on the first trading day of the existing data
        """
        df = df.loc[df["close"].first_valid_index() :]
        _close = df["close"].iloc[0]
        return _close

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """manual adjust data: All fields (except change) are standardized according to the close of the first day"""
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df = df.set_index(self._date_field_name)

        # If skip_manual_adjust is set, return early without scaling
        if getattr(self, "skip_manual_adjust", False):
            return df.reset_index()
    
        # compute a numeric first close; if unavailable or zero, warn and skip adjustments
        try:
            _close = float(self._get_first_close(df))
        except Exception:
            logger.warning("Unable to determine first close for manual adjustment; skipping manual adjustment")
            return df.reset_index()
    
        if _close == 0 or pd.isna(_close):
            logger.warning("First close is zero or NaN; skipping manual adjustment to avoid division by zero")
            return df.reset_index()
    
        # Convert numeric columns (except symbol/adjclose/change) to numeric safely, then apply adjustments
        for _col in df.columns:
            if _col in [self._symbol_field_name, "adjclose", "change"]:
                continue
            # coerce values to numeric; invalid parsing becomes NaN
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
            if _col == "volume":
                df[_col] = df[_col] * _close
            else:
                df[_col] = df[_col] / _close
    
        return df.reset_index()


class YahooNormalize1dExtend(YahooNormalize1d):
    def __init__(
        self, old_qlib_data_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        old_qlib_data_dir: str, Path
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        super(YahooNormalize1dExtend, self).__init__(date_field_name, symbol_field_name)
        self.column_list = ["open", "high", "low", "close", "volume", "factor", "change"]
        self.old_qlib_data = self._get_old_data(old_qlib_data_dir)

    def _get_old_data(self, qlib_data_dir: [str, Path]):
        qlib_data_dir = str(Path(qlib_data_dir).expanduser().resolve())
        qlib.init(provider_uri=qlib_data_dir, expression_cache=None, dataset_cache=None)
        df = D.features(D.instruments("all"), ["$" + col for col in self.column_list])
        df.columns = self.column_list
        return df

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(YahooNormalize1dExtend, self).normalize(df)
        # If normalization returned no valid rows, skip extension to avoid iloc[0] out-of-bounds
        if df is None or df.empty:
            logger.warning(
                f"{getattr(self, '_symbol_field_name', 'symbol')} normalize returned empty dataframe; skipping extend"
            )
            return pd.DataFrame()
        df.set_index(self._date_field_name, inplace=True)
        symbol_name = df[self._symbol_field_name].iloc[0]
        old_symbol_list = self.old_qlib_data.index.get_level_values("instrument").unique().to_list()
        if str(symbol_name).upper() not in old_symbol_list:
            return df.reset_index()
        old_df = self.old_qlib_data.loc[str(symbol_name).upper()]
        latest_date = old_df.index[-1]
        df = df.loc[latest_date:]
        # after: df = df.loc[latest_date:]
        if df.empty:
            logger.warning(f"{symbol_name} has no new rows after {latest_date}; skipping extension for this symbol")
            return pd.DataFrame()
        new_latest_data = df.iloc[0]
        old_latest_data = old_df.loc[latest_date]
        for col in self.column_list[:-1]:
            if col == "volume":
                df[col] = df[col] / (new_latest_data[col] / old_latest_data[col])
            else:
                df[col] = df[col] * (old_latest_data[col] / new_latest_data[col])
        return df.drop(df.index[0]).reset_index()


class YahooNormalize1min(YahooNormalize, ABC):
    """Normalised to 1min using local 1d data"""

    AM_RANGE = None  # type: tuple  # eg: ("09:30:00", "11:29:00")
    PM_RANGE = None  # type: tuple  # eg: ("13:00:00", "14:59:00")

    # Whether the trading day of 1min data is consistent with 1d
    CONSISTENT_1d = True
    CALC_PAUSED_NUM = True

    def __init__(
        self, qlib_data_1d_dir: [str, Path], date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs
    ):
        """

        Parameters
        ----------
        qlib_data_1d_dir: str, Path
            the qlib data to be updated for yahoo, usually from: Normalised to 1min using local 1d data
        date_field_name: str
            date field name, default is date
        symbol_field_name: str
            symbol field name, default is symbol
        """
        super(YahooNormalize1min, self).__init__(date_field_name, symbol_field_name)
        qlib.init(provider_uri=qlib_data_1d_dir)
        self.all_1d_data = D.features(D.instruments("all"), ["$paused", "$volume", "$factor", "$close"], freq="day")

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return list(D.calendar(freq="day"))

    @property
    def calendar_list_1d(self):
        calendar_list_1d = getattr(self, "_calendar_list_1d", None)
        if calendar_list_1d is None:
            calendar_list_1d = self._get_1d_calendar_list()
            setattr(self, "_calendar_list_1d", calendar_list_1d)
        return calendar_list_1d

    def generate_1min_from_daily(self, calendars: Iterable) -> pd.Index:
        return generate_minutes_calendar_from_daily(
            calendars, freq="1min", am_range=self.AM_RANGE, pm_range=self.PM_RANGE
        )

    def adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calc_adjusted_price(
            df=df,
            _date_field_name=self._date_field_name,
            _symbol_field_name=self._symbol_field_name,
            frequence="1min",
            consistent_1d=self.CONSISTENT_1d,
            calc_paused=self.CALC_PAUSED_NUM,
            _1d_data_all=self.all_1d_data,
        )
        return df

    @abc.abstractmethod
    def symbol_to_yahoo(self, symbol):
        raise NotImplementedError("rewrite symbol_to_yahoo")


class YahooNormalizeUS:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: from MSN
        return get_calendar_list("US_ALL")


class YahooNormalizeUS1d(YahooNormalizeUS, YahooNormalize1d):
    pass


class YahooNormalizeUS1dExtend(YahooNormalizeUS, YahooNormalize1dExtend):
    pass


class YahooNormalizeUS1min(YahooNormalizeUS, YahooNormalize1min):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("US_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)

class YahooNormalizeHK:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("HK_ALL")

class YahooNormalizeHK1d(YahooNormalizeHK, YahooNormalize1d):
    pass

class YahooNormalizeHK1dExtend(YahooNormalizeHK, YahooNormalize1dExtend): 
    pass


class YahooNormalizeIN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("IN_ALL")


class YahooNormalizeIN1d(YahooNormalizeIN, YahooNormalize1d):
    pass


class YahooNormalizeIN1min(YahooNormalizeIN, YahooNormalize1min):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("IN_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class YahooNormalizeCN:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: from MSN
        return get_calendar_list("ALL")


class YahooNormalizeCN1d(YahooNormalizeCN, YahooNormalize1d):
    pass


class YahooNormalizeCN1dExtend(YahooNormalizeCN, YahooNormalize1dExtend):
    pass


class YahooNormalizeCN1min(YahooNormalizeCN, YahooNormalize1min):
    AM_RANGE = ("09:30:00", "11:29:00")
    PM_RANGE = ("13:00:00", "14:59:00")

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return self.generate_1min_from_daily(self.calendar_list_1d)

    def symbol_to_yahoo(self, symbol):
        if "." not in symbol:
            _exchange = symbol[:2]
            _exchange = ("ss" if _exchange.islower() else "SS") if _exchange.lower() == "sh" else _exchange
            symbol = symbol[2:] + "." + _exchange
        return symbol

    def _get_1d_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("ALL")


class YahooNormalizeBR:
    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list("BR_ALL")


class YahooNormalizeBR1d(YahooNormalizeBR, YahooNormalize1d):
    pass


class YahooNormalizeBR1min(YahooNormalizeBR, YahooNormalize1min):
    CALC_PAUSED_NUM = False

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        # TODO: support 1min
        raise ValueError("Does not support 1min")

    def _get_1d_calendar_list(self):
        return get_calendar_list("BR_ALL")

    def symbol_to_yahoo(self, symbol):
        return fname_to_code(symbol)


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region=REGION_CN):
        """

        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved, default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data, default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1; when collecting data, it is recommended that max_workers be set to 1
        interval: str
            freq, value from [1min, 1d], default 1d
        region: str
            region, value from ["CN", "US", "BR"], default "CN"
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"YahooCollector{self.region.upper()}{self.interval}"

    @property
    def normalize_class_name(self):
        return f"YahooNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
        symbols: str = None,
        **kwargs,
    ):
        """download data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        start: str
            start datetime, default "2000-01-01"; closed interval(including start)
        end: str
            end datetime, default ``pd.Timestamp(datetime.datetime.now() + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --region CN --start 2020-11-01 --end 2020-11-10 --delay 0.1 --interval 1m
        """
        #if self.interval == "1d" and pd.Timestamp(end) > pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d")):
            #raise ValueError(f"end_date: {end} is greater than the current date.")

        super(Run, self).download_data(
            max_collector_count,
            delay,
            start,
            end,
            check_data_length,
            limit_nums,
            symbols=symbols,
            **kwargs,
        )

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
        skip_manual_adjust: bool = False,
    ):
        """normalize data

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        end_date: str
            if not None, normalize the last date saved (including end_date); if None, it will ignore this parameter; by default None
        qlib_data_1d_dir: str
            if interval==1min, qlib_data_1d_dir cannot be None, normalize 1min needs to use 1d data;

                qlib_data_1d can be obtained like this:
                    $ python scripts/get_data.py qlib_data --target_dir <qlib_data_1d_dir> --interval 1d
                    $ python scripts/data_collector/yahoo/collector.py update_data_to_bin --qlib_data_1d_dir <qlib_data_1d_dir> --trading_date 2021-06-01
                or:
                    download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region cn --interval 1d
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source_cn_1min --normalize_dir ~/.qlib/stock_data/normalize_cn_1min --region CN --interval 1min
        """
        if self.interval.lower() == "1min":
            if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
                raise ValueError(
                    "If normalize 1min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
                )
        super(Run, self).normalize_data(
            date_field_name,
            symbol_field_name,
            end_date=end_date,
            qlib_data_1d_dir=qlib_data_1d_dir,
            skip_manual_adjust=skip_manual_adjust,
        )

    def normalize_data_1d_extend(
        self, old_qlib_data_dir, date_field_name: str = "date", symbol_field_name: str = "symbol"
    ):
        """normalize data extend; extending yahoo qlib data(from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data)

        Notes
        -----
            Steps to extend yahoo qlib data:

                1. download qlib data: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data; save to <dir1>

                2. collector source data: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#collector-data; save to <dir2>

                3. normalize new source data(from step 2): python scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <dir1> --source_dir <dir2> --normalize_dir <dir3> --region CN --interval 1d

                4. dump data: python scripts/dump_bin.py dump_update --data_path <dir3> --qlib_dir <dir1> --freq day --date_field_name date --symbol_field_name symbol --exclude_fields symbol,date

                5. update instrument(eg. csi300): python python scripts/data_collector/cn_index/collector.py --index_name CSI300 --qlib_dir <dir1> --method parse_instruments

        Parameters
        ----------
        old_qlib_data_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol

        Examples
        ---------
            $ python collector.py normalize_data_1d_extend --old_qlib_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --region CN --interval 1d
        """
        _class = getattr(self._cur_module, f"{self.normalize_class_name}Extend")
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            old_qlib_data_dir=old_qlib_data_dir,
        )
        yc.normalize()

    def download_today_data(
        self,
        max_collector_count=2,
        delay=0.5,
        check_data_length=None,
        limit_nums=None,
    ):
        """download today data from Internet

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        limit_nums: int
            using for debug, by default None

        Notes
        -----
            Download today's data:
                start_time = datetime.datetime.now().date(); closed interval(including start)
                end_time = pd.Timestamp(start_time + pd.Timedelta(days=1)).date(); open interval(excluding end)

            check_data_length, example:
                daily, one year: 252 // 4
                us 1min, a week: 6.5 * 60 * 5
                cn 1min, a week: 4 * 60 * 5

        Examples
        ---------
            # get daily data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1d
            # get 1m data
            $ python collector.py download_today_data --source_dir ~/.qlib/stock_data/source --region CN --delay 0.1 --interval 1m
        """
        start = datetime.datetime.now().date()
        end = pd.Timestamp(start + pd.Timedelta(days=1)).date()
        self.download_data(
            max_collector_count,
            delay,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            check_data_length,
            limit_nums,
        )

    def update_data_to_bin(
        self,
        qlib_data_1d_dir: str,
        end_date: str = None,
        check_data_length: int = None,
        delay: float = 1,
        exists_skip: bool = True,
        skip_download: bool = False,
    ):
        """update yahoo data to bin

        Parameters
        ----------
        qlib_data_1d_dir: str
            the qlib data to be updated for yahoo, usually from: https://github.com/microsoft/qlib/tree/main/scripts#download-cn-data

        end_date: str
            end datetime, default ``pd.Timestamp(trading_date + pd.Timedelta(days=1))``; open interval(excluding end)
        check_data_length: int
            check data length, if not None and greater than 0, each symbol will be considered complete if its data length is greater than or equal to this value, otherwise it will be fetched again, the maximum number of fetches being (max_collector_count). By default None.
        delay: float
            time.sleep(delay), default 1
        exists_skip: bool
            exists skip, by default False
        Notes
        -----
            If the data in qlib_data_dir is incomplete, np.nan will be populated to trading_date for the previous trading day

        Examples
        -------
            $ python collector.py update_data_to_bin --qlib_data_1d_dir <user data dir> --trading_date <start date> --end_date <end date>
        """

        if self.interval.lower() != "1d":
            logger.warning(f"currently supports 1d data updates: --interval 1d")

        # download qlib 1d data
        qlib_data_1d_dir = str(Path(qlib_data_1d_dir).expanduser().resolve())
        if not exists_qlib_data(qlib_data_1d_dir):
            GetData().qlib_data(
                target_dir=qlib_data_1d_dir, interval=self.interval, region=self.region, exists_skip=exists_skip
            )

        # start/end date using future calendar and robust catch-up logic
        calendar_df = pd.read_csv(Path(qlib_data_1d_dir).joinpath("calendars/day.txt"))
        last_trading_day = pd.Timestamp(calendar_df.iloc[-1, 0]).strftime("%Y-%m-%d")
        next_trading_day = next_trading_day_from_future(qlib_data_1d_dir, last_trading_day)
        if next_trading_day is None:
            next_trading_day = (pd.Timestamp(last_trading_day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        region_lower = self.region.lower()
        tz_name = {
            "cn": "Asia/Shanghai",
            "us": "America/New_York",
            "hk": "Asia/Hong_Kong",
            "in": "Asia/Kolkata",
            "br": "Brazil/East",
        }.get(region_lower, "UTC")
        try:
            now_ts = datetime.datetime.now(ZoneInfo(tz_name))
        except Exception:
            now_ts = datetime.datetime.now()
        today = now_ts.date().strftime("%Y-%m-%d")
        market_close_time = {
            "cn": datetime.time(15, 5),
            "hk": datetime.time(16, 15),
            "us": datetime.time(16, 5),
            "in": datetime.time(15, 35),
            "br": datetime.time(17, 10),
        }.get(region_lower, datetime.time(16, 0))

        if last_trading_day == today:
            logger.info(f"skip update: data already up to today (last_trading_day={last_trading_day})")
            return
        if today < next_trading_day:
            logger.info(
                f"skip update: today {today} is before next trading day {next_trading_day}; no new trading day yet"
            )
            return
        if today == next_trading_day and now_ts.time() < market_close_time:
            logger.info(
                f"skip update: market not closed for region={region_lower} (now={now_ts.time()} < close={market_close_time})"
            )
            return

        trading_date = next_trading_day
        if end_date is None:
            if today > next_trading_day:
                end_date = (pd.Timestamp(today) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                end_date = (pd.Timestamp(trading_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"trading_date: {trading_date}")
        print(f"end_date: {end_date}")
        # download data from yahoo (can be skipped via flag)
        # NOTE: when downloading data from YahooFinance, max_workers is recommended to be 1
        if not skip_download:
            self.download_data(
                delay=delay,
                start=trading_date,
                end=end_date,
                check_data_length=check_data_length,
                update_mode="update_data_to_bin",
            )
        else:
            logger.info(
                f"skip_download=True: reusing existing raw source files in {Path(self.source_dir).expanduser()}"
            )
            try:
                src_dir = Path(self.source_dir).expanduser()
                if not src_dir.exists():
                    logger.warning(f"source_dir does not exist: {src_dir}")
                else:
                    has_csv = any(src_dir.glob("*.csv"))
                    if not has_csv:
                        logger.warning(
                            "No CSV files found in source_dir; normalization may have nothing to process."
                        )
            except Exception:
                pass
        # NOTE: a larger max_workers setting here would be faster
        self.max_workers = (
            max(multiprocessing.cpu_count() - 2, 1)
            if self.max_workers is None or self.max_workers <= 1
            else self.max_workers
        )
        # normalize data
        self.normalize_data_1d_extend(qlib_data_1d_dir)

        # dump bin
        _dump = DumpDataUpdate(
            data_path=self.normalize_dir,
            qlib_dir=qlib_data_1d_dir,
            exclude_fields="symbol,date",
            max_workers=self.max_workers,
        )
        _dump.dump()

        # parse index
        _region = self.region.lower()
        if _region not in ["cn", "us"]:
            logger.warning(f"Unsupported region: region={_region}, component downloads will be ignored")
            return
        index_list = ["CSI100", "CSI300"] if _region == "cn" else ["SP500", "NASDAQ100", "DJIA", "SP400"]
        get_instruments = getattr(
            importlib.import_module(f"data_collector.{_region}_index.collector"), "get_instruments"
        )
        for _index in index_list:
            get_instruments(str(qlib_data_1d_dir), _index, market_index=f"{_region}_index")


if __name__ == "__main__":
    fire.Fire(Run)
