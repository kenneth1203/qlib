# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import copy
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List, Text, Tuple, Union, Set, Optional
from abc import ABC

import qlib
from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.log import get_module_logger
from qlib.utils import get_pre_trading_date, load_dataset
from qlib.contrib.strategy.order_generator import OrderGenerator, OrderGenWOInteract
from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer


class BaseSignalStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame] = None,
        model=None,
        dataset=None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        risk_degree : float
            position percentage of total value.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super().__init__(level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs)

        self.risk_degree = risk_degree

        # This is trying to be compatible with previous version of qlib task config
        if model is not None and dataset is not None:
            warnings.warn("`model` `dataset` is deprecated; use `signal`.", DeprecationWarning)
            signal = model, dataset

        self.signal: Signal = create_signal_from(signal)

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will use in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amount of your total value by default
        return self.risk_degree


class TopkDropoutStrategy(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    # 4. Regenerate results with forbid_all_trade_at_limit set to false and flip the default to false, as it is consistent with reality.
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        **kwargs,
    ):
        """
        Parameters
        -----------
        topk : int
            the number of stocks in the portfolio.
        n_drop : int
            number of stocks to be replaced in each trading date.
        method_sell : str
            dropout method_sell, random/bottom.
        method_buy : str
            dropout method_buy, random/top.
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.

            if only_tradable:

                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.

            else:

                strategy will make buy sell decision without checking the tradable state of the stock.
        forbid_all_trade_at_limit : bool
            if forbid all trades when limit_up or limit_down reached.

            if forbid_all_trade_at_limit:

                strategy will not do any trade when price reaches limit up/down, even not sell at limit up nor buy at
                limit down, though allowed in reality.

            else:

                strategy will sell at limit up and buy ad limit down.
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)
        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:
            # Otherwise, the stock will make decision without the stock tradable info
            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        current_temp: Position = copy.deepcopy(self.trade_position)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        # last position (sorted by score)
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")
        # combine(new stocks + last stocks),  we will drop stocks from this list
        # In case of dropping higher score stock and buying lower score stock.
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        # Get the stock list we really want to sell (After filtering the case that we sell high and buy low)
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:  # No enough candidates
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # Get the stock list we really want to buy
        buy = today[: len(sell) + self.topk - len(last)]
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                # check hold limit
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                # sell order
                sell_amount = current_temp.get_stock_amount(code=code)
                # sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,  # 0 for sell, 1 for buy
                )
                # is order executable
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    # update cash
                    cash += trade_val - trade_cost
        # buy new stock
        # note the current has been changed
        # current_stock_list = current_temp.get_stock_list()
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
        # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock suspended
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            # buy order
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,  # 1 for buy
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)

class MACDTopkDropoutStrategy_v2(TopkDropoutStrategy):
    """MACD Topk Dropout Strategy v2.

    - No internal indicator cache or indicator computation.
    - All indicator data is provided by workflow through args (indicator_data).
    - Keeps the same buy/sell trigger logic as MACDTopkDropoutStrategy.
    """

    def __init__(
        self,
        *,
        topk,
        n_drop,
        rsi_overbought: float = 70.0,
        mfi_overbought: float = 80.0,
        kdj_overbought: float = 80.0,
        kdj_oversold: float = 20.0,
        stop_loss_pct: float = 0.30,
        initial_risk_degree: float = 0.5,
        ramp_steps: int = 5,
        take_profit_pct: float = 1.0,
        take_profit_sell_ratio: float = 0.50,
        trailing_stop_drawdown: float = 0.20,
        min_turnover: float = 0.001,
        allow_missing_shares: bool = False,
        indicator_data: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs,
    ):
        super().__init__(topk=topk, n_drop=n_drop, **kwargs)
        self.rsi_overbought = rsi_overbought
        self.mfi_overbought = mfi_overbought
        self.kdj_overbought = kdj_overbought
        self.kdj_oversold = kdj_oversold
        self.stop_loss_pct = stop_loss_pct
        self.initial_risk_degree = initial_risk_degree
        self.ramp_steps = ramp_steps
        self.take_profit_pct = take_profit_pct
        self.take_profit_sell_ratio = take_profit_sell_ratio
        self.trailing_stop_drawdown = trailing_stop_drawdown
        self.min_turnover = min_turnover
        self.allow_missing_shares = allow_missing_shares
        self.indicator_data = indicator_data or {}
        # Cache normalized indicator DataFrames once at initialization
        try:
            self._cached_day_df = self._normalize_indicator_df(self._get_indicator_df("day"))
        except Exception:
            self._cached_day_df = pd.DataFrame()
        try:
            self._cached_month_df = self._normalize_indicator_df(self._get_indicator_df("month"))
        except Exception:
            self._cached_month_df = pd.DataFrame()
        try:
            self._cached_week_df = self._normalize_indicator_df(self._get_indicator_df("week"))
        except Exception:
            self._cached_week_df = pd.DataFrame()
        try:
            self._cached_year_df = self._normalize_indicator_df(self._get_indicator_df("year"))
        except Exception:
            self._cached_year_df = pd.DataFrame()
        # take-profit/trailing-stop tracking
        self._tp_holdout: Set[str] = set()
        self._tp_high: Dict[str, float] = {}
        self._issued_shares: Optional[pd.Series] = None
        self._issued_shares_loaded: bool = False

    def _load_issued_shares(self) -> pd.Series:
        if self._issued_shares_loaded:
            return self._issued_shares if self._issued_shares is not None else pd.Series(dtype=float)
        self._issued_shares_loaded = True
        try:
            from qlib.config import C

            provider_uri = getattr(C, "provider_uri", None)
            if isinstance(provider_uri, dict):
                provider_uri = provider_uri.get("__DEFAULT_FREQ") or next(iter(provider_uri.values()), None)
            if provider_uri is not None and not isinstance(provider_uri, str):
                provider_uri = str(provider_uri)
        except Exception:
            provider_uri = None
        if not provider_uri:
            self._issued_shares = pd.Series(dtype=float)
            return self._issued_shares
        try:
            path = os.path.join(os.path.expanduser(str(provider_uri)), "boardlot", "issued_shares.txt")
            if not os.path.exists(path):
                self._issued_shares = pd.Series(dtype=float)
                return self._issued_shares
            df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]
                df.columns = ["instrument", "shares"]
            elif df.shape[1] == 1:
                df.columns = ["instrument"]
                df["shares"] = np.nan
            else:
                self._issued_shares = pd.Series(dtype=float)
                return self._issued_shares
            df["instrument"] = df["instrument"].astype(str)
            df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
            df = df.dropna(subset=["shares"])
            self._issued_shares = df.set_index("instrument")["shares"].astype(float)
            return self._issued_shares
        except Exception:
            self._issued_shares = pd.Series(dtype=float)
            return self._issued_shares

    def _get_indicator_df(self, key: str) -> pd.DataFrame:
        df = self.indicator_data.get(key)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def _normalize_indicator_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            names = list(df.index.names)
            if "instrument" in names and "datetime" in names:
                df = df.reorder_levels(["instrument", "datetime"]).sort_index()
        return df

    def _compute_macd_flags(self, instruments: List[str], end_time) -> Dict[str, Dict[str, bool]]:
        # use cached normalized DataFrames initialized at construction
        df_all = self._cached_day_df if hasattr(self, "_cached_day_df") else self._normalize_indicator_df(self._get_indicator_df("day"))
        df_month = self._cached_month_df if hasattr(self, "_cached_month_df") else self._normalize_indicator_df(self._get_indicator_df("month"))
        df_week = self._cached_week_df if hasattr(self, "_cached_week_df") else self._normalize_indicator_df(self._get_indicator_df("week"))
        df_year = self._cached_year_df if hasattr(self, "_cached_year_df") else self._normalize_indicator_df(self._get_indicator_df("year"))
        if not instruments or df_all.empty:
            print("No instruments or indicator data available for MACD flag computation.")
            return {}
        end_ts = pd.Timestamp(end_time)
        flags: Dict[str, Dict[str, bool]] = {}

        for inst in instruments:
            try:
                sub = df_all.xs(inst, level="instrument")
            except Exception:
                continue
            try:
                if isinstance(sub.index, pd.DatetimeIndex):
                    sub = sub[sub.index <= end_ts]
                if sub.empty:
                    continue
                row = sub.iloc[-1]
                prev_row = sub.iloc[-2] if len(sub) >= 2 else None
                prev_row2 = sub.iloc[-3] if len(sub) >= 3 else None
                dif = float(row.get("$DIF")) if "$DIF" in row else None
                dea = float(row.get("$DEA")) if "$DEA" in row else None
                dif_prev = float(prev_row.get("$DIF")) if prev_row is not None and "$DIF" in prev_row else None
                dea_prev = float(prev_row.get("$DEA")) if prev_row is not None and "$DEA" in prev_row else None
                macd = float(row.get("$MACD")) if "$MACD" in row else None
                macd_prev = float(prev_row.get("$MACD")) if prev_row is not None and "$MACD" in prev_row else None
                macd_prev2 = float(prev_row2.get("$MACD")) if prev_row2 is not None and "$MACD" in prev_row2 else None
                mfi = float(row.get("$MFI")) if "$MFI" in row else None

                # monthly EMA alignment (EMA5 > EMA10 > EMA20), MFI > MA(MFI,10),
                # and MACD not two consecutive declines
                monthly_ok = True
                if not df_month.empty:
                    try:
                        sub_m = df_month.xs(inst, level="instrument")
                        if isinstance(sub_m.index, pd.DatetimeIndex):
                            sub_m = sub_m[sub_m.index <= end_ts]
                        if not sub_m.empty:
                            row_m = sub_m.iloc[-1]
                            prev_row_m = sub_m.iloc[-2] if len(sub_m) >= 2 else None
                            prev_row_m2 = sub_m.iloc[-3] if len(sub_m) >= 3 else None
                            ema5_m = float(row_m.get("$EMA5")) if "$EMA5" in row_m else None
                            ema10_m = float(row_m.get("$EMA10")) if "$EMA10" in row_m else None
                            ema20_m = float(row_m.get("$EMA20")) if "$EMA20" in row_m else None
                            macd_m = float(row_m.get("$MACD")) if "$MACD" in row_m else None
                            mfi_m = float(row_m.get("$MFI")) if "$MFI" in row_m else None
                            macd_m_prev = (
                                float(prev_row_m.get("$MACD"))
                                if prev_row_m is not None and "$MACD" in prev_row_m
                                else None
                            )
                            macd_m_prev2 = (
                                float(prev_row_m2.get("$MACD"))
                                if prev_row_m2 is not None and "$MACD" in prev_row_m2
                                else None
                            )
                            # MFI > MA(MFI,10) using monthly series
                            mfi_ok = False
                            try:
                                if "$MFI" in sub_m.columns:
                                    if "$MFI_MA10" in sub_m.columns:
                                        mfi_ma10 = row_m.get("$MFI_MA10")
                                        if mfi_m is not None and not pd.isna(mfi_ma10):
                                            mfi_ok = mfi_m > float(mfi_ma10)
                                    else:
                                        mfi_series = pd.to_numeric(sub_m["$MFI"], errors="coerce")
                                        mfi_recent = mfi_series.tail(10)
                                        if len(mfi_recent) >= 2:
                                            mfi_ma10 = mfi_recent.mean()
                                            if mfi_m is not None and not pd.isna(mfi_ma10):
                                                mfi_ok = mfi_m > float(mfi_ma10)
                            except Exception:
                                mfi_ok = False

                            if ema5_m is None or ema10_m is None or ema20_m is None or not mfi_ok:
                                monthly_ok = False
                            else:
                                monthly_ok = ema5_m > ema10_m > ema20_m
                            if monthly_ok and macd_m is not None and macd_m_prev is not None and macd_m_prev2 is not None:
                                monthly_ok = not (macd_m < macd_m_prev < macd_m_prev2)
                    except Exception:
                        monthly_ok = False
                else:
                    monthly_ok = False

                weekly_ok = False
                if not df_week.empty:
                    try:
                        sub_w = df_week.xs(inst, level="instrument")
                        if isinstance(sub_w.index, pd.DatetimeIndex):
                            sub_w = sub_w[sub_w.index <= end_ts]
                        if not sub_w.empty and "$MFI" in sub_w.columns:
                            row_w = sub_w.iloc[-1]
                            mfi_w = float(row_w.get("$MFI")) if "$MFI" in row_w else None
                            if "$MFI_MA10" in sub_w.columns:
                                mfi_ma10 = row_w.get("$MFI_MA10")
                                if mfi_w is not None and not pd.isna(mfi_ma10):
                                    weekly_ok = mfi_w > float(mfi_ma10)
                            else:
                                mfi_series = pd.to_numeric(sub_w["$MFI"], errors="coerce")
                                mfi_recent = mfi_series.tail(10)
                                if len(mfi_recent) >= 2:
                                    mfi_ma10 = mfi_recent.mean()
                                    if mfi_w is not None and not pd.isna(mfi_ma10):
                                        weekly_ok = mfi_w > float(mfi_ma10)

                            if weekly_ok and "$close" in sub_w.columns:
                                close_series = pd.to_numeric(sub_w["$close"], errors="coerce")
                                close_recent = close_series.tail(10)
                                mfi_recent = pd.to_numeric(sub_w["$MFI"], errors="coerce").tail(10)
                                if not close_recent.empty and not mfi_recent.empty:
                                    close_w = close_recent.iloc[-1]
                                    mfi_w_recent = mfi_recent.iloc[-1]
                                    if not pd.isna(close_w) and not pd.isna(mfi_w_recent):
                                        close_high = close_w == close_recent.max()
                                        if close_high:
                                            mfi_high = mfi_w_recent == mfi_recent.max()
                                            if not mfi_high:
                                                weekly_ok = False
                    except Exception:
                        weekly_ok = False

                yearly_ok = False
                if not df_year.empty:
                    try:
                        sub_y = df_year.xs(inst, level="instrument")
                        if isinstance(sub_y.index, pd.DatetimeIndex):
                            sub_y = sub_y[sub_y.index <= end_ts]
                        if not sub_y.empty and "$MFI" in sub_y.columns:
                            row_y = sub_y.iloc[-1]
                            mfi_y = float(row_y.get("$MFI")) if "$MFI" in row_y else None
                            if "$MFI_MA10" in sub_y.columns:
                                mfi_ma10 = row_y.get("$MFI_MA10")
                                if mfi_y is not None and not pd.isna(mfi_ma10):
                                    yearly_ok = mfi_y > float(mfi_ma10)
                            else:
                                mfi_series = pd.to_numeric(sub_y["$MFI"], errors="coerce")
                                mfi_recent = mfi_series.tail(10)
                                if len(mfi_recent) >= 2:
                                    mfi_ma10 = mfi_recent.mean()
                                    if mfi_y is not None and not pd.isna(mfi_ma10):
                                        yearly_ok = mfi_y > float(mfi_ma10)
                    except Exception:
                        yearly_ok = False

                vol_ok = False
                try:
                    if "$volume" in sub.columns:
                        vol_series = pd.to_numeric(sub["$volume"], errors="coerce")
                        vol_recent = vol_series[vol_series.index <= end_ts].tail(10)
                        if len(vol_recent) >= 9:
                            good_days = (vol_recent.fillna(0) > 0).sum()
                            vol_ok = int(good_days) >= 9
                        else:
                            vol_ok = False
                except Exception:
                    vol_ok = False

                mfi_day_ok = False
                try:
                    if "$MFI" in sub.columns:
                        if "$MFI_MA10" in sub.columns:
                            mfi_ma10 = row.get("$MFI_MA10")
                            if mfi is not None and not pd.isna(mfi_ma10):
                                mfi_day_ok = mfi > float(mfi_ma10)
                        else:
                            mfi_series = pd.to_numeric(sub["$MFI"], errors="coerce")
                            mfi_recent = mfi_series[mfi_series.index <= end_ts].tail(10)
                            if len(mfi_recent) >= 2:
                                mfi_ma10 = mfi_recent.mean()
                                if mfi is not None and not pd.isna(mfi_ma10):
                                    mfi_day_ok = mfi > float(mfi_ma10)
                except Exception:
                    mfi_day_ok = False
            except Exception:
                dif = dea = dif_prev = dea_prev = rsi = kdj_k = kdj_d = mfi = kdj_k_prev = kdj_d_prev = None
                macd = macd_prev = macd_prev2 = macd_prev3 = None
                ema10 = ema60 = ema120 = None
            if None in (dif, dea, dif_prev, dea_prev):
                continue

            macd_down_2 = (
                macd is not None
                and macd_prev is not None
                and macd_prev2 is not None
                and macd < macd_prev < macd_prev2
            )

            macd_buy = (
                dif > dea
                and dif > 0
                and not macd_down_2
                and vol_ok
                and monthly_ok
                and mfi_day_ok
                and weekly_ok
                and yearly_ok
            )

            buy = macd_buy
            sell = []
            flags[inst] = {"buy": bool(buy), "sell": bool(sell)}
        return flags

    def _get_turnover_mask(self, instruments: List[str], trade_date) -> pd.Series:
        if not instruments:
            return pd.Series(dtype=bool)
        shares = self._load_issued_shares()
        if shares.empty:
            return pd.Series(self.allow_missing_shares, index=pd.Index(instruments, name="instrument"))
        df_all = self._normalize_indicator_df(self._get_indicator_df("day"))
        if df_all.empty or "$volume" not in df_all.columns:
            return pd.Series(self.allow_missing_shares, index=pd.Index(instruments, name="instrument"))
        trade_ts = pd.Timestamp(trade_date)
        mask = pd.Series(False, index=pd.Index(instruments, name="instrument"))
        for inst in instruments:
            try:
                sub = df_all.xs(inst, level="instrument")
            except Exception:
                continue
            try:
                sub = sub[sub.index <= trade_ts]
                if sub.empty:
                    continue
                vol_series = pd.to_numeric(sub["$volume"], errors="coerce")
                med_vol = vol_series.tail(20).median()
                share = shares.get(inst)
                if share is None or pd.isna(share):
                    continue
                turnover = med_vol / share
                mask.loc[inst] = turnover >= float(self.min_turnover)
            except Exception:
                continue
        if self.allow_missing_shares:
            mask = mask.fillna(True)
        else:
            mask = mask.fillna(False)
        return mask

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        # MACD flags on union of holdings and candidate universe
        current_stock_list_all = list(self.trade_position.get_stock_list())

        # take-profit and trailing-stop processing
        forced_sell_amounts: Dict[str, float] = {}
        tp_holdout_add: Set[str] = set()
        tp_holdout_remove: Set[str] = set()
        for code in current_stock_list_all:
            try:
                buy_price = self.trade_position.position.get(code, {}).get("buy_price", None)
            except Exception:
                buy_price = None
            try:
                cur_price = self.trade_exchange.get_deal_price(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=OrderDir.SELL,
                )
            except Exception:
                cur_price = None

            if code in self._tp_holdout:
                # update high watermark and check trailing stop
                if cur_price is not None:
                    prev_high = self._tp_high.get(code, cur_price)
                    self._tp_high[code] = max(prev_high, cur_price)
                    if self._tp_high[code] > 0 and cur_price <= self._tp_high[code] * (1 - self.trailing_stop_drawdown):
                        try:
                            amt = self.trade_position.get_stock_amount(code=code)
                        except Exception:
                            amt = 0
                        if amt and amt > 0:
                            forced_sell_amounts[code] = amt
                            tp_holdout_remove.add(code)
                continue

            if buy_price is not None and cur_price is not None and buy_price > 0:
                if cur_price >= buy_price * (1 + self.take_profit_pct):
                    try:
                        amt = self.trade_position.get_stock_amount(code=code)
                    except Exception:
                        amt = 0
                    if amt and amt > 0:
                        sell_amt = amt * self.take_profit_sell_ratio
                        factor = self.trade_exchange.get_factor(
                            stock_id=code, start_time=trade_start_time, end_time=trade_end_time
                        )
                        sell_amt = self.trade_exchange.round_amount_by_trade_unit(sell_amt, factor)
                        if sell_amt and sell_amt > 0:
                            forced_sell_amounts[code] = sell_amt
                            tp_holdout_add.add(code)
                            if cur_price is not None:
                                self._tp_high[code] = cur_price

        holdout_set = set(self._tp_holdout) | tp_holdout_add
        current_stock_list = [c for c in current_stock_list_all if c not in holdout_set]
        candidates: Set[str] = set(current_stock_list) | set(pred_score.index.tolist())
        macd_flags = self._compute_macd_flags(list(candidates), trade_start_time)
        bullish = {k for k, v in macd_flags.items() if v.get("buy") and k not in holdout_set}
        bearish = {k for k, v in macd_flags.items() if v.get("sell")}

        if self.min_turnover is not None and bullish:
            turnover_mask = self._get_turnover_mask(list(bullish), trade_start_time)
            bullish = {k for k in bullish if bool(turnover_mask.get(k, self.allow_missing_shares))}

        stop_loss_set: Set[str] = set()
        month_df = self._cached_month_df if hasattr(self, "_cached_month_df") else self._normalize_indicator_df(self._get_indicator_df("month"))
        if self.stop_loss_pct is not None:
            for code in current_stock_list_all:
                try:
                    buy_price = self.trade_position.position.get(code, {}).get("buy_price", None)
                except Exception:
                    buy_price = None
                if buy_price is None:
                    continue
                try:
                    cur_price = self.trade_exchange.get_deal_price(
                        stock_id=code,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,
                    )
                except Exception:
                    cur_price = None
                if cur_price is None:
                    continue
                monthly_break = True
                if month_df is not None and not month_df.empty:
                    try:
                        sub_m = month_df.xs(code, level="instrument")
                        if isinstance(sub_m.index, pd.DatetimeIndex):
                            sub_m = sub_m[sub_m.index <= pd.Timestamp(trade_start_time)]
                        if not sub_m.empty:
                            row_m = sub_m.iloc[-1]
                            close_m = row_m.get("$close")
                            ema10_m = row_m.get("$EMA10")
                            if close_m is not None and ema10_m is not None and not pd.isna(close_m) and not pd.isna(ema10_m):
                                monthly_break = float(close_m) < float(ema10_m)
                    except Exception:
                        monthly_break = True
                if cur_price <= buy_price * (1 - self.stop_loss_pct) and monthly_break:
                    stop_loss_set.add(code)

        if not bullish and not stop_loss_set and not forced_sell_amounts:
            return TradeDecisionWO([], self)

        last_scored = pred_score.reindex(current_stock_list).sort_values(ascending=False)
        bearish_holdings = last_scored[last_scored.index.isin(bearish)]
        sell_list = list(stop_loss_set)
        for code in forced_sell_amounts.keys():
            if code not in sell_list:
                sell_list.append(code)
        sell_set = set(sell_list)
        for code in bearish_holdings.sort_values(ascending=True).index:
            if code not in sell_set:
                sell_list.append(code)
                sell_set.add(code)
        if len(sell_list) < self.n_drop:
            remaining = last_scored[~last_scored.index.isin(sell_set)]
            extra_needed = self.n_drop - len(sell_list)
            extra_sell = list(remaining.sort_values(ascending=True).index[:extra_needed])
            sell_list.extend(extra_sell)

        bullish_new = pred_score[pred_score.index.isin(bullish) & ~pred_score.index.isin(current_stock_list)]
        bullish_ranked = bullish_new.sort_values(ascending=False)

        sold_effective = len([c for c in sell_list if c in current_stock_list])
        post_sell_holdings = len(current_stock_list) - sold_effective
        open_slots = max(self.topk - post_sell_holdings, 0)
        max_buys = min(len(sell_list) + open_slots, len(bullish_ranked))
        buy_list = list(bullish_ranked.index[:max_buys]) if max_buys > 0 else []

        if not buy_list and not stop_loss_set and not forced_sell_amounts:
            return TradeDecisionWO([], self)

        if self.only_tradable:
            def is_tradable(code, direction):
                return self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=direction,
                )
        else:
            def is_tradable(code, direction):
                return True

        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()

        time_per_step = self.trade_calendar.get_freq()
        for code in sell_list:
            if not is_tradable(code, None if self.forbid_all_trade_at_limit else OrderDir.SELL):
                continue
            if code not in stop_loss_set and code not in forced_sell_amounts:
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
            if code in forced_sell_amounts:
                sell_amount = forced_sell_amounts[code]
            else:
                sell_amount = current_temp.get_stock_amount(code=code)
            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(sell_order, position=current_temp)
                cash += trade_val - trade_cost

        if self.ramp_steps is not None and self.ramp_steps > 0:
            ramp = min(1.0, self.initial_risk_degree + (1 - self.initial_risk_degree) * (trade_step / self.ramp_steps))
        else:
            ramp = 1.0
        risk_scale = self.initial_risk_degree if trade_step == 0 else ramp
        value = cash * self.risk_degree * risk_scale / len(buy_list) if len(buy_list) > 0 else 0
        for code in buy_list:
            if not is_tradable(code, None if self.forbid_all_trade_at_limit else OrderDir.BUY):
                continue
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price if buy_price not in (None, 0) else 0
            factor = self.trade_exchange.get_factor(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            if buy_amount <= 0:
                continue
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)

        if tp_holdout_add:
            self._tp_holdout.update(tp_holdout_add)
        if tp_holdout_remove:
            for c in tp_holdout_remove:
                self._tp_holdout.discard(c)
                self._tp_high.pop(c, None)

        return TradeDecisionWO(sell_order_list + buy_order_list, self)


class WeightStrategyBase(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        *,
        order_generator_cls_or_obj=OrderGenWOInteract,
        **kwargs,
    ):
        """
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report

            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:

                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.
        """
        super().__init__(**kwargs)

        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator: OrderGenerator = order_generator_cls_or_obj()
        else:
            self.order_generator: OrderGenerator = order_generator_cls_or_obj

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """
        Generate target position from score for this date and the current position.The cash is not considered in the position

        Parameters
        -----------
        score : pd.Series
            pred score for this trade date, index is stock_id, contain 'score' column.
        current : Position()
            current position.
        trade_start_time: pd.Timestamp
        trade_end_time: pd.Timestamp
        """
        raise NotImplementedError()

    def generate_trade_decision(self, execute_result=None):
        # generate_trade_decision
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return TradeDecisionWO([], self)
        current_temp = copy.deepcopy(self.trade_position)
        assert isinstance(current_temp, Position)  # Avoid InfPosition

        target_weight_position = self.generate_target_weight_position(
            score=pred_score, current=current_temp, trade_start_time=trade_start_time, trade_end_time=trade_end_time
        )
        order_list = self.order_generator.generate_order_list_from_target_weight_position(
            current=current_temp,
            trade_exchange=self.trade_exchange,
            risk_degree=self.get_risk_degree(trade_step),
            target_weight_position=target_weight_position,
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        return TradeDecisionWO(order_list, self)


class EnhancedIndexingStrategy(WeightStrategyBase):
    """Enhanced Indexing Strategy

    Enhanced indexing combines the arts of active management and passive management,
    with the aim of outperforming a benchmark index (e.g., S&P 500) in terms of
    portfolio return while controlling the risk exposure (a.k.a. tracking error).

    Users need to prepare their risk model data like below:

    .. code-block:: text

        ├── /path/to/riskmodel
        ├──── 20210101
        ├────── factor_exp.{csv|pkl|h5}
        ├────── factor_cov.{csv|pkl|h5}
        ├────── specific_risk.{csv|pkl|h5}
        ├────── blacklist.{csv|pkl|h5}  # optional

    The risk model data can be obtained from risk data provider. You can also use
    `qlib.model.riskmodel.structured.StructuredCovEstimator` to prepare these data.

    Args:
        riskmodel_path (str): risk model path
        name_mapping (dict): alternative file names
    """

    FACTOR_EXP_NAME = "factor_exp.pkl"
    FACTOR_COV_NAME = "factor_cov.pkl"
    SPECIFIC_RISK_NAME = "specific_risk.pkl"
    BLACKLIST_NAME = "blacklist.pkl"

    def __init__(
        self,
        *,
        riskmodel_root,
        market="csi500",
        turn_limit=None,
        name_mapping={},
        optimizer_kwargs={},
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logger = get_module_logger("EnhancedIndexingStrategy")

        self.riskmodel_root = riskmodel_root
        self.market = market
        self.turn_limit = turn_limit

        self.factor_exp_path = name_mapping.get("factor_exp", self.FACTOR_EXP_NAME)
        self.factor_cov_path = name_mapping.get("factor_cov", self.FACTOR_COV_NAME)
        self.specific_risk_path = name_mapping.get("specific_risk", self.SPECIFIC_RISK_NAME)
        self.blacklist_path = name_mapping.get("blacklist", self.BLACKLIST_NAME)

        self.optimizer = EnhancedIndexingOptimizer(**optimizer_kwargs)

        self.verbose = verbose

        self._riskdata_cache = {}

    def get_risk_data(self, date):
        if date in self._riskdata_cache:
            return self._riskdata_cache[date]

        root = self.riskmodel_root + "/" + date.strftime("%Y%m%d")
        if not os.path.exists(root):
            return None

        factor_exp = load_dataset(root + "/" + self.factor_exp_path, index_col=[0])
        factor_cov = load_dataset(root + "/" + self.factor_cov_path, index_col=[0])
        specific_risk = load_dataset(root + "/" + self.specific_risk_path, index_col=[0])

        if not factor_exp.index.equals(specific_risk.index):
            # NOTE: for stocks missing specific_risk, we always assume it has the highest volatility
            specific_risk = specific_risk.reindex(factor_exp.index, fill_value=specific_risk.max())

        universe = factor_exp.index.tolist()

        blacklist = []
        if os.path.exists(root + "/" + self.blacklist_path):
            blacklist = load_dataset(root + "/" + self.blacklist_path).index.tolist()

        self._riskdata_cache[date] = factor_exp.values, factor_cov.values, specific_risk.values, universe, blacklist

        return self._riskdata_cache[date]

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        trade_date = trade_start_time
        pre_date = get_pre_trading_date(trade_date, future=True)  # previous trade date

        # load risk data
        outs = self.get_risk_data(pre_date)
        if outs is None:
            self.logger.warning(f"no risk data for {pre_date:%Y-%m-%d}, skip optimization")
            return None
        factor_exp, factor_cov, specific_risk, universe, blacklist = outs

        # transform score
        # NOTE: for stocks missing score, we always assume they have the lowest score
        score = score.reindex(universe).fillna(score.min()).values

        # get current weight
        # NOTE: if a stock is not in universe, its current weight will be zero
        cur_weight = current.get_stock_weight_dict(only_stock=False)
        cur_weight = np.array([cur_weight.get(stock, 0) for stock in universe])
        assert all(cur_weight >= 0), "current weight has negative values"
        cur_weight = cur_weight / self.get_risk_degree(trade_date)  # sum of weight should be risk_degree
        if cur_weight.sum() > 1 and self.verbose:
            self.logger.warning(f"previous total holdings excess risk degree (current: {cur_weight.sum()})")

        # load bench weight
        bench_weight = D.features(
            D.instruments("all"), [f"${self.market}_weight"], start_time=pre_date, end_time=pre_date
        ).squeeze()
        bench_weight.index = bench_weight.index.droplevel(level="datetime")
        bench_weight = bench_weight.reindex(universe).fillna(0).values

        # whether stock tradable
        # NOTE: currently we use last day volume to check whether tradable
        tradable = D.features(D.instruments("all"), ["$volume"], start_time=pre_date, end_time=pre_date).squeeze()
        tradable.index = tradable.index.droplevel(level="datetime")
        tradable = tradable.reindex(universe).gt(0).values
        mask_force_hold = ~tradable

        # mask force sell
        mask_force_sell = np.array([stock in blacklist for stock in universe], dtype=bool)

        # optimize
        weight = self.optimizer(
            r=score,
            F=factor_exp,
            cov_b=factor_cov,
            var_u=specific_risk**2,
            w0=cur_weight,
            wb=bench_weight,
            mfh=mask_force_hold,
            mfs=mask_force_sell,
        )

        target_weight_position = {stock: weight for stock, weight in zip(universe, weight) if weight > 0}

        if self.verbose:
            self.logger.info("trade date: {:%Y-%m-%d}".format(trade_date))
            self.logger.info("number of holding stocks: {}".format(len(target_weight_position)))
            self.logger.info("total holding weight: {:.6f}".format(weight.sum()))

        return target_weight_position
