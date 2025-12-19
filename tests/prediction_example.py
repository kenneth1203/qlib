import argparse
import pandas as pd
import matplotlib.pyplot as plt
import qlib
from qlib.constant import REG_HK
from qlib.data import D
import sys
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Kronos HK 7-day forecast using qlib bin data")
    parser.add_argument("--symbol", default="00001.HK", help="HK ticker, e.g. 00001.HK")
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/hk_data", help="qlib bin data path")
    parser.add_argument("--lookback", type=int, default=300, help="history days fed into Kronos")
    parser.add_argument("--min_history", type=int, default=60, help="minimum clean history required before fallback")
    parser.add_argument("--pred_len", type=int, default=120, help="number of trading days to draw for prediction line")
    parser.add_argument("--device", default="cuda:0", help="Torch device for inference")
    parser.add_argument("--max_context", type=int, default=512, help="Max context tokens for Kronos")
    parser.add_argument("--fill_missing", action="store_true", help="forward/backward fill missing OHLCV before dropping")
    return parser.parse_args()


def load_hk_ohlcv(symbol, provider_uri, lookback, pred_len, min_history, fill_missing=False):
    qlib.init(provider_uri=provider_uri, region=REG_HK)

    cal = D.calendar(start_time=None, end_time=None, freq="day")
    print(cal)
    if len(cal) == 0:
        raise RuntimeError("No trading calendar loaded; check your HK qlib data path")

    need = lookback
    start_idx = max(0, len(cal) - need)
    start_time = cal[start_idx]
    end_time = cal[-1]

    fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
    feat = D.features([symbol], fields, start_time=start_time, end_time=end_time, freq="day", disk_cache=True)
    if feat.empty:
        raise ValueError(f"No data returned for {symbol}. Check symbol or provider_uri: {provider_uri}")
    
    feat = feat.droplevel("instrument")
    feat.index.name = "timestamps"
    feat = feat.rename(columns=lambda x: x.lstrip("$"))
    feat = feat.sort_index()

    cols = ["open", "high", "low", "close", "volume", "amount"]
    base = feat.replace([float("inf"), float("-inf")], pd.NA)

    # If amount is absent or fully NaN, synthesize from close*volume
    if "amount" not in base.columns:
        base["amount"] = base["close"] * base["volume"]
    else:
        need_amount_fill = base["amount"].isna().all()
        if need_amount_fill:
            base["amount"] = base["close"] * base["volume"]
    print(base.tail())
    if fill_missing:
        na_before = base[cols].isna().sum()
        base[cols] = base[cols].ffill().bfill()
        na_after = base[cols].isna().sum()
        print(f"Filled missing OHLCV for {symbol}: {int(na_before.sum())} -> {int(na_after.sum())} gaps")

    cleaned = base.dropna(subset=["open", "high", "low", "close", "volume"])
    dropped = len(base) - len(cleaned)
    if dropped:
        print(f"Dropped {dropped} rows with NaN/inf in OHLCV for {symbol}")

    print(f"Rows available after cleaning for {symbol}: {len(cleaned)} (requested lookback {lookback})")

    if len(cleaned) == 0:
        raise ValueError(f"No clean data for {symbol} after cleaning; check provider_uri or pick another ticker")

    if len(cleaned) < min_history:
        raise ValueError(f"Not enough clean data for {symbol}; need at least {min_history}, got {len(cleaned)}")

    if len(cleaned) < lookback:
        print(f"Using shorter history: requested {lookback}, available {len(cleaned)}")
        return cleaned

    return cleaned.tail(lookback)


def build_future_calendar(last_ts, pred_len):
    return pd.bdate_range(last_ts + pd.Timedelta(days=1), periods=pred_len, freq="B")


def plot_prediction(history_df, pred_df, symbol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(history_df.index, history_df["close"], label="History", color="tab:blue", linewidth=1.4)
    ax1.plot(pred_df.index, pred_df["close"], label="Prediction", color="tab:red", linewidth=1.6)
    ax1.set_title(f"{symbol} - Close price forecast", fontsize=14)
    ax1.set_ylabel("Close", fontsize=12)
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.bar(history_df.index, history_df["volume"], label="History", color="tab:blue", width=0.8)
    ax2.bar(pred_df.index, pred_df["volume"], label="Prediction", color="tab:red", width=0.8, alpha=0.7)
    ax2.set_ylabel("Volume", fontsize=12)
    ax2.legend(loc="best", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    hist_df = load_hk_ohlcv(
        symbol=args.symbol,
        provider_uri=args.provider_uri,
        lookback=args.lookback,
        pred_len=args.pred_len,
        min_history=args.min_history,
        fill_missing=args.fill_missing,
    )

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=args.max_context)

    x_df = hist_df[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = x_df.index.to_series()
    # Build prediction timeline starting from the calendar's last trading day
    cal = D.calendar(start_time=None, end_time=None, freq="day")
    last_cal_day = pd.to_datetime(cal[-1])
    y_bdays = pd.bdate_range(start=last_cal_day, periods=args.pred_len, freq="B")
    y_timestamp = pd.Series(y_bdays)

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )
    pred_df.index = y_timestamp.values

    print("Forecasted Data Head:")
    print(pred_df.head())

    plot_prediction(hist_df, pred_df, args.symbol)


if __name__ == "__main__":
    main()

