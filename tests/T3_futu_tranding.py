#!/usr/bin/env python3
"""
T3 Futu simulated trader

Workflow per run:
- Load decision_<YYYYMMDD>_<recorder_id>.csv; skip trading if missing.
- Fetch simulated positions from Futu OpenD; split manual vs bot via local cache.
- Buy strategy: take top-K rows with buy==True ordered by final_score; place 1-lot buys for symbols not already bot-held.
- Sell strategy: rotate up to N bot-held symbols per day that are buy==False or absent from decision list.
- Orders are placed in simulate env; order monitoring runs until filled or timeout; Telegram notifications for summary and fills/failures.
- Interactive mode requires Telegram confirmation before placing orders; timeout cancels the session.

Notes:
- Requires Futu OpenD running and futu API installed.
- Telegram is optional; if token/chat_id are missing, messages are printed only.
- Bot ownership is tagged via a local cache file; manual holdings are never touched.
"""
import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from qlib.qlib.utils.notify import TelegramNotifier, resolve_notify_params

try:
    from futu import (
        OpenHKTradeContext,
        RET_OK,
        TrdEnv,
        TrdSide,
        OrderType,
    )
except Exception as e:  # pragma: no cover - dependency guard
    OpenHKTradeContext = None  # type: ignore
    RET_OK = -1  # type: ignore
    TrdEnv = None  # type: ignore
    TrdSide = None  # type: ignore
    OrderType = None  # type: ignore
    FUTU_IMPORT_ERROR = e
else:
    FUTU_IMPORT_ERROR = None


@dataclass
class DecisionRow:
    code: str
    buy: bool
    final_score: float

def to_futu_code(inst: str) -> str:
    s = str(inst).upper()
    if s.startswith("HK."):
        s = s.split(".", 1)[1]
    if s.endswith(".HK"):
        s = s.split(".", 1)[0]
    if s.isdigit():
        s = s.zfill(5)
    return f"HK.{s}"


def load_decision(decision_path: str, topk: int) -> List[DecisionRow]:
    if not os.path.exists(decision_path):
        return []
    df = pd.read_csv(decision_path)
    if "instrument" not in df.columns:
        return []
    df["buy"] = df.get("buy", False).astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    df["final_score"] = pd.to_numeric(df.get("final_score", 0.0), errors="coerce").fillna(0.0)
    df = df.sort_values(["buy", "final_score"], ascending=[False, False])
    rows: List[DecisionRow] = []
    for _, r in df.head(max(topk, len(df))).iterrows():
        rows.append(DecisionRow(code=to_futu_code(r["instrument"]), buy=bool(r["buy"]), final_score=float(r["final_score"])))
    return rows


def load_cache(path: str) -> Dict[str, Dict[str, float]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data.get("positions", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache(path: str, positions: Dict[str, Dict[str, float]]):
    data = {"updated": datetime.datetime.utcnow().isoformat() + "Z", "positions": positions}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def split_positions(live_positions: List[dict], bot_cache: Dict[str, Dict[str, float]]):
    manual: List[dict] = []
    bot: List[dict] = []
    for row in live_positions:
        code = str(row.get("code"))
        if code in bot_cache:
            bot.append(row)
        else:
            manual.append(row)
    return manual, bot


def compute_orders(decisions: List[DecisionRow], bot_positions: List[dict], topk: int, rotation_limit: int, lot_size: int):
    decision_map = {d.code: d for d in decisions}
    bot_codes = {p.get("code") for p in bot_positions}
    buy_plan = []
    for d in decisions:
        if not d.buy:
            continue
        if d.code in bot_codes:
            continue
        buy_plan.append({"code": d.code, "qty": lot_size})
        if len(buy_plan) >= topk:
            break
    sell_candidates = []
    for p in bot_positions:
        code = str(p.get("code"))
        d = decision_map.get(code)
        should_sell = (d is None) or (not d.buy)
        if should_sell:
            qty = int(p.get("qty", p.get("can_sell_qty", 0)))
            if qty > 0:
                sell_candidates.append({"code": code, "qty": min(qty, lot_size)})
    sell_plan = sell_candidates[:rotation_limit] if rotation_limit > 0 else []
    return buy_plan, sell_plan


def place_orders(trd_ctx, trd_env, orders: List[dict], side: str, notifier: TelegramNotifier, adjust_limit: int, dry_run: bool) -> List[str]:
    order_ids: List[str] = []
    for od in orders:
        code = od["code"]
        qty = od["qty"]
        if dry_run:
            notifier.send(f"[dry-run] {side} {qty} @ {code}")
            continue
        ret, res = trd_ctx.place_order(
            price=0,
            qty=qty,
            code=code,
            trd_side=TrdSide.BUY if side == "BUY" else TrdSide.SELL,
            order_type=OrderType.MARKET,
            trd_env=trd_env,
            adjust_price=1,
            adjust_limit=adjust_limit,
        )
        if ret != RET_OK:
            notifier.send(f"{side} order failed for {code}: {res}")
            continue
        order_id = str(res.get("order_id")) if isinstance(res, dict) else str(res)
        order_ids.append(order_id)
        notifier.send(f"{side} submitted {code} x{qty}, order_id={order_id}")
    return order_ids


def monitor_orders(trd_ctx, trd_env, order_ids: List[str], notifier: TelegramNotifier, timeout_sec: int, poll_interval: int):
    if not order_ids:
        return
    deadline = time.time() + timeout_sec
    pending = set(order_ids)
    last_status: Dict[str, str] = {}
    while pending and time.time() < deadline:
        try:
            ret, data = trd_ctx.order_list(trd_env=trd_env)
            if ret != RET_OK:
                time.sleep(poll_interval)
                continue
            for _, row in data.iterrows():
                oid = str(row.get("order_id"))
                if oid not in pending:
                    continue
                status = str(row.get("status"))
                last_status[oid] = status
                if status in {"FILLED_ALL", "CANCELLED_ALL", "FAILED", "DISABLED"}:
                    pending.discard(oid)
                    notifier.send(
                        f"Order {oid} done: status={status}, code={row.get('code')}, qty={row.get('qty')}, dealt={row.get('dealt_qty')}, avg_px={row.get('dealt_avg_price')}"
                    )
            time.sleep(poll_interval)
        except Exception:
            time.sleep(poll_interval)
    for oid in list(pending):
        notifier.send(f"Order {oid} still pending after timeout; consider manual check.")


def reconcile_cache(live_positions: List[dict], bot_cache: Dict[str, Dict[str, float]], new_buys: List[dict], sells: List[dict]):
    keep_codes = {od["code"] for od in new_buys} | {od["code"] for od in sells} | set(bot_cache.keys())
    new_cache: Dict[str, Dict[str, float]] = {}
    for p in live_positions:
        code = str(p.get("code"))
        if code not in keep_codes:
            continue
        qty = float(p.get("qty", p.get("can_sell_qty", 0)))
        if qty <= 0:
            continue
        new_cache[code] = {"qty": qty}
    return new_cache


def build_decision_path(target_day: Optional[str], recorder_id: Optional[str]) -> str:
    if target_day is None:
        target_day = datetime.date.today().strftime("%Y%m%d")
    if recorder_id is None:
        return ""
    return f"decision_{target_day}_{recorder_id}.csv"


def main():
    parser = argparse.ArgumentParser(description="Futu simulated trader driven by decision CSV")
    parser.add_argument("--recorder_id", required=True, help="recorder/run id matching decision file")
    parser.add_argument("--target_day", help="target day YYYYMMDD (default today)")
    parser.add_argument("--decision_path", help="optional explicit decision csv path")
    parser.add_argument("--topk", type=int, default=5, help="buy top-K buy==True rows")
    parser.add_argument("--rotation_limit", type=int, default=1, help="max sells per day")
    parser.add_argument("--lot_size", type=int, default=1, help="lots per order")
    parser.add_argument("--host", default=os.getenv("FUTU_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("FUTU_PORT", "11111")))
    parser.add_argument("--mode", choices=["auto", "interactive"], default="auto")
    parser.add_argument("--confirm_timeout", type=int, default=300, help="seconds to wait for telegram confirmation")
    parser.add_argument("--order_timeout", type=int, default=600, help="seconds to wait for orders to finish")
    parser.add_argument("--poll_interval", type=int, default=5, help="seconds between order status polls")
    parser.add_argument("--cache_path", default="bot_positions.json", help="local cache tagging bot-held positions")
    parser.add_argument("--telegram_token", help="telegram bot token")
    parser.add_argument("--telegram_chat_id", help="telegram chat id")
    parser.add_argument("--adjust_limit", type=int, default=5, help="price adjust ticks for marketable orders")
    parser.add_argument("--notify_config", help="path to JSON config with telegram_token/chat_id")
    parser.add_argument("--dry_run", action="store_true", help="log only, do not place orders")
    args = parser.parse_args()

    if FUTU_IMPORT_ERROR:
        raise SystemExit(f"futu not available: {FUTU_IMPORT_ERROR}")

    decision_path = args.decision_path
    if not decision_path:
        decision_path = build_decision_path(args.target_day, args.recorder_id)
    if not decision_path or not os.path.exists(decision_path):
        print(f"Decision file missing; skip trading. path={decision_path}")
        return

    decisions = load_decision(decision_path, args.topk)
    if not decisions:
        print("No decisions loaded; nothing to do.")
        return

    tok, chat = resolve_notify_params(args.telegram_token, args.telegram_chat_id, args.notify_config)
    notifier = TelegramNotifier(tok, chat)
    notifier.send(f"T3 session start: decision={os.path.abspath(decision_path)}, topk={args.topk}")

    session_id = datetime.datetime.utcnow().strftime("T3-%Y%m%d-%H%M%S")
    if args.mode == "interactive":
        if not notifier.wait_for_confirmation(session_id, args.confirm_timeout):
            print("Interactive confirmation failed; exiting.")
            return

    trd_env = TrdEnv.SIMULATE
    with OpenHKTradeContext(host=args.host, port=args.port) as trd_ctx:
        ret, _ = trd_ctx.unlock_trade("", True) if hasattr(trd_ctx, "unlock_trade") else (RET_OK, None)  # noqa: SIM115
        if ret != RET_OK:
            raise SystemExit("Failed to unlock trade; ensure OpenD is configured for simulate env.")

        pos_ret, live_pos_df = trd_ctx.position_list(trd_env=trd_env)
        if pos_ret != RET_OK:
            raise SystemExit(f"position_list failed: {live_pos_df}")
        live_positions = live_pos_df.to_dict("records")

        bot_cache = load_cache(args.cache_path)
        manual_positions, bot_positions = split_positions(live_positions, bot_cache)
        buy_plan, sell_plan = compute_orders(decisions, bot_positions, args.topk, args.rotation_limit, args.lot_size)

        notifier.send(
            f"Plan:\nManual holdings: {len(manual_positions)} untouched\nBot holdings: {len(bot_positions)}\nSells: {sell_plan}\nBuys: {buy_plan}"
        )

        if not buy_plan and not sell_plan:
            notifier.send("Nothing to trade today.")
            return

        sell_ids = place_orders(trd_ctx, trd_env, sell_plan, "SELL", notifier, args.adjust_limit, args.dry_run)
        buy_ids = place_orders(trd_ctx, trd_env, buy_plan, "BUY", notifier, args.adjust_limit, args.dry_run)

        monitor_orders(trd_ctx, trd_env, sell_ids + buy_ids, notifier, args.order_timeout, args.poll_interval)

        pos_ret, live_pos_df = trd_ctx.position_list(trd_env=trd_env)
        if pos_ret == RET_OK:
            live_positions = live_pos_df.to_dict("records")
            new_cache = reconcile_cache(live_positions, bot_cache, buy_plan, sell_plan)
            if not args.dry_run:
                save_cache(args.cache_path, new_cache)
        notifier.send("T3 session complete.")


if __name__ == "__main__":
    main()
