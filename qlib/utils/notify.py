#!/usr/bin/env python3
"""Lightweight Telegram notifier utilities with JSON config loading.

Usage patterns (caller side):
- token/chat id can come from CLI, env (TELEGRAM_TOKEN/CHAT_ID), or a JSON config file.
- If credentials are missing, messages are printed to stdout and execution continues.
- Messages longer than Telegram's limit are split into multiple sends.
"""
import json
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests

TELEGRAM_LIMIT = 4096  # hard limit from Telegram
DEFAULT_MAX_LEN = 3900  # slight headroom for formatting
DEFAULT_NOTIFY_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "notify_config.json")
)


def _split_message(text: str, max_len: int = DEFAULT_MAX_LEN) -> List[str]:
    if len(text) <= max_len:
        return [text]
    parts: List[str] = []
    remaining = text
    while len(remaining) > max_len:
        cut = remaining.rfind("\n", 0, max_len)
        if cut == -1 or cut < max_len // 2:
            cut = max_len
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
        if remaining.startswith("\n"):
            remaining = remaining[1:]
    if remaining:
        parts.append(remaining)
    return parts


def load_notify_config(config_path: Optional[str]) -> Dict[str, str]:
    if not config_path:
        config_path = DEFAULT_NOTIFY_CONFIG_PATH
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_notify_params(token: Optional[str], chat_id: Optional[str], config_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    cfg = load_notify_config(config_path)
    tok = token or cfg.get("telegram_token") or os.getenv("TELEGRAM_TOKEN")
    chat = chat_id or cfg.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID")
    return tok, chat


class TelegramNotifier:
    def __init__(self, token: Optional[str], chat_id: Optional[str], max_len: int = DEFAULT_MAX_LEN, parse_mode: Optional[str] = None):
        self.token = token
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}" if token else None
        self.update_offset = None
        self.max_len = min(max_len, TELEGRAM_LIMIT)
        self.parse_mode = parse_mode  # None by default to avoid Markdown parse issues
        if self.base:
            try:
                resp = requests.get(f"{self.base}/getUpdates", timeout=5)
                if resp.ok:
                    res = resp.json().get("result", [])
                    if res:
                        self.update_offset = res[-1].get("update_id", 0) + 1
            except Exception:
                self.update_offset = None

    def can_send(self) -> bool:
        return bool(self.base and self.chat_id)

    def send(self, text: str):
        chunks = _split_message(text, self.max_len)
        for part in chunks:
            self._send_single(part)

    def _send_single(self, text: str):
        if not self.can_send():
            print(f"[telegram skipped] {text}")
            return
        try:
            payload = {"chat_id": self.chat_id, "text": text}
            if self.parse_mode:
                payload["parse_mode"] = self.parse_mode
            resp = requests.post(f"{self.base}/sendMessage", json=payload, timeout=10)
            if not resp.ok:
                print(f"[telegram error] status={resp.status_code} body={resp.text}: {text}")
                # Retry without Markdown formatting in case of parsing issues
                payload.pop("parse_mode", None)
                retry = requests.post(f"{self.base}/sendMessage", json=payload, timeout=10)
                if not retry.ok:
                    print(f"[telegram error retry] status={retry.status_code} body={retry.text}: {text}")
        except Exception as e:
            print(f"[telegram error] {e}: {text}")

    def send_lines(self, lines: Iterable[str]):
        text = "\n".join([str(x) for x in lines])
        self.send(text)

    def wait_for_confirmation(self, session_id: str, timeout_sec: int) -> bool:
        if not self.can_send():
            print("[interactive] Telegram not configured; denying confirmation")
            return False
        deadline = time.time() + timeout_sec
        self.send(
            f"Confirm session {session_id}?\nReply 'yes {session_id}' to proceed or 'no {session_id}' to cancel."
        )
        while time.time() < deadline:
            try:
                remain = max(1, int(deadline - time.time()))
                params = {"timeout": remain}
                if self.update_offset is not None:
                    params["offset"] = self.update_offset
                resp = requests.get(f"{self.base}/getUpdates", params=params, timeout=remain + 2)
                if not resp.ok:
                    time.sleep(1)
                    continue
                updates = resp.json().get("result", [])
                for upd in updates:
                    self.update_offset = max(self.update_offset or 0, upd.get("update_id", 0) + 1)
                    msg = upd.get("message") or {}
                    txt = str(msg.get("text", "")).lower()
                    if session_id.lower() in txt:
                        if txt.startswith("yes") or " yes" in txt:
                            self.send(f"Session {session_id} confirmed; proceeding.")
                            return True
                        if txt.startswith("no") or " no" in txt or "cancel" in txt:
                            self.send(f"Session {session_id} canceled by user.")
                            return False
            except Exception:
                time.sleep(1)
        self.send(f"Session {session_id} timed out; skipping.")
        return False


__all__ = [
    "TelegramNotifier",
    "load_notify_config",
    "resolve_notify_params",
    "DEFAULT_NOTIFY_CONFIG_PATH",
]
