#!/usr/bin/env python3
"""
LLM-assisted stock advice generator via OpenRouter.

For a given HK stock code, this script gathers:
- 1y OHLCV from qlib
- cached company fundamentals/news from ~/.qlib/qlib_data/hk_data/news/{code}.json
- today's trending news buckets in the same folder
- decision scores from a decision_YYYYMMDD_<recorder>.csv row

It then prompts a multi-role analyst panel (technical, fundamental, news, bull vs bear debate,
manager decision, final trading advisor) through OpenRouter and optionally sends the summary via Telegram.
"""

import argparse
import json
import os
import sys
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import time
import random
import requests

import qlib
from qlib.constant import REG_HK
from qlib.data import D
from qlib.utils.func import to_qlib_inst
from qlib.utils.notify import TelegramNotifier, resolve_notify_params, load_notify_config

# Ensure stdout is UTF-8 on Windows to avoid GBK-related mojibake when printing Chinese
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
TRENDING_FILES = ["finance_today.json", "media_today.json", "social_today.json", "tech_today.json"]
STAGE_ORDER = ["technical", "fundamental", "news", "social", "bull", "bear", "manager", "trade"]


def _cache_path(cache_dir: Path, code: str, stage: str, model: str) -> Path:
	# store caches under per-code subfolders: cache_dir/<safe_code>/<stage>__<model>.txt
	cache_dir.mkdir(parents=True, exist_ok=True)
	safe_code = code.replace("/", "_").replace("\\", "_")
	safe_stage = stage.replace("/", "_")
	safe_model = model.replace("/", "_").replace(":", "_")
	subdir = cache_dir / safe_code
	# ensure the per-code folder exists
	subdir.mkdir(parents=True, exist_ok=True)
	return subdir / f"{safe_stage}__{safe_model}.txt"


def _normalize_display_code(code: str) -> str:
	"""Normalize various code inputs into 5-digit.HK display form.

	Examples: '700' -> '00700.HK', '00700' -> '00700.HK', '00700.HK' -> '00700.HK'
	"""
	s = str(code).strip()
	# if already in DDDDD.HK form, return as-is
	import re as _re

	if _re.match(r"^\d{5}\.HK$", s):
		return s
	digits = ''.join([c for c in s if c.isdigit()])
	if not digits:
		return s
	return digits.zfill(5) + ".HK"


def load_stage_cache(cache_dir: Path, code: str, stage: str, model: str) -> Optional[str]:
	path = _cache_path(cache_dir, code, stage, model)
	if path.exists():
		print(f"[cache] hit stage={stage} path={path}")
		try:
			return path.read_text(encoding="utf-8")
		except Exception:
			print(f"[cache] read failed stage={stage} path={path}")
			return None
	return None


def save_stage_cache(cache_dir: Path, code: str, stage: str, model: str, text: str) -> None:
	path = _cache_path(cache_dir, code, stage, model)
	# For portfolio aggregation runs (batch mode), avoid overwriting existing cache in the same day
	if stage == "portfolio":
		stamp = datetime.date.today().strftime("%Y%m%d")
		suffix = int(time.time())
		alt = path.with_name(f"{stage}_{stamp}_{suffix}__{model.replace('/', '_').replace(':', '_')}.txt")
		path = alt
	try:
		path.write_text(str(text), encoding="utf-8")
		print(f"[cache] saved stage={stage} path={path}")
	except Exception:
		print(f"[cache] save failed stage={stage} path={path}")
		pass


def load_portfolio_list(path: Path) -> List[Any]:
	if not path.exists():
		raise FileNotFoundError(f"portfolio file not found: {path}")
	txt = path.read_text(encoding="utf-8").strip()
	if not txt:
		return []
	# Accept simple plain-text list (one code per line). If file is a JSON list, accept that too.
	first = txt.lstrip()[0]
	if first == "[":
		try:
			data = json.loads(txt)
			if isinstance(data, list):
				return data
		except Exception:
			pass
	# otherwise treat as plain lines
	lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
	return lines


def assemble_portfolio_items(raw_items: List[Any], cache_dir: Path, model: str) -> Tuple[List[Dict[str, str]], List[str]]:
	items: List[Dict[str, str]] = []
	missing: List[str] = []
	for entry in raw_items:
		if isinstance(entry, str):
			code = entry
			summary = load_stage_cache(cache_dir, code, "summary", model)
			if summary is None:
				missing.append(code)
				continue
			trade = load_stage_cache(cache_dir, code, "trade", model)
			items.append({"code": code, "summary": summary, "trade": trade or ""})
		elif isinstance(entry, dict):
			code = entry.get("code") or entry.get("inst") or entry.get("symbol")
			if not code:
				continue
			summary = entry.get("summary") or load_stage_cache(cache_dir, code, "summary", model)
			trade = entry.get("trade") or load_stage_cache(cache_dir, code, "trade", model)
			if not summary:
				missing.append(code)
				continue
			items.append({"code": code, "summary": summary, "trade": trade or ""})
	return items, missing


def _read_json(path: Path) -> Any:
	if not path.exists():
		return None
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return None


def _parse_rating_date(s: Any) -> Optional[datetime.date]:
	"""Parse various date string formats to date; return None if invalid."""
	if s is None:
		return None
	try:
		txt = str(s).strip()
		if not txt:
			return None
		# Common formats seen on sites
		fmts = [
			"%Y-%m-%d",
			"%Y/%m/%d",
			"%Y-%m-%d %H:%M",
			"%Y/%m/%d %H:%M",
			"%Y-%m-%d %H:%M:%S",
			"%Y/%m/%d %H:%M:%S",
		]
		for fmt in fmts:
			try:
				return datetime.datetime.strptime(txt, fmt).date()
			except Exception:
				pass
		# Fallback to pandas if available
		try:
			ts = pd.to_datetime(txt, errors="coerce")
			if pd.isna(ts):
				return None
			return ts.date()
		except Exception:
			return None
	except Exception:
		return None


def _select_instrument(row_inst: str) -> str:
	try:
		return to_qlib_inst(row_inst)
	except Exception:
		return str(row_inst)


def load_decision_row(decision_path: Path, code: str) -> Dict[str, Any]:
	if not decision_path.exists():
		raise FileNotFoundError(f"Decision file missing: {decision_path}")
	df = pd.read_csv(decision_path)
	df.columns = [c.strip() for c in df.columns]
	target = to_qlib_inst(code)
	df["_inst"] = df.iloc[:, 0].map(_select_instrument)
	row = df.loc[df["_inst"] == target]
	if row.empty:
		# try without suffix
		bare = target.replace(".HK", "")
		row = df[df["_inst"].str.replace(".HK", "", regex=False) == bare]
	if row.empty:
		print(f"[warn] Instrument {code} not found in {decision_path}; using zeroed decision data")
		return {
			"_inst": target,
			"model_score": 0,
			"kronos_score": 0,
			"bull_score": 0,
			"bear_score": 0,
			"net_score": 0,
			"net_c": 0,
			"final_score": 0,
			"streak_days": 0,
			"avg_dollar_vol": 0,
			"buy": False,
			"final_score_ranking": None,
		}
	print(f"[info] loaded decision row for {target} from {decision_path}")
	# compute 1-based model ranking position within the CSV (if available)
	final_score_ranking = None
	try:
		pos_df = df.reset_index(drop=True)
		matches = pos_df[pos_df["_inst"] == target]
		if matches.empty:
			# try without .HK suffix
			bare = target.replace(".HK", "")
			matches = pos_df[pos_df["_inst"].str.replace(".HK", "", regex=False) == bare]
		if not matches.empty:
			final_score_ranking = int(matches.index[0]) + 1
	except Exception:
		final_score_ranking = None
	result = row.iloc[0].to_dict()
	if final_score_ranking is not None:
		result["final_score_ranking"] = final_score_ranking
	return result


def load_company_news(news_dir: Path, code: str) -> Dict[str, Any]:
	primary = _read_json(news_dir / f"{code}.json") or {}
	trending: Dict[str, Any] = {}
	for fname in TRENDING_FILES:
		trending[fname] = _read_json(news_dir / fname) or []
	return {"primary": primary, "trending": trending}


def summarize_prices(code: str, provider_uri: str, lookback_days: int = 400) -> Dict[str, Any]:
	qlib.init(provider_uri=os.path.expanduser(provider_uri), region=REG_HK)
	print(f"[info] qlib initialized provider={provider_uri} region=HK")
	end_dt = datetime.date.today()
	start_dt = end_dt - datetime.timedelta(days=lookback_days)
	inst = to_qlib_inst(code)
	try:
		df = D.features([inst], ["$open", "$high", "$low", "$close", "$volume"], start_time=str(start_dt), end_time=str(end_dt), freq="day")
	except Exception as e:
		raise RuntimeError(f"Failed to fetch qlib data for {inst}: {e}")
	if df.empty:
		raise RuntimeError(f"No qlib data for {inst}")
	print(f"[info] fetched ohlcv rows={len(df)} from {start_dt} to {end_dt}")
	# qlib may return MultiIndex columns like (instrument, feature).
	# Normalize to a DataFrame with exactly the OHLCV columns we need.
	orig_cols = list(df.columns)
	# attempt to find feature names in last level
	flat_cols = []
	for c in orig_cols:
		if isinstance(c, tuple):
			flat_cols.append(c[-1])
		else:
			flat_cols.append(c)
	# candidate feature names we expect (with and without $)
	need = ["$open", "$high", "$low", "$close", "$volume"]
	alt_need = [n.lstrip("$") for n in need]
	col_map = {}
	for want in need:
		if want in flat_cols:
			idx = flat_cols.index(want)
			col_map[want] = orig_cols[idx]
		else:
			# try without $
			w2 = want.lstrip("$")
			if w2 in flat_cols:
				idx = flat_cols.index(w2)
				col_map[want] = orig_cols[idx]
	# If we didn't find all, try fuzzy matching by suffix
	for want in need:
		if want not in col_map:
			for i, fc in enumerate(flat_cols):
				if isinstance(fc, str) and fc.lower().endswith(want.lstrip("$").lower()):
					col_map[want] = orig_cols[i]
					break
	# Ensure we have mapping for all required columns
	if len(col_map) < len(need):
		raise RuntimeError(f"Could not locate all OHLCV columns in qlib result. cols={flat_cols}")
	# select and rename
	df2 = df.loc[:, [col_map[n] for n in need]].copy()
	df2.columns = ["open", "high", "low", "close", "volume"]
	df = df2.reset_index().set_index("datetime").sort_index()
	close = df["close"].astype(float)
	high = df["high"].astype(float)
	low = df["low"].astype(float)
	open_ = df["open"].astype(float)
	volume = df["volume"].astype(float)
	stats = {
		"last_date": close.index.max().strftime("%Y-%m-%d"),
		"last_close": float(close.iloc[-1]),
		"return_5d": float((close.iloc[-1] / close.tail(5).iloc[0]) - 1.0) if len(close) >= 5 else np.nan,
		"return_20d": float((close.iloc[-1] / close.tail(20).iloc[0]) - 1.0) if len(close) >= 20 else np.nan,
		"return_1y": float((close.iloc[-1] / close.iloc[0]) - 1.0),
		"volatility_20d": float(close.pct_change().tail(20).std()),
	}
	ma_short = close.rolling(20).mean()
	ma_mid = close.rolling(60).mean()
	ma_long = close.rolling(120).mean()
	stats["ma_trend"] = {
		"ma20": float(ma_short.iloc[-1]) if not np.isnan(ma_short.iloc[-1]) else None,
		"ma60": float(ma_mid.iloc[-1]) if not np.isnan(ma_mid.iloc[-1]) else None,
		"ma120": float(ma_long.iloc[-1]) if not np.isnan(ma_long.iloc[-1]) else None,
		"ma20_gt_ma60": bool(ma_short.iloc[-1] > ma_mid.iloc[-1]) if not np.isnan(ma_short.iloc[-1]) and not np.isnan(ma_mid.iloc[-1]) else None,
		"ma60_gt_ma120": bool(ma_mid.iloc[-1] > ma_long.iloc[-1]) if not np.isnan(ma_mid.iloc[-1]) and not np.isnan(ma_long.iloc[-1]) else None,
	}

	# --- RSI(14) ---
	try:
		delta = close.diff()
		up = delta.clip(lower=0.0)
		down = -delta.clip(upper=0.0)
		rs_up = up.rolling(window=14).mean()
		rs_down = down.rolling(window=14).mean()
		rs = rs_up / rs_down
		rsi = 100.0 - (100.0 / (1.0 + rs))
		stats["rsi_14"] = float(rsi.iloc[-1]) if len(rsi) > 0 and not np.isnan(rsi.iloc[-1]) else None
	except Exception:
		stats["rsi_14"] = None

	# --- ATR(14) ---
	try:
		prev_close = close.shift(1)
		tr1 = high - low
		tr2 = (high - prev_close).abs()
		tr3 = (low - prev_close).abs()
		tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
		atr = tr.rolling(window=14).mean()
		stats["atr_14"] = float(atr.iloc[-1]) if len(atr) > 0 and not np.isnan(atr.iloc[-1]) else None
	except Exception:
		stats["atr_14"] = None

	# --- MACD(12,26,9) ---
	try:
		ema12 = close.ewm(span=12, adjust=False).mean()
		ema26 = close.ewm(span=26, adjust=False).mean()
		macd_line = ema12 - ema26
		signal = macd_line.ewm(span=9, adjust=False).mean()
		hist = macd_line - signal
		stats["macd"] = {
			"macd": float(macd_line.iloc[-1]) if len(macd_line) > 0 and not np.isnan(macd_line.iloc[-1]) else None,
			"signal": float(signal.iloc[-1]) if len(signal) > 0 and not np.isnan(signal.iloc[-1]) else None,
			"hist": float(hist.iloc[-1]) if len(hist) > 0 and not np.isnan(hist.iloc[-1]) else None,
			"hist_mean_20": float(hist.tail(20).mean()) if len(hist) > 0 else None,
			"hist_std_20": float(hist.tail(20).std()) if len(hist) > 0 else None,
			"hist_positive": bool(hist.iloc[-1] > 0) if len(hist) > 0 and not np.isnan(hist.iloc[-1]) else None,
		}
	except Exception:
		stats["macd"] = {"macd": None, "signal": None, "hist": None, "hist_mean_20": None, "hist_std_20": None, "hist_positive": None}

	# --- KDJ (9,3,3) ---
	try:
		low_n = low.rolling(window=9).min()
		high_n = high.rolling(window=9).max()
		den = (high_n - low_n)
		rsv = (close - low_n) / den * 100.0
		rsv = rsv.fillna(0.0)
		K = rsv.ewm(alpha=1/3, adjust=False).mean()
		KD = K.ewm(alpha=1/3, adjust=False).mean()
		J = 3.0 * K - 2.0 * D
		stats["kdj"] = {
			"k": float(K.iloc[-1]) if len(K) > 0 and not np.isnan(K.iloc[-1]) else None,
			"d": float(KD.iloc[-1]) if len(KD) > 0 and not np.isnan(KD.iloc[-1]) else None,
			"j": float(J.iloc[-1]) if len(J) > 0 and not np.isnan(J.iloc[-1]) else None,
		}
	except Exception:
		stats["kdj"] = {"k": None, "d": None, "j": None}
	tail = df.tail(365).copy()
	tail.index = tail.index.strftime("%Y-%m-%d")
	return {"stats": stats, "recent_ohlcv": tail}


def _pick_finance_snippets(company_blob: Dict[str, Any]) -> Dict[str, Any]:
	if not company_blob:
		return {}
	company = company_blob.get("company", {}) if isinstance(company_blob.get("company"), dict) else {}
	profile = company.get("profile", {}) if isinstance(company.get("profile"), dict) else {}
	review = company.get("review", "") if isinstance(company, dict) else ""
	outlook = company.get("outlook", "") if isinstance(company, dict) else ""
	finance = company_blob.get("finance", {}) if isinstance(company_blob.get("finance"), dict) else {}
	dividends = company_blob.get("dividends", []) if isinstance(company_blob.get("dividends"), list) else []
	rights = company_blob.get("rights", {}) if isinstance(company_blob.get("rights"), dict) else {}
	major_holder_changes_raw = rights.get("major_holder_changes", []) if isinstance(rights.get("major_holder_changes"), list) else []
	# keep only last 1 year of major holder changes; fallback to latest 10 by date if none
	cutoff_mh = datetime.date.today() - datetime.timedelta(days=365)
	major_holder_changes: List[Dict[str, Any]] = []
	for r in major_holder_changes_raw:
		if isinstance(r, dict):
			dt = _parse_rating_date(r.get("date"))
			if dt and dt >= cutoff_mh:
				major_holder_changes.append(r)
	if not major_holder_changes and major_holder_changes_raw:
		def _key_mh(x):
			dt = _parse_rating_date(x.get("date")) if isinstance(x, dict) else None
			return dt or datetime.date.min
		major_holder_changes = sorted([r for r in major_holder_changes_raw if isinstance(r, dict)], key=_key_mh, reverse=True)[:10]
	share_buyback = rights.get("share_buyback", []) if isinstance(rights.get("share_buyback"), list) else []
	ratings_raw = company_blob.get("ratings", []) if isinstance(company_blob.get("ratings"), list) else []
	# keep only last 60 days of ratings; fallback to latest 5 by date if none
	cutoff = datetime.date.today() - datetime.timedelta(days=60)
	ratings: List[Dict[str, Any]] = []
	for r in ratings_raw:
		if isinstance(r, dict):
			dt = _parse_rating_date(r.get("rating_date"))
			if dt and dt >= cutoff:
				ratings.append(r)
	if not ratings and ratings_raw:
		# fallback: sort by rating_date desc and take latest 5
		def _key_r(x):
			dt = _parse_rating_date(x.get("rating_date")) if isinstance(x, dict) else None
			return dt or datetime.date.min
		ratings = sorted([r for r in ratings_raw if isinstance(r, dict)], key=_key_r, reverse=True)[:5]
	return {
		"profile": profile,
		"review": review,
		"outlook": outlook,
		"finance_standard": finance.get("finance_standard", []),
		"finance_status": finance.get("finance_status", []),
		"balance_sheet": finance.get("balance_sheet", []),
		"cash_flow": finance.get("cash_flow", []),
		"dividends": dividends,
		"rights_major_holder_changes": major_holder_changes,
		"rights_share_buyback": share_buyback,
		"ratings": ratings,
	}


def _format_decision_context(decision_row: Dict[str, Any]) -> str:
	keys = [
		"final_score_ranking",
		"model_score",
		"kronos_score",
		"bull_score",
		"bear_score",
		"net_score",
		"net_c",
		"final_score",
		"streak_days",
		"avg_dollar_vol",
		"buy",
	]
	parts = []
	for k in keys:
		if k in decision_row:
			parts.append(f"{k}={decision_row.get(k)}")
	return "; ".join(parts)


def _recent_ohlcv_lines(price_info: Dict[str, Any], days: int = 10) -> List[str]:
	recent = price_info.get("recent_ohlcv")
	if not isinstance(recent, pd.DataFrame):
		return []
	tail = recent.tail(days)
	lines: List[str] = []
	for idx, row in tail.iterrows():
		def _fmt(x):
			try:
				return f"{float(x):.2f}"
			except Exception:
				return "NA"

		vol = row.get("volume") if isinstance(row, (dict, pd.Series)) else None
		if pd.notna(vol):
			try:
				vol_str = str(int(vol))
			except Exception:
				try:
					vol_str = str(int(float(vol)))
				except Exception:
					vol_str = "NA"
		else:
			vol_str = "NA"

		lines.append(
			f"{idx}: o={_fmt(row.get('open'))} h={_fmt(row.get('high'))} l={_fmt(row.get('low'))} c={_fmt(row.get('close'))} v={vol_str}"
		)
	return lines


def build_technical_messages(code: str, decision_row: Dict[str, Any], price_info: Dict[str, Any]) -> List[Dict[str, str]]:
	stats = price_info.get("stats", {})
	ma = stats.get("ma_trend", {})

	def _fmt_num(val: Any, fmt: str = ".2f") -> str:
		"""Format numeric values safely for prompt strings."""
		try:
			fval = float(val)
			if np.isnan(fval):
				return "NA"
			return format(fval, fmt)
		except Exception:
			return "NA"

	macd_stats = stats.get("macd", {}) or {}
	kdj_stats = stats.get("kdj", {}) or {}
	rsi_str = _fmt_num(stats.get("rsi_14"))
	atr_str = _fmt_num(stats.get("atr_14"))
	macd_str = " ".join(
		[
			f"macd={_fmt_num(macd_stats.get('macd'))}",
			f"sig={_fmt_num(macd_stats.get('signal'))}",
			f"hist={_fmt_num(macd_stats.get('hist'))}",
			f"hist_mean20={_fmt_num(macd_stats.get('hist_mean_20'))}",
			f"hist_std20={_fmt_num(macd_stats.get('hist_std_20'))}",
			f"hist_pos={macd_stats.get('hist_positive')}",
		]
	)
	kdj_str = f"K={_fmt_num(kdj_stats.get('k'))} D={_fmt_num(kdj_stats.get('d'))} J={_fmt_num(kdj_stats.get('j'))}"
	scoring = (
		"final_score = (w_model * model_score + w_net * net_c) * kronos_score * 100; "
		"model_score = Qlib預測模型分數(0-1); 在topk20名單中排名越前分數越高; "
		"net_c = 1/(1+exp(-net_score/3)), net_score=bull_score-bear_score; bull_score 和 bear_score 分別是技術指標中的多頭和空頭得分; "
		"kronos_score = 1/(1+exp(-alpha*r)), alpha=5;  kronos 為預測未來14日股價走勢，數據靠近0.5為中性。"
		"streak_days=qlib預測TopK連續出現天數; avg_dollar_vol=平均成交金額; buy=True 表示通過基本篩選。"
	)
	price_summary = (
		f"最後交易日: {stats.get('last_date')} 收盤: {stats.get('last_close'):.4f}"
		f" | 5日: {stats.get('return_5d', np.nan):.2%} 20日: {stats.get('return_20d', np.nan):.2%}"
		f" | 1年: {stats.get('return_1y', np.nan):.2%} 波動率20日: {stats.get('volatility_20d', np.nan):.2%}"
		f" | MA20={ma.get('ma20')} MA60={ma.get('ma60')} MA120={ma.get('ma120')}"
		f" | MA20>MA60={ma.get('ma20_gt_ma60')} MA60>MA120={ma.get('ma60_gt_ma120')}"
		f" | RSI14={rsi_str} ATR14={atr_str}"
		f" | MACD({macd_str})"
		f" | KDJ({kdj_str})"
	)
	recent_lines = _recent_ohlcv_lines(price_info, days=180)
	msg = [
		{
			"role": "system",
			"content": "你是股票市場技術分析師，聚焦K線形態、技術指標、趨勢結構、超賣超買、支撐阻力、動能分析、量價結構、風險點，並給出買賣時機初步判斷。",
		},
		{"role": "system", "content": scoring},
		{
			"role": "user",
			"content": "\n".join(
				[
					f"股票代號: {code}",
					f"決策數據: {_format_decision_context(decision_row)}",
					f"價格摘要: {price_summary}",
					"最近180日OHLCV:" if recent_lines else "",
					"\n".join(recent_lines),
					"請輸出 JSON: {technical_report, signals, risk, final_score_ranking}. signals 包含買/賣/觀望和理由。",
				]
			),
		},
	]
	#print(f"[debug] technical messages: {msg}")
	return msg

def build_fundamental_messages(code: str, decision_row: Dict[str, Any], finance_snip: Dict[str, Any]) -> List[Dict[str, str]]:
	msg = [
		{
			"role": "system",
			"content": "你是股票基本面分析師，聚焦盈利能力、成長、估值、財務風險與催化，給出估值區間與結論。",
		},
		{
			"role": "user",
			"content": "\n".join(
				[
					f"股票代號: {code}",
					f"決策數據: {_format_decision_context(decision_row)}",
					f"公司概況: {finance_snip.get('profile', {})}",
					f"管理層回顧(review): {finance_snip.get('review', '')}",
					f"展望(outlook): {finance_snip.get('outlook', '')}",
					f"財務指標(完整 finance_standard): {finance_snip.get('finance_standard', [])}",
					f"財務狀況(finance_status): {finance_snip.get('finance_status', [])}",
					f"資產負債表(balance_sheet): {finance_snip.get('balance_sheet', [])}",
					f"現金流量表(cash_flow): {finance_snip.get('cash_flow', [])}",
					f"分紅派息(dividends): {finance_snip.get('dividends', [])}",
					f"權益變動-主要持有人(major_holder_changes): {finance_snip.get('rights_major_holder_changes', [])}",
					f"權益變動-股份回購(share_buyback): {finance_snip.get('rights_share_buyback', [])}",
					f"投行評級(ratings): {finance_snip.get('ratings', [])}",
					"請輸出 JSON: {company_name_chinese, fundamental_report, valuation_view, risk}. 務必先總結重點，再給結論。",
				]
			),
		},
	]
	#print(f"[debug] fundamental messages: {msg}")
	return msg


def _format_trending_items(items: Any) -> List[str]:
	lines: List[str] = []
	if isinstance(items, list) and items:
		for it in items:
			if isinstance(it, dict):
				title = it.get("title") or it.get("headline") or str(it)
				time = it.get("time") or it.get("date") or ""
				lines.append(f"{time}: {title}" if time else str(title))
			else:
				lines.append(str(it))
	return lines


def build_news_messages(code: str, news_blob: Dict[str, Any]) -> List[Dict[str, str]]:
	primary_news = news_blob.get("primary", {})
	company_news = primary_news.get("news", []) if isinstance(primary_news, dict) else []
	news_lines = [f"{n.get('time','')}: {n.get('title','')}" for n in company_news]
	trending_all = news_blob.get("trending", {}) or {}
	finance_lines = _format_trending_items(trending_all.get("finance_today.json")) if isinstance(trending_all, dict) else []
	msg = [
		{
			"role": "system",
			"content": "你是新聞與舆情分析師，判斷新聞對股價的潛在正負影響、情緒強度與持續性，給出催化/風險清單。",
		},
		{
			"role": "user",
			"content": "\n".join(
				[
					f"股票代號: {code}",
					"全部公司新聞:" if news_lines else "公司新聞: 無", "\n".join(news_lines),
					"finance_today 全部新聞:" if finance_lines else "finance_today: 無", "\n".join(finance_lines),
					"請輸出 JSON: {news_impact, sentiment, catalysts, risks}. sentiment 需標示偏多/偏空與強度。",
				]
			),
		},
	]
	#print(f"[debug] news messages: {msg}")
	return msg


def build_social_messages(code: str, news_blob: Dict[str, Any], finance_snip: Dict[str, Any]) -> List[Dict[str, str]]:
	trending_all = news_blob.get("trending", {}) or {}
	media_lines = _format_trending_items(trending_all.get("media_today.json")) if isinstance(trending_all, dict) else []
	social_lines = _format_trending_items(trending_all.get("social_today.json")) if isinstance(trending_all, dict) else []
	tech_lines = _format_trending_items(trending_all.get("tech_today.json")) if isinstance(trending_all, dict) else []
	company_brief = finance_snip.get("profile", {}) if isinstance(finance_snip, dict) else {}
	brief_line = f"公司簡介: {company_brief}" if company_brief else "公司簡介: N/A"
	msg = [
		{
			"role": "system",
			"content": "你是社媒與輿情分析師，處理當日媒體/社交/科技熱點。請先判斷每則是否與該公司業務或催化相關，標註關聯度與理由；對無關內容要明確標示無關。然後評估整體熱度傳播、情緒方向與1-10天內對股價的可能影響。",
		},
		{
			"role": "user",
			"content": "\n".join(
				[
					f"股票代號: {code}",
					brief_line,
					"媒體新聞(media_today):" if media_lines else "媒體新聞: 無", "\n".join(media_lines),
					"社交熱點(social_today):" if social_lines else "社交熱點: 無", "\n".join(social_lines),
					"科技/行業(tech_today):" if tech_lines else "科技/行業: 無", "\n".join(tech_lines),
					"請輸出 JSON: {hot_topics, sentiment, short_term_price_impact, risk_flags}. hot_topics 需列關聯度與理由；sentiment 需標示方向與強度，短期=1-10天。",
				]
			),
		},
	]
	#print(f"[debug] social messages: {msg}")
	return msg


def build_portfolio_messages(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
	lines = []
	for it in items:
		code = it.get("code", "")
		summary = it.get("summary", "")
		lines.append(f"{code}: {summary}")
	scoring = (
		"排序邏輯: 先對每檔打短期技術信號分、中期基本面改善分、長期估值空間分，再做風險調整(波動、流動性、負面事件)，並結合 bull/bear 置信度。"
	)
	msg = [
		{"role": "system", "content": "你是投資組合評審員，根據多支股票的摘要做綜合排序並輸出JSON格式，使用繁體中文，不要刪減任何一支股票。"},
		{
			"role": "user",
			"content": "\n".join(
				[
					scoring,
					"請從每支股票的 summary 抽取技術面、基本面、新聞情緒、bull/bear置信度，輸出綜合排序表 (JSON 格式)。",
					"股票清單摘要:",
					"\n".join(lines),
					"請輸出: JSON {ranking: [{code, company_name_chinese, decision, entry_point, hold_time, take_profit, support_position, stop_loss, rationale, final_score_ranking, score: {tech_score, fundamental_score, valuation_score, sentiment, bull_conf, bear_conf, risk_adjust, total_score}}], top_picks, notes}。",
					"以上JSON欄位說明包括但不限於：company_name_chinese=於港股巿場中文名稱；decision=買入/觀望/賣出；entry_point=建議進場點;hold_time=建議持倉時間(短中長期)；take_profit=建議止盈點(三階段)；support_position=股價支撐位(多個)；stop_loss=建議止損點；rationale=決策理由摘要；top_picks=首選標的清單；notes=其他說明。",
				]
			),
		},
	]
	#print(f"[debug] portfolio messages: {msg}")
	return msg

def build_bull_messages(tech: str, fund: str, news: str, social: str, decision_row: Dict[str, Any]) -> List[Dict[str, str]]:
	content = "\n".join(
		[
			"多頭研究員：請基於四份報告進行多頭論證，列出看漲理由、關鍵數據、潛在風險與反駁，給出結論。",
			f"技術分析: {tech}",
			f"基本面: {fund}",
			f"新聞舆情: {news}",
			f"社媒輿情: {social}",
			f"決策數據: {_format_decision_context(decision_row)}",
			"輸出 JSON: {bull_view, risks, confidence (0-1)}。",
		]
	)
	#print(f"[debug] bull content: {content}")
	return [{"role": "system", "content": "你是多頭研究員，需站在看漲立場進行辯論。"}, {"role": "user", "content": content}]


def build_bear_messages(tech: str, fund: str, news: str, social: str, decision_row: Dict[str, Any]) -> List[Dict[str, str]]:
	content = "\n".join(
		[
			"空頭研究員：請基於四份報告進行空頭論證，列出看跌理由、下行催化、風險對沖，給出結論。",
			f"技術分析: {tech}",
			f"基本面: {fund}",
			f"新聞舆情: {news}",
			f"社媒輿情: {social}",
			f"決策數據: {_format_decision_context(decision_row)}",
			"輸出 JSON: {bear_view, upside_risks, confidence (0-1)}。",
		]
	)
	#print(f"[debug] bear content: {content}")
	return [{"role": "system", "content": "你是空頭研究員，需站在看跌立場進行辯論。"}, {"role": "user", "content": content}]


def build_manager_messages(bull: str, bear: str) -> List[Dict[str, str]]:
	content = "\n".join(
		[
			"研究經理：綜合多空觀點，評估置信度，做出初步投資建議 (買入/觀望/賣出) 與理由。",
			f"多頭觀點: {bull}",
			f"空頭觀點: {bear}",
			"輸出 JSON: {manager_view, recommendation, rationale, risks}. recommendation 需包含建議方向與條件。",
		]
	)
	#print(f"[debug] manager content: {content}")
	return [{"role": "system", "content": "你是研究經理，需整合多空並給出結論。"}, {"role": "user", "content": content}]


def build_trade_messages(manager: str, decision_row: Dict[str, Any], price_info: Dict[str, Any]) -> List[Dict[str, str]]:
	stats = price_info.get("stats", {})
	content = "\n".join(
		[
			"最終交易決策顧問：請給出買入/賣出/觀望訊號、長/中/短期策略、倉位建議、進出場與止損止盈。簡明可執行。",
			f"研究經理結論: {manager}",
			f"決策分數與指標: {_format_decision_context(decision_row)}",
			f"近期價格摘要: last={stats.get('last_close')} ret20={stats.get('return_20d')} vol20={stats.get('volatility_20d')}",
			"輸出 JSON: {trade_plan:decision, last_close, {short, mid, long}, sizing, entries, stops, take_profits, note}.",
		]
	)
	#print(f"[debug] trade content: {content}")
	return [{"role": "system", "content": "你是交易決策顧問，需落地為具體交易計畫。"}, {"role": "user", "content": content}]


def build_summary_messages(code: str, decision_row: Dict[str, Any], stage_outputs: Dict[str, str], executed: List[str]) -> List[Dict[str, Any]]:
	"""Build prompt messages to ask the LLM to produce a structured JSON next-day trade plan.

	The messages follow the user's specification: combine technical, fundamental, news, social,
	bull/bear/manager/trade modules and request a JSON output with decision/confidence/key_reasons/trade_plan/risk_flags.
	"""
	# assemble analysis sections
	parts = []
	parts.append(f"股票代號: {code}")
	parts.append("\n--- 以下為分階段報告 (原始 LLM 回覆) ---")
	for st in executed:
		txt = stage_outputs.get(st, "")
		parts.append(f"[{st}]\n{txt}\n")

	user_prompt = "\n".join([
		"你是一個專業的港股投資分析助手，擅長結合技術面、基本面、新聞情緒、資金面和交易計劃，輸出隔天的交易操作建議。",
		"請根據下列分階段報告，產出結構化 JSON，欄位如下：",
		"{",
		'  "decision": "買入 / 觀望 / 賣出",',
		'  "confidence": 0.0-1.0,',
		'  "key_reasons": ["技術面理由","基本面理由","新聞情緒理由","資金面理由"],',
		'  "trade_plan": {"entry": "建倉條件","add_on": "加倉條件","stop_loss": "止損條件","take_profit": "止盈目標"},',
		'  "risk_flags": ["主要風險1","主要風險2"]',
		"}",
		"要求：",
		"- JSON 嚴格遵守欄位，回傳純 JSON，不要加額外說明文字。",
		"- confidence 使用數值 (0 到 1)，越接近 1 表示越有把握。",
		"- key_reasons 至少包含一條來自技術面與一條來自基本面；若無法判斷請填空字串。",
		"- trade_plan 的 entry/add_on/stop_loss/take_profit 請用簡短明確條件或價位範圍。",
		"以下為分階段報告原文：",
		"\n".join(parts),
	])

	messages = [
		{"role": "system", "content": "你是股票投資策略專家，請根據使用者提供的分階段報告，回傳符合規格的 JSON 結果。"},
		{"role": "user", "content": user_prompt},
	]
	return messages


def call_openrouter(api_key: str, model: str, messages: List[Dict[str, Any]],
					temperature: float = 0.7, max_tokens: int = 3500,
					timeout: int = 60, max_retries: int = 3, backoff_factor: float = 1.0) -> str:
	"""Call OpenRouter with retries and exponential backoff.

	Raises RuntimeError on unrecoverable errors.
	"""
	headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
	payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

	for attempt in range(1, max_retries + 1):
		try:
			resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=timeout)
		except requests.exceptions.RequestException as e:
			if attempt == max_retries:
				raise RuntimeError(f"OpenRouter request failed after {attempt} attempts: {e}")
			sleep = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
			print(f"[llm] request exception: {e}; retrying in {sleep:.1f}s ({attempt}/{max_retries})")
			time.sleep(sleep)
			continue

		# Rate limit handling
		if resp.status_code == 429:
			retry_after = resp.headers.get("Retry-After")
			try:
				wait = float(retry_after) if retry_after is not None else backoff_factor * (2 ** (attempt - 1))
			except Exception:
				wait = backoff_factor * (2 ** (attempt - 1))
			if attempt == max_retries:
				raise RuntimeError(f"OpenRouter rate limited (429). Response: {resp.text}")
			print(f"[llm] 429 rate limit, retrying in {wait}s ({attempt}/{max_retries})")
			time.sleep(wait)
			continue

		# transient server errors -> retry
		if 500 <= resp.status_code < 600:
			if attempt == max_retries:
				raise RuntimeError(f"OpenRouter server error {resp.status_code}: {resp.text}")
			sleep = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
			print(f"[llm] server error {resp.status_code}; retrying in {sleep:.1f}s ({attempt}/{max_retries})")
			time.sleep(sleep)
			continue

		if not resp.ok:
			# client errors (400/401/403) usually not retriable
			raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}")

		# try parse JSON (may fail if response ended prematurely)
		try:
			data = resp.json()
		except Exception as e:
			if attempt == max_retries:
				raise RuntimeError(f"OpenRouter returned invalid JSON after {attempt} attempts: {e}; text={resp.text[:1000]}")
			sleep = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
			print(f"[llm] JSON parse error: {e}; retrying in {sleep:.1f}s ({attempt}/{max_retries})")
			time.sleep(sleep)
			continue

		try:
			usage = data.get("usage", {}) 
			print(f"Prompt tokens: {usage.get('prompt_tokens','N/A')} | "
				f"Completion tokens: {usage.get('completion_tokens','N/A')} | "
				f"Total tokens: {usage.get('total_tokens','N/A')}")
			return data["choices"][0]["message"]["content"]
		except Exception as e:
			if attempt == max_retries:
				raise RuntimeError(f"Unexpected OpenRouter response structure: {data}")
			sleep = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
			print(f"[llm] unexpected response shape: {e}; retrying in {sleep:.1f}s ({attempt}/{max_retries})")
			time.sleep(sleep)
			continue


def resolve_openrouter_key(explicit_key: Optional[str], config_path: Optional[str]) -> Optional[str]:
	if explicit_key:
		k = explicit_key.strip()
		if k.lower().startswith("bearer "):
			return k.split(None, 1)[1]
		return k
	cfg = load_notify_config(config_path)
	if isinstance(cfg, dict):
		v = cfg.get("openrouter_api_key")
		if v:
			v2 = str(v).strip()
			if v2.lower().startswith("bearer "):
				return v2.split(None, 1)[1]
			return v2
	return os.getenv("OPENROUTER_API_KEY")


def format_summary(code: str, decision_row: Dict[str, Any], stage_outputs: Dict[str, str], executed: List[str]) -> str:
	header = f"【{code} LLM 決策】"
	metrics = f"決策分數: {decision_row.get('final_score', 'N/A')} | kronos={decision_row.get('kronos_score', 'N/A')} | net_c={decision_row.get('net_c', 'N/A')} | buy={decision_row.get('buy', 'N/A')}"
	lines = [header, metrics, "--- 分階段回應 ---"]
	for st in executed:
		lines.append(f"[{st}] {stage_outputs.get(st, '').strip()}")
	return "\n".join([str(x) for x in lines if str(x).strip()])


def parse_args():
	parser = argparse.ArgumentParser(description="Ask OpenRouter for stock advice with staged multi-role prompts")
	parser.add_argument("code", nargs="?", help="HK stock code, e.g., 00700.HK or 700; optional when --portfolio is used")
	parser.add_argument("--decision_csv", type=str, help="decision CSV path (decision_YYYYMMDD_recorder.csv)")
	parser.add_argument("--provider_uri", type=str, default="~/.qlib/qlib_data/hk_data", help="qlib provider uri")
	parser.add_argument("--news_dir", type=str, default=str(Path.home() / ".qlib/qlib_data/hk_data/news"))
	parser.add_argument("--openrouter_model", type=str, default="tngtech/deepseek-r1t2-chimera:free")
	parser.add_argument("--openrouter_api_key", type=str, help="override OpenRouter API key")
	parser.add_argument("--notify_config", type=str, help="path to notify_config.json")
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--max_tokens", type=int, default=60000)
	parser.add_argument("--lookback_days", type=int, default=365)
	parser.add_argument("--stages", type=str, default="all", help="comma list from: technical,fundamental,news,social,bull,bear,manager,trade (default all)")
	parser.add_argument(
		"--cache_dir",
		type=str,
		default=str(Path.home() / ".qlib" / "qlib_data" / "hk_data" / "cache_ai_advice"),
		help="directory to store stage cache",
	)
	parser.add_argument("--cache_stages", type=str, default="", help="comma list of stages to read/write cache (default none)")
	# --portfolio_file removed: use --portfolio (comma-separated codes) instead
	parser.add_argument("--portfolio", type=str, help="comma-separated list of instrument codes for portfolio ranking, e.g. 00700.HK,00005.HK")
	parser.add_argument("--telegram_token")
	parser.add_argument("--telegram_chat_id")
	parser.add_argument("--send_telegram", action="store_true", help="send summary via Telegram if configured")
	return parser.parse_args()


def _expand_stages(raw: str) -> List[str]:
	if raw.strip().lower() == "all":
		selected = set(STAGE_ORDER)
	else:
		parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
		selected = set([p for p in parts if p in STAGE_ORDER])
	if "manager" in selected:
		selected.update({"bull", "bear"})
	if "trade" in selected:
		selected.update({"manager", "bull", "bear"})
	return [s for s in STAGE_ORDER if s in selected]


def _expand_cache_stages(raw: str) -> List[str]:
	if not raw:
		return []
	if raw.strip().lower() == "all":
		# Return all defined stages plus portfolio
		return [s for s in STAGE_ORDER] + ["portfolio"]
	parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
	extra = set(["portfolio"])  # allow portfolio as cache-stage
	allowed = set(STAGE_ORDER) | extra
	return [p for p in parts if p in allowed]


def main():
	args = parse_args()
	news_dir = Path(args.news_dir)
	stages = _expand_stages(args.stages)
	cache_dir = Path(args.cache_dir)
	cache_stages = set(_expand_cache_stages(args.cache_stages))
	print(f"[info] stages={stages} cache_dir={cache_dir} cache_stages={sorted(cache_stages)}")
	if args.portfolio:
		# normalize portfolio codes to display form (DDDDD.HK)
		raw_port = [_normalize_display_code(s.strip()) for s in str(args.portfolio).split(",") if s.strip()]
		api_key = resolve_openrouter_key(args.openrouter_api_key, args.notify_config)
		if not api_key:
			raise RuntimeError("Missing OpenRouter API key (set OPENROUTER_API_KEY or notify_config)")
		items, missing = assemble_portfolio_items(raw_port, cache_dir, args.openrouter_model)
		if not items:
			raise RuntimeError(f"No summaries found for portfolio codes; missing={missing}")
		msgs = build_portfolio_messages(items)
		cached = load_stage_cache(cache_dir, "portfolio", "portfolio", args.openrouter_model) if "portfolio" in cache_stages else None
		if cached is not None:
			text = cached
		else:
			print("[llm] calling portfolio ...")
			try:
				text = call_openrouter(api_key, args.openrouter_model, msgs, temperature=args.temperature, max_tokens=args.max_tokens)
			except Exception as e:
				print(f"[llm][error] portfolio call failed: {e}")
				text = ""
		try:
			save_stage_cache(cache_dir, "portfolio", "portfolio", args.openrouter_model, text)
		except Exception:
			print("[cache] warning: failed to save cache for portfolio")
		#print(text)
		tok, chat = resolve_notify_params(args.telegram_token, args.telegram_chat_id, args.notify_config)
		notifier = TelegramNotifier(tok, chat, parse_mode=None)
		if notifier.can_send():
			notifier.send(text)
		else:
			print("error sending telegram.")
		if missing:
			print(f"[portfolio] missing summaries for: {missing}")
		return

	if not args.code:
		raise RuntimeError("Missing code; provide CODE or use --portfolio")
	# normalize requested code to display form (DDDDD.HK) so cache/news filenames use HK
	args.code = _normalize_display_code(args.code)

	decision_path = Path(args.decision_csv) if args.decision_csv else None
	if decision_path is None:
		candidates = sorted(Path.cwd().glob("decision_*.csv"), reverse=True)
		if not candidates:
			raise RuntimeError("No decision_*.csv found; please provide --decision_csv")
		decision_path = candidates[0]
	print(f"[info] decision_csv={decision_path}")

	decision_row = load_decision_row(decision_path, args.code)
	print(f"[info] decision row keys sample={list(decision_row.keys())[:10]}")
	price_info = summarize_prices(args.code, args.provider_uri, lookback_days=args.lookback_days)
	news_blob = load_company_news(news_dir, args.code)
	finance_snip = _pick_finance_snippets(news_blob.get("primary", {}))

	api_key = resolve_openrouter_key(args.openrouter_api_key, args.notify_config)
	if not api_key:
		raise RuntimeError("Missing OpenRouter API key (set OPENROUTER_API_KEY or notify_config)")
	print(f"[info] using model={args.openrouter_model} temperature={args.temperature} max_tokens={args.max_tokens}")

	outputs: Dict[str, str] = {}
	for stage in stages:
		if stage == "technical":
			msgs = build_technical_messages(args.code, decision_row, price_info)
		elif stage == "fundamental":
			msgs = build_fundamental_messages(args.code, decision_row, finance_snip)
		elif stage == "news":
			msgs = build_news_messages(args.code, news_blob)
		elif stage == "social":
			msgs = build_social_messages(args.code, news_blob, finance_snip)
		elif stage == "bull":
			msgs = build_bull_messages(
				outputs.get("technical", ""),
				outputs.get("fundamental", ""),
				outputs.get("news", ""),
				outputs.get("social", ""),
				decision_row,
			)
		elif stage == "bear":
			msgs = build_bear_messages(
				outputs.get("technical", ""),
				outputs.get("fundamental", ""),
				outputs.get("news", ""),
				outputs.get("social", ""),
				decision_row,
			)
		elif stage == "manager":
			msgs = build_manager_messages(outputs.get("bull", ""), outputs.get("bear", ""))
		elif stage == "trade":
			msgs = build_trade_messages(outputs.get("manager", ""), decision_row, price_info)
		else:
			continue
		cached = load_stage_cache(cache_dir, args.code, stage, args.openrouter_model) if stage in cache_stages else None
		if cached is not None:
			text = cached
		else:
			print(f"[llm] calling stage={stage} ...")
			try:
				text = call_openrouter(api_key, args.openrouter_model, msgs, temperature=args.temperature, max_tokens=args.max_tokens)
			except Exception as e:
				print(f"[llm][error] stage={stage} call failed: {e}")
				text = ""
		# Always save stage output to cache directory (user requested saving regardless of load)
		try:
			save_stage_cache(cache_dir, args.code, stage, args.openrouter_model, text)
		except Exception:
			# Don't fail the whole run on cache write errors
			print(f"[cache] warning: failed to save cache for stage={stage}")
		outputs[stage] = text

	# Prepare and save the raw combined textual summary (summary_raw)
	summary_raw = format_summary(args.code, decision_row, outputs, stages)
	try:
		save_stage_cache(cache_dir, args.code, "summary_raw", args.openrouter_model, summary_raw)
	except Exception:
		print("[cache] warning: failed to save summary_raw cache")

	# Build structured summary messages and request a JSON summary from the LLM
	try:
		msgs = build_summary_messages(args.code, decision_row, outputs, stages)
		print("[llm] calling build_summary (structured JSON)")
		structured = ""
		try:
			structured = call_openrouter(api_key, args.openrouter_model, msgs, temperature=args.temperature, max_tokens=args.max_tokens)
		except Exception as e:
			print(f"[llm][error] build_summary call failed: {e}")
		# Save the structured summary as the canonical 'summary' cache
		try:
			save_stage_cache(cache_dir, args.code, "summary", args.openrouter_model, structured)
		except Exception:
			print("[cache] warning: failed to save structured summary cache")
		# Optionally overwrite summary variable for downstream usage
		summary = structured or summary_raw
	except Exception as e:
		print(f"[build_summary] failed: {e}")

	tok, chat = resolve_notify_params(args.telegram_token, args.telegram_chat_id, args.notify_config)
	notifier = TelegramNotifier(tok, chat, parse_mode=None)
	if args.send_telegram and notifier.can_send():
		notifier.send(summary)
	elif args.send_telegram:
		print("[telegram skipped] missing token/chat_id")


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)
