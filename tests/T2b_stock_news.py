import argparse
import json
import time
from datetime import datetime, timedelta
import sys
import random
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Ensure stdout uses UTF-8 on platforms (Windows) where the default encoding
# may be a legacy codec (e.g., GBK). This avoids UnicodeEncodeError when
# printing JSON containing non-ASCII characters.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Default timeouts (connect, read) in seconds; read can be extended via CLI
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 30.0
REQUEST_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
SESSION = None

TRENDING_URL = "https://news.kenneth1203.dpdns.org/api/news/trending"
TRENDING_FILES = {
    "media": "media_today.json",
    "finance": "finance_today.json",
    "social": "social_today.json",
    "tech": "tech_today.json",
}


def _rand_sleep(base_seconds: float):
    """Sleep for a random duration around the provided base seconds.

    Uses a jitter in [0.5x, 1.5x] of base_seconds, clamped at >= 0.
    """
    try:
        base = float(base_seconds)
    except Exception:
        base = 0.0
    dur = max(0.0, random.uniform(0.5, 1.5) * base)
    if dur > 0:
        time.sleep(dur)


def _get_soup(url: str) -> BeautifulSoup:
    global SESSION
    sess = SESSION or requests
    try:
        resp = sess.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ReadTimeout as exc:
        # Retry once on read timeout
        try:
            print(f"_get_soup: read timeout, retrying once for {url}", file=sys.stderr)
            resp = sess.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        except Exception as exc2:
            print(f"_get_soup: retry failed for {url}: {exc2}", file=sys.stderr)
            return None
    except Exception as exc:
        print(f"_get_soup: request failed for {url}: {exc}", file=sys.stderr)
        return None
    # Respect server-declared charset; only fall back to apparent_encoding when missing.
    enc = None
    ct = resp.headers.get("Content-Type", "").lower()
    if "charset" in ct:
        if "utf-8" in ct:
            enc = "utf-8"
        elif "gbk" in ct or "gb2312" in ct:
            enc = "gbk"
    if not enc:
        enc = resp.encoding or resp.apparent_encoding or "utf-8"
    resp.encoding = enc
    try:
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:
        print(f"_get_soup: failed parsing HTML from {url}: {exc}", file=sys.stderr)
        return None


def init_session(retries: int = 3, backoff: float = 0.5):
    """Initialize a global requests Session with retry/backoff for both HTTP and HTTPS."""
    global SESSION
    s = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    SESSION = s


def _parse_table_by_id(soup: BeautifulSoup, tbody_id: str):
    tbody = soup.find("tbody", id=tbody_id)
    if not tbody:
        return []
    rows = []
    for tr in tbody.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
        if cols:
            rows.append(cols)
    return rows


def fetch_financial_indicators(code: str):
    url = f"https://stock.finance.sina.com.cn/hkstock/finance/{code}.html"
    soup = _get_soup(url)
    if not soup:
        print(f"fetch_financial_indicators: failed to fetch page for {code}", file=sys.stderr)
        return {
            "finance_standard": [],
            "balance_sheet": [],
            "cash_flow": [],
            "finance_status": [],
        }
    return {
        "finance_standard": _parse_table_by_id(soup, "tableGetFinanceStandard"),
        "balance_sheet": _parse_table_by_id(soup, "tableGetBalanceSheet"),
        "cash_flow": _parse_table_by_id(soup, "tableGetCashFlow"),
        "finance_status": _parse_table_by_id(soup, "tableGetFinanceStatus"),
    }


def fetch_company_info(code: str):
    url = f"https://stock.finance.sina.com.cn/hkstock/info/{code}.html"
    soup = _get_soup(url)
    if not soup:
        print(f"fetch_company_info: failed to fetch page for {code}", file=sys.stderr)
        return {"profile": {}, "review": "", "outlook": ""}
    profile = {}
    info_table = soup.select_one("#sub01_c1 table")
    if info_table:
        for tr in info_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                key = tds[0].get_text(strip=True)
                value = tds[1].get_text(strip=True)
                profile[key] = value
    review = soup.select_one("#sub01_c2")
    outlook = soup.select_one("#sub01_c3")
    return {
        "profile": profile,
        "review": review.get_text(strip=True) if review else "",
        "outlook": outlook.get_text(strip=True) if outlook else "",
    }


def fetch_company_news(code: str, page: int = 1):
    url = f"https://stock.finance.sina.com.cn/hkstock/go.php/CompanyNews/page/{page}/code/{code}/.phtml"
    soup = _get_soup(url)
    if not soup:
        print(f"fetch_company_news: failed to fetch page {page} for {code}", file=sys.stderr)
        return []
    news_blocks = soup.select("div.part02 ul.list01 li")
    results = []
    for block in news_blocks:
        a_tag = block.find("a")
        time_tag = block.find("span", class_="rt")
        if a_tag and time_tag:
            results.append({
                "title": a_tag.get_text(strip=True),
                "link": a_tag.get("href"),
                "time": time_tag.get_text(strip=True),
            })
    return results


def fetch_news_pages(code: str, max_pages: int = 3, sleep_seconds: float = 1.0):
    all_news = []
    for page in range(1, max_pages + 1):
        try:
            page_news = fetch_company_news(code, page=page)
        except Exception as exc:
            print(f"fetch_news_pages: exception fetching page {page} for {code}: {exc}", file=sys.stderr)
            page_news = []
        if not page_news:
            break
        all_news.extend(page_news)
        try:
            time.sleep(sleep_seconds)
        except Exception:
            pass
    return all_news


def _parse_table_rows(table, skip_header: bool = True):
    rows = []
    if not table:
        return rows
    body = table.find("tbody") or table
    for idx, tr in enumerate(body.find_all("tr")):
        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if not cols:
            continue
        if skip_header and idx == 0:
            # Skip header row when present in tbody
            continue
        rows.append(cols)
    return rows


def fetch_ratings(code: str, pages: int = 3, sleep_seconds: float = 0.5):
    """Fetch investment bank ratings for HK stocks from Sina."""
    results = []
    for page in range(1, pages + 1):
        url = f"http://money.finance.sina.com.cn/hk/view/rating.php?symbol={code}&p={page}"
        soup = _get_soup(url)
        if not soup:
            print(f"fetch_ratings: no soup for page {page} ({url})", file=sys.stderr)
            time.sleep(sleep_seconds)
            continue
        table = soup.select_one("#dataList table")
        page_rows = _parse_table_rows(table, skip_header=False)
        if not page_rows:
            break
        for cols in page_rows:
            # Expected 9 columns: name, code, bank, latest_rating, last_price, target_price, target_change, rating_date, report
            if len(cols) >= 8:
                results.append({
                    "stock_name": cols[0],
                    "stock_code": cols[1],
                    "bank": cols[2],
                    "latest_rating": cols[3],
                    "last_price": cols[4],
                    "target_price": cols[5],
                    "target_change": cols[6],
                    "rating_date": cols[7],
                    "report": cols[8] if len(cols) >= 9 else "",
                })
        try:
            time.sleep(sleep_seconds)
        except Exception:
            pass
    return results


def fetch_dividends(code: str):
    """Fetch HK stock dividends (分红派息) from Sina."""
    url = f"http://stock.finance.sina.com.cn/hkstock/dividends/{code}.html"
    soup = _get_soup(url)
    if not soup:
        print(f"fetch_dividends: failed to fetch page for {code}", file=sys.stderr)
        return []
    div_c1 = soup.find("div", id="sub01_c1")
    table = div_c1.find("table") if div_c1 else None

    dividends = []
    # Expected columns: 公布日期, 年度, 派息事项, 派息内容, 方式, 除净日, 截止过户日期, 派息日期
    for cols in _parse_table_rows(table):
        if len(cols) >= 8:
            dividends.append({
                "announce_date": cols[0],
                "year": cols[1],
                "item": cols[2],
                "content": cols[3],
                "method": cols[4],
                "ex_date": cols[5],
                "book_close_range": cols[6],
                "pay_date": cols[7],
            })

    return dividends


def fetch_rights_changes(code: str):
    """Fetch HK stock rights changes (major holders and buybacks) from Sina.

    Returns a dict with two lists:
    - major_holder_changes: [{date, shareholder_cn, holder_name, involved_shares, prev_balance, prev_pct, prev_category, curr_balance, curr_pct, curr_category, share_type}]
    - share_buyback: [{date, shares, max_price, min_price, total_price, avg_price}]
    """
    url = f"http://stock.finance.sina.com.cn/hkstock/rights/{code}.html"
    soup = _get_soup(url)
    if not soup:
        print(f"fetch_rights_changes: failed to fetch page for {code}", file=sys.stderr)
        return {"major_holder_changes": [], "share_buyback": []}

    # 主要持有人权益变动
    major_list = []
    div_c2 = soup.find("div", id="sub01_c2")
    table_c2 = div_c2.find("table") if div_c2 else None
    for cols in _parse_table_rows(table_c2):
        # Expected 11 columns per provided layout
        # 事件日期, 中文股东名称, 持股人名字, 涉及股份(股), 先前结余(股), 先前结余率(%), 先前类别, 目前结余(股), 目前结余率(%), 目前类别, 股份类别
        if len(cols) >= 11:
            major_list.append({
                "date": cols[0],
                "shareholder_cn": cols[1],
                "holder_name": cols[2],
                "involved_shares": cols[3],
                "prev_balance": cols[4],
                "prev_pct": cols[5],
                "prev_category": cols[6],
                "curr_balance": cols[7],
                "curr_pct": cols[8],
                "curr_category": cols[9],
                "share_type": cols[10],
            })

    # 公司股份回购
    buyback_list = []
    div_c3 = soup.find("div", id="sub01_c3")
    table_c3 = div_c3.find("table") if div_c3 else None
    for cols in _parse_table_rows(table_c3):
        # Expected 6 columns: 回购日期, 回购数量(股), 最高回购价(港元), 最低回购价(港元), 回购总价(港元), 平均回购价(港元)
        if len(cols) >= 6:
            buyback_list.append({
                "date": cols[0],
                "shares": cols[1],
                "max_price": cols[2],
                "min_price": cols[3],
                "total_price": cols[4],
                "avg_price": cols[5],
            })

    return {
        "major_holder_changes": major_list,
        "share_buyback": buyback_list,
    }


def load_cache(cache_path: Path):
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _is_empty_section(key: str, value) -> bool:
    if value is None:
        return True
    if key in {"ratings", "news", "dividends"}:
        return isinstance(value, list) and len(value) == 0
    if key == "company" and isinstance(value, dict):
        return not value.get("profile") and not value.get("review") and not value.get("outlook")
    if key == "finance" and isinstance(value, dict):
        parts = (
            value.get("finance_standard"),
            value.get("balance_sheet"),
            value.get("cash_flow"),
            value.get("finance_status"),
        )
        return all(isinstance(p, list) and len(p) == 0 for p in parts)
    if key == "rights" and isinstance(value, dict):
        mh = value.get("major_holder_changes")
        sb = value.get("share_buyback")
        return (isinstance(mh, list) and len(mh) == 0) and (isinstance(sb, list) and len(sb) == 0)
    if isinstance(value, dict):
        return len(value) == 0
    if isinstance(value, str):
        return value == ""
    return False


def save_cache(cache_path: Path, data):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cache(cache_path) or {}

    merged = dict(existing) if isinstance(existing, dict) else {}
    # Always update code if provided
    if isinstance(data, dict):
        if "code" in data:
            merged["code"] = data["code"]

        # Define keys we protect from empty overwrites
        protect_keys = {"company", "finance", "dividends", "rights", "ratings", "news"}

        for k, v in data.items():
            if k in protect_keys and k in merged and _is_empty_section(k, v):
                # Skip overwriting non-empty existing values with empty new values
                continue
            merged[k] = v
    else:
        merged = data

    # Preserve a stable key order when writing
    def _ordered(d: dict) -> dict:
        preferred = [
            "code",
            "company",
            "finance",
            "dividends",
            "rights",
            "ratings",
            "news",
            "trending",
            "meta",
        ]
        out = {}
        for k in preferred:
            if isinstance(d, dict) and k in d:
                out[k] = d[k]
        if isinstance(d, dict):
            for k in d.keys():
                if k not in out:
                    out[k] = d[k]
        return out

    ordered = _ordered(merged if isinstance(merged, dict) else {})
    cache_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_all(code: str, force_update: bool = False, max_pages: int = 3, rating_pages: int = 3, sleep_seconds: float = 1.0):
    # normalize incoming code: accept '00700', '700', or '00700.HK' and produce
    # two forms: base (digits only, zero-padded 5) for URL fetches, and display
    # (5-digit.HK) for cache filenames and user-facing outputs.
    base_digits = ''.join([c for c in str(code) if c.isdigit()])
    base = base_digits.zfill(5)
    display = f"{base}.HK"
    cache_path = Path.home() / ".qlib" / "qlib_data" / "hk_data" / "news" / f"{display}.json"
    if not force_update:
        cached = load_cache(cache_path)
        if cached is not None:
            try:
                meta = cached.get("meta", {}) if isinstance(cached, dict) else {}
                fetched_at = meta.get("fetched_at")
                if fetched_at:
                    # expected format: "YYYY-MM-DD HH:MM:SS"
                    fmt = "%Y-%m-%d %H:%M:%S"
                    fetched_dt = datetime.strptime(fetched_at, fmt)
                    if (datetime.now() - fetched_dt) < timedelta(hours=4):
                        print(f"Using cached data for {display} from {fetched_at}")
                        return cached
            except Exception:
                # on any parsing error, treat cache as stale and continue to fetch
                pass
    # use base (digits only) for external fetch URLs
    print(f"Fetching company info for {display}...")
    company = fetch_company_info(base)
    _rand_sleep(sleep_seconds)
    print
    finance = fetch_financial_indicators(base)
    _rand_sleep(sleep_seconds)
    print(f"Fetching dividends for {display}...")
    dividends = fetch_dividends(base)
    _rand_sleep(sleep_seconds)
    print(f"Fetching rights changes for {display}...")
    rights = fetch_rights_changes(base)
    _rand_sleep(sleep_seconds)
    print(f"Fetching ratings for {display}...")
    ratings = fetch_ratings(base, pages=rating_pages, sleep_seconds=sleep_seconds)
    _rand_sleep(sleep_seconds)
    print(f"Fetching news for {display}...")
    news = fetch_news_pages(base, max_pages=max_pages, sleep_seconds=sleep_seconds)
    # Preserve insertion order: company -> finance -> dividends -> rights -> ratings -> news
    result = {
        "code": display,
        "company": company,
        "finance": finance,
        "dividends": dividends,
        "rights": rights,
        "ratings": ratings,
        "news": news,
        "meta": {"fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    save_cache(cache_path, result)
    return result


def fetch_trending(save_dir: Path = None):
    global SESSION
    sess = SESSION or requests
    try:
        resp = sess.get(TRENDING_URL, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"fetch_trending: failed to fetch trending URL: {exc}", file=sys.stderr)
        payload = {}
    base_dir = save_dir or Path.home() / ".qlib" / "qlib_data" / "hk_data" / "news"
    base_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in TRENDING_FILES.items():
        data = payload.get(key, [])
        target = base_dir / filename
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: payload.get(k, []) for k in TRENDING_FILES.keys()}


def main():
    parser = argparse.ArgumentParser(description="Fetch HK stock info/news/finance as JSON")
    parser.add_argument("code", nargs="?", help="HK stock code, e.g., 01810 (optional when --fetch_trending is used)")
    parser.add_argument("--force_update", action="store_true", help="Fetch fresh data and overwrite cache")
    parser.add_argument("--max_pages", type=int, default=3, help="Max news pages to crawl")
    parser.add_argument("--rating_pages", type=int, default=3, help="Max rating pages to crawl")
    parser.add_argument("--sleep_seconds", type=float, default=3.0, help="Sleep between news pages")
    parser.add_argument("--timeout_connect", type=float, default=DEFAULT_CONNECT_TIMEOUT, help="Connect timeout in seconds")
    parser.add_argument("--timeout_read", type=float, default=DEFAULT_READ_TIMEOUT, help="Read timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Number of HTTP retries on failure (connect/read/status)")
    parser.add_argument("--backoff", type=float, default=0.5, help="Exponential backoff factor between retries")
    parser.add_argument("--fetch_trending", action="store_true", help="Also fetch trending news categories and save to cache")
    parser.add_argument(
        "--test_fetch",
        choices=["company", "finance", "dividends", "rights", "ratings", "news"],
        help="Run only a single fetch (no cache) and print the JSON result",
    )
    args = parser.parse_args()

    # Apply CLI timeouts globally
    global REQUEST_TIMEOUT
    REQUEST_TIMEOUT = (args.timeout_connect, args.timeout_read)
    init_session(retries=args.retries, backoff=args.backoff)

    # Single-fetch test mode: run one fetch and print JSON, then exit.
    if args.test_fetch:
        if not args.code:
            parser.error("`code` is required for --test_fetch")
        base_digits = ''.join([c for c in str(args.code) if c.isdigit()])
        base = base_digits.zfill(5)
        if args.test_fetch == "company":
            res = fetch_company_info(base)
        elif args.test_fetch == "finance":
            res = fetch_financial_indicators(base)
        elif args.test_fetch == "dividends":
            res = fetch_dividends(base)
        elif args.test_fetch == "rights":
            res = fetch_rights_changes(base)
        elif args.test_fetch == "ratings":
            res = fetch_ratings(base, pages=args.rating_pages, sleep_seconds=args.sleep_seconds)
        elif args.test_fetch == "news":
            res = fetch_news_pages(base, max_pages=args.max_pages, sleep_seconds=args.sleep_seconds)
        else:
            res = None
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # If user only wants trending data, allow running without a `code`
    if args.fetch_trending and not args.code:
        trending = fetch_trending()
        cats = ", ".join(sorted(trending.keys())) if isinstance(trending, dict) else ""
        print(f"fetch_trending success. categories: {cats}")
        return

    # For code-based fetches, require a code
    if not args.code:
        parser.error("`code` is required unless --fetch_trending is specified")

    data = fetch_all(
        args.code,
        force_update=args.force_update,
        max_pages=args.max_pages,
        rating_pages=args.rating_pages,
        sleep_seconds=args.sleep_seconds,
    )
    
    news_len = len(data.get("news", [])) if isinstance(data, dict) else 0
    msg = f"fetch success: code={args.code}, news_count={news_len}"
    if args.fetch_trending:
        data["trending"] = fetch_trending()
        msg += ", trending fetched"
    print(msg)

if __name__ == "__main__":
    main()
