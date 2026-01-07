import argparse
import json
import time
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

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

TRENDING_URL = "https://news.kenneth1203.dpdns.org/api/news/trending"
TRENDING_FILES = {
    "media": "media_today.json",
    "finance": "finance_today.json",
    "social": "social_today.json",
    "tech": "tech_today.json",
}


def _get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
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
    return BeautifulSoup(resp.text, "html.parser")


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
    return {
        "finance_standard": _parse_table_by_id(soup, "tableGetFinanceStandard"),
        "balance_sheet": _parse_table_by_id(soup, "tableGetBalanceSheet"),
        "cash_flow": _parse_table_by_id(soup, "tableGetCashFlow"),
        "finance_status": _parse_table_by_id(soup, "tableGetFinanceStatus"),
    }


def fetch_company_info(code: str):
    url = f"https://stock.finance.sina.com.cn/hkstock/info/{code}.html"
    soup = _get_soup(url)
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


def fetch_news_pages(code: str, max_pages: int = 3, sleep_seconds: float = 0.5):
    all_news = []
    for page in range(1, max_pages + 1):
        page_news = fetch_company_news(code, page=page)
        if not page_news:
            break
        all_news.extend(page_news)
        time.sleep(sleep_seconds)
    return all_news


def load_cache(cache_path: Path):
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_cache(cache_path: Path, data):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_all(code: str, force_update: bool = False, max_pages: int = 3, sleep_seconds: float = 0.5):
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
            return cached
    # use base (digits only) for external fetch URLs
    company = fetch_company_info(base)
    finance = fetch_financial_indicators(base)
    news = fetch_news_pages(base, max_pages=max_pages, sleep_seconds=sleep_seconds)
    result = {
        "code": display,
        "company": company,
        "finance": finance,
        "news": news,
        "meta": {"fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    save_cache(cache_path, result)
    return result


def fetch_trending(save_dir: Path = None):
    resp = requests.get(TRENDING_URL, headers=DEFAULT_HEADERS, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
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
    parser.add_argument("--sleep_seconds", type=float, default=0.5, help="Sleep between news pages")
    parser.add_argument("--fetch_trending", action="store_true", help="Also fetch trending news categories and save to cache")
    args = parser.parse_args()

    # If user only wants trending data, allow running without a `code`
    if args.fetch_trending and not args.code:
        trending = fetch_trending()
        cats = ", ".join(sorted(trending.keys())) if isinstance(trending, dict) else ""
        print(f"fetch_trending success. categories: {cats}")
        return

    # For code-based fetches, require a code
    if not args.code:
        parser.error("`code` is required unless --fetch_trending is specified")

    data = fetch_all(args.code, force_update=args.force_update, max_pages=args.max_pages, sleep_seconds=args.sleep_seconds)
    news_len = len(data.get("news", [])) if isinstance(data, dict) else 0
    msg = f"fetch success: code={args.code}, news_count={news_len}"
    if args.fetch_trending:
        data["trending"] = fetch_trending()
        msg += ", trending fetched"
    print(msg)

if __name__ == "__main__":
    main()
