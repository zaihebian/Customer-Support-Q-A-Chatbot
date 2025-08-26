# download_product.py
import re
import time
from pathlib import Path
import argparse
import requests
from requests.adapters import HTTPAdapter, Retry

BASE = "https://www.hollandandbarrett.ie/shop/offers/end-of-summer-sale/"
PAGE_URL = BASE + "?page={page}"  # no #fragment

PRODUCT_RE = re.compile(r'/shop/product/[^"\']+')
PAGE_LINK_RE = re.compile(r'\?page=(\d+)\b')

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/125 Safari/537.36",
        "Accept-Language": "en-IE,en;q=0.9",
    })
    retries = Retry(total=5, backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def count_products(html: str) -> int:
    return len(PRODUCT_RE.findall(html or ""))

def discover_last_page(html: str) -> int:
    # Look for ?page=N links and return the max; default to 1 if none.
    nums = [int(n) for n in PAGE_LINK_RE.findall(html or "")]
    return max(nums) if nums else 1

def fetch(session, url, timeout=20):
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def download_all(outdir: Path, delay: float = 0.6, hard_max: int = 200):
    outdir.mkdir(parents=True, exist_ok=True)
    s = make_session()

    # Page 1 (base URL)
    html1 = fetch(s, BASE)
    (outdir / "page-1.html").write_text(html1, encoding="utf-8")
    print("Saved", outdir / "page-1.html")

    # Determine how many pages to fetch
    last = discover_last_page(html1)
    if last < 1:
        last = 1
    last = min(last, hard_max)

    # If pagination not discoverable, fall back to keep-going mode
    fallback_mode = (last == 1)

    if not fallback_mode:
        # Download 2..last
        for p in range(2, last + 1):
            url = PAGE_URL.format(page=p)
            path = outdir / f"page-{p}.html"
            if path.exists():
                print("Exists", path)
            else:
                html = fetch(s, url)
                path.write_text(html, encoding="utf-8")
                print("Saved", path)
            time.sleep(delay)
    else:
        # Keep fetching increasing pages until we hit consecutive empty pages
        empty_streak = 0
        p = 2
        while p <= hard_max and empty_streak < 2:
            url = PAGE_URL.format(page=p)
            path = outdir / f"page-{p}.html"
            html = fetch(s, url)
            path.write_text(html, encoding="utf-8")
            print("Saved", path)
            products = count_products(html)
            if products == 0:
                empty_streak += 1
            else:
                empty_streak = 0
            p += 1
            time.sleep(delay)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_raw_hb_sale", help="Output folder")
    ap.add_argument("--delay", type=float, default=0.6, help="Delay between requests (s)")
    ap.add_argument("--hard-max", type=int, default=200, help="Safety cap on pages")
    args = ap.parse_args()

    download_all(Path(args.out), delay=args.delay, hard_max=args.hard_max)
