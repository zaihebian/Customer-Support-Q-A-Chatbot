# download_hb_help_section.py
import time, re, os
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

SECTION_URL = "https://help.hollandandbarrett.com/hc/en-gb/sections/360004725680-Ordering-and-Delivery"
OUT_DIR = Path("data_raw_help")  # change if you like
DELAY = 0.6                     # seconds between requests

# Make a polite, retrying session
def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-GB,en;q=0.8",
    })
    retries = Retry(total=5, backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

# Safe filename from article URL
def article_filename(url: str) -> str:
    # Prefer Zendesk pattern: /articles/<id>-<slug>
    m = re.search(r"/articles/(\d+)-([^/?#]+)", url)
    if m:
        art_id, slug = m.group(1), m.group(2)
        return f"{art_id}-{slug}.html"
    # fallback
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", url.split("/")[-1] or "page")
    return clean + ".html"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = make_session()

    # 1) Fetch the section page
    r = s.get(SECTION_URL, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # 2) Collect article links (unique, absolute)
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Keep only article pages under this help centre
        if "/hc/" in href and "/articles/" in href:
            links.add(urljoin(SECTION_URL, href))

    links = sorted(links)  # stable order
    print(f"Found {len(links)} article links")

    # 3) Download each article’s HTML
    for i, url in enumerate(links, 1):
        fname = article_filename(url)
        out_path = OUT_DIR / fname
        if out_path.exists():
            print(f"[{i}/{len(links)}] Exists {fname}")
            continue
        try:
            rr = s.get(url, timeout=20)
            rr.raise_for_status()
            out_path.write_text(rr.text, encoding="utf-8")
            print(f"[{i}/{len(links)}] Saved {fname}")
            time.sleep(DELAY)
        except Exception as e:
            print(f"[{i}/{len(links)}] Skip {fname} — {e}")

if __name__ == "__main__":
    main()
