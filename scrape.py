# scrape.py
import pathlib, re, requests

URLS = [
    "https://help.hollandandbarrett.com/hc/en-gb",

]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0 Safari/537.36"
}

OUT = pathlib.Path("data_raw"); OUT.mkdir(exist_ok=True)

for url in URLS:
    r = requests.get(url, timeout=15, headers=headers)
    r.raise_for_status()
    fname = re.sub(r"[^a-z0-9]+","_", url.lower()) + ".html"
    (OUT / fname).write_text(r.text, encoding="utf-8")
    print("Saved", fname)
