# clean_product.py
import re, json, html, pathlib

# ---- Directories ----
RAW_DIR = pathlib.Path("data_raw_hb_sale")   # folder with downloaded HTML pages
OUT_DIR = pathlib.Path("data_clean")         # unified output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)   # ensure folder exists

OUT_PATH = OUT_DIR / "products.md"           # save as markdown

# ---- Helper to extract a balanced JSON array ----
def extract_balanced_array(s: str, start_idx: int) -> str | None:
    depth = 0
    for i in range(start_idx, len(s)):
        ch = s[i]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return s[start_idx:i+1]
    return None

# ---- Process each HTML file ----
all_products = []

for html_file in sorted(RAW_DIR.glob("*.html")):
    html_text = html_file.read_text(encoding="utf-8", errors="ignore")

    for m in re.finditer(r'"tiles"\s*:\s*\[', html_text):
        arr_text = extract_balanced_array(html_text, m.end() - 1)
        if not arr_text:
            continue

        try:
            data = json.loads(arr_text)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict) and all(k in obj for k in ("url","name","skuId")):
                        all_products.append({
                            "name": (obj.get("name") or "").strip(),
                            "price": obj.get("actualPrice") or obj.get("oldPrice") or "",
                        })
        except Exception:
            for m2 in re.finditer(
                r'"url"\s*:\s*"(?P<url>\/shop\/product[^"]+)"[^{}]*?'
                r'"name"\s*:\s*"(?P<name>[^"]+)"[^{}]*?'
                r'"skuId"\s*:\s*"(?P<sku>[^"]+)"'
                r'[^{}]*?(?:"actualPrice"\s*:\s*"(?P<price>[^"]+)"|)',
                arr_text, flags=re.DOTALL):
                g = m2.groupdict()
                all_products.append({
                    "name": html.unescape(g["name"]).strip(),
                    "price": g.get("price") or "",
                })

# ---- Deduplicate while preserving order ----
deduped = []
seen = set()
for p in all_products:
    key = (p["name"], p["price"])
    if key in seen:
        continue
    seen.add(key)
    deduped.append(p)

# ---- Save as Markdown bullet list ----
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("# Products\n\n")
    for p in deduped:
        if p["price"]:
            f.write(f"- {p['name']} — {p['price']}\n")
        else:
            f.write(f"- {p['name']}\n")

print(f"Saved {len(deduped)} products to {OUT_PATH}")
