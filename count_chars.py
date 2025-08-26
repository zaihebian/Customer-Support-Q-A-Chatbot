# count_chars.py
import pathlib
import csv
import statistics

DATA = pathlib.Path("data_clean")
out_file = "doc_char_counts.csv"

rows = []
counts = []
for p in DATA.glob("*.md"):
    text = p.read_text(encoding="utf-8")
    char_count = len(text)
    rows.append({"filename": p.name, "char_count": char_count})
    counts.append(char_count)

# save CSV
with open(out_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "char_count"])
    writer.writeheader()
    writer.writerows(rows)

# compute median
median_val = statistics.median(counts) if counts else 0
print(f"✅ Saved character counts to {out_file}")
print(f"📊 Median char count: {median_val}")

