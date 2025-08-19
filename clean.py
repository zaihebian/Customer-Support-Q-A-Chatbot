# clean.py
import pathlib, re, markdownify
from bs4 import BeautifulSoup

RAW = pathlib.Path("data_raw")        # your uploaded HTML
CLEAN = pathlib.Path("data_clean")    # cleaned Markdown here
CLEAN.mkdir(exist_ok=True)

DROP_SELECTORS = [
    "nav", "footer", "header", "script", "style",
    ".cookie", ".newsletter", ".subscribe"
]

def html_to_markdown(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")

    # remove boilerplate blocks
    for sel in DROP_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()

    # kill obvious skip/link-only junk
    for a in soup.find_all("a"):
        if a.get_text(strip=True).lower() in {"skip to main content", "see more"}:
            a.decompose()

    # get a page title if present
    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled"

    # convert to Markdown
    md = markdownify.markdownify(str(soup), heading_style="ATX")

    # add title to top
    md = f"# {title}\n\n{md}"

    # post-process Markdown: simplify further
    md = postprocess_markdown(md)
    return md

def strip_section(md: str, heading_words: tuple[str, ...]) -> str:
    """
    Remove sections whose H1/H2 line matches any of heading_words,
    up to the next heading or end-of-file.
    """
    pattern = r"(?:^|\n)#{1,3}\s*(%s)[^\n]*\n[\s\S]*?(?=\n#{1,3}\s|\Z)" % "|".join(
        re.escape(w) for w in heading_words
    )
    return re.sub(pattern, "\n", md, flags=re.IGNORECASE)

def postprocess_markdown(md: str) -> str:
    # 1) remove image placeholders: ![alt](url)
    md = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md)

    # 2) turn links into plain text: [text](url) -> text
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)

    # 3) promote bold Qs to headings: **Question?** -> ## Question?
    md = re.sub(r"^\s*\*\*(.+?)\*\*\s*$", r"## \1", md, flags=re.MULTILINE)

    # 4) drop “Related articles”, “Articles in this section”, etc.
    md = strip_section(md, ("Related articles", "Articles in this section"))

    # 5) drop lone junk lines
    junk_lines = {
        "skip to main content",
        "see more",
    }
    md = "\n".join(
        line for line in md.splitlines()
        if line.strip().lower() not in junk_lines
    )

    # 6) de-duplicate repeated top title line
    lines = [l for l in md.splitlines()]
    if len(lines) >= 2 and lines[0].startswith("# "):
        first_title = lines[0][2:].strip().lower()
        if lines[1].strip().lower() == first_title:
            lines.pop(1)
    md = "\n".join(lines)

    # 7) collapse extra blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md

# process all HTML files
for html_file in RAW.glob("*.html"):
    try:
        html_text = html_file.read_text(encoding="utf-8", errors="ignore")
        md_text = html_to_markdown(html_text)
        out_path = CLEAN / (html_file.stem + ".md")
        out_path.write_text(md_text, encoding="utf-8")
        print("Saved", out_path.name)
    except Exception as e:
        print("Skip", html_file.name, "-", e)
