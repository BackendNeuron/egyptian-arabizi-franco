"""The steps around the model, so it only ever sees what it was trained on.

1. split the text into words, spaces and everything else (punctuation, emojis, text already in Arabic)
2. lowercase words and shorten letters stretched three or more times ("ehhh" -> "eh", "awyyyy" -> "awy")
3. pass through unchanged: numbers on their own, and English work/tech words (english_words.txt);
   "el meeting" becomes "الmeeting", as in STYLE.md
4. convert each run of Arabizi words with the model, at most MAX_SPAN words at a time (it was trained on
   short chunks and sentences)
5. put everything back in its original order and spacing"""

import re
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z0-9'’]+|\s+|[^\sA-Za-z0-9'’]+")
STRETCH = re.compile(r"([a-z])\1{2,}")
ARTICLE = {"el", "al", "il", "l"}
MAX_SPAN = 6


def load_english(path: Path) -> set:
    return {w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip() and not w.startswith("#")}


def plan(text: str, english: set) -> list:
    """The text as parts: ("keep", text) is copied as it is, ("convert", [words]) is one run of Arabizi."""
    items = []  # (kind, original, normalised): kind is "space", "ar", or "keep"
    for tok in TOKEN.findall(text):
        if tok.isspace():
            items.append(("space", tok, tok))
            continue
        w = STRETCH.sub(r"\1", tok.lower())
        if not re.match(r"[a-z0-9]", w) or w.isdigit():
            items.append(("keep", tok, tok))
        elif w in english:
            items.append(("en", tok, w))
        else:
            items.append(("ar", tok, w))

    # "el meeting" -> one kept part "الmeeting"
    merged, i = [], 0
    while i < len(items):
        k, orig, w = items[i]
        if (k == "ar" and w in ARTICLE and i + 2 < len(items) and items[i + 1][0] == "space"
                and items[i + 2][0] == "en"):
            merged.append(("keep", "ال" + items[i + 2][2], None))
            i += 3
            continue
        merged.append(("keep", orig, None) if k == "en" else (k, orig, w))
        i += 1

    parts, run = [], []

    def flush():
        for j in range(0, len(run), MAX_SPAN):
            if j:
                parts.append(("keep", " "))
            parts.append(("convert", run[j:j + MAX_SPAN]))
        run.clear()

    for j, (k, orig, w) in enumerate(merged):
        if k == "ar":
            run.append(w)
        elif k == "space" and run and j + 1 < len(merged) and merged[j + 1][0] == "ar":
            continue  # a space between two Arabizi words stays inside the run
        else:
            flush()
            parts.append(("keep", orig))
    flush()
    return parts


def assemble(parts: list, converted: list) -> str:
    """Puts the model's outputs (one per "convert" part, in order) back between the kept parts."""
    out, it = [], iter(converted)
    for kind, val in parts:
        out.append(next(it) if kind == "convert" else val)
    return "".join(out)
