"""Builds data/pretrain.tsv: a large synthetic Arabizi -> Arabic set for pre-training, from the Egyptian Arabic
Wikipedia (wikimedia/wikipedia, 20231101.arz on Hugging Face, CC BY-SA 4.0).

The text is downloaded into data/wiki/ (not committed) and never published with the repo: only the model trained
on it is. Steps:
  1. split the articles into sentences; keep short, Arabic-only ones (2-10 words, no digits or Latin letters)
  2. normalise them to STYLE.md (no diacritics or tatweel, ا for أ/إ/آ, في not فى ...)
  3. drop duplicates and cap repeated patterns (many articles are bot-written stubs that share their wording)
  4. cut short chunks of 1-4 words from each sentence: chunks are easier to learn than whole sentences, and
     full sentences are converted chunk by chunk later anyway
  5. write each chunk's Arabizi with training.variants.from_arabic, using the hand-written spellings for words
     the seeds already have, letter rules for the rest
Run: python tools/build_pretrain.py [--files 2] [--max-pairs 300000]"""

import argparse
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from training.variants import from_arabic, word_table  # noqa: E402
from build_dataset import is_test, load_seeds  # noqa: E402

URL = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.arz/train-0000{i}-of-00003.parquet"
WIKI = ROOT / "data/wiki"
DIACRITICS = re.compile(r"[ً-ْٰـ]")  # harakat, superscript alef, tatweel
ARABIC_ONLY = re.compile(r"^[ء-ي ]+$")
# common words Wikipedia spells with ى or ه where STYLE.md wants ي
WORD_FIX = {"فى": "في", "هى": "هي", "اللى": "اللي", "دى": "دي", "انتى": "انتي"}


def download(n_files: int) -> list[Path]:
    WIKI.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_files):
        p = WIKI / f"arz-{i}.parquet"
        if not p.exists():
            print("downloading", URL.format(i=i))
            urllib.request.urlretrieve(URL.format(i=i), p)
        paths.append(p)
    return paths


def normalise(s: str) -> str:
    s = DIACRITICS.sub("", s)
    s = re.sub("[أإآ]", "ا", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(WORD_FIX.get(w, w) for w in s.split())


def sentences(paths: list[Path]):
    import pyarrow.parquet as pq

    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(columns=["text"], batch_size=2000):
            for text in batch.column(0).to_pylist():
                for s in re.split(r"[.!?؟\n:؛,،()\[\]«»\"]+", text):
                    s = normalise(s)
                    if 2 <= len(s.split()) <= 10 and len(s) <= 70 and ARABIC_ONLY.match(s):
                        yield s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=2, help="how many of the 3 parquet files to use")
    ap.add_argument("--max-pairs", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    seeds = load_seeds()
    test_ar = {ar for ar, _, _ in seeds if is_test(ar)}
    words = word_table([(ar, az) for ar, az, _ in seeds if not is_test(ar)])

    seen, pattern = set(), Counter()
    chunks = []
    for s in sentences(download(a.files)):
        if s in seen or s in test_ar:
            continue
        seen.add(s)
        w = s.split()
        key = " ".join(w[1:4])  # stubs repeat everything after their first word ("X قرية في مركز ...")
        pattern[key] += 1
        if pattern[key] > 20:
            continue
        for _ in range(2):
            n = rnd.randint(1, min(4, len(w)))
            i = rnd.randint(0, len(w) - n)
            chunks.append(" ".join(w[i:i + n]))
        if len(chunks) >= a.max_pairs * 1.3:
            break

    pairs = {(from_arabic(c, rnd, words).lower(), c) for c in chunks}
    pairs = [p for p in pairs if p[0].strip()]
    rnd.shuffle(pairs)
    pairs = pairs[: a.max_pairs]
    with open(ROOT / "data/pretrain.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("arabizi\tarabic\n")
        f.writelines(f"{az}\t{ar}\n" for az, ar in pairs)
    print(f"sentences kept {len(seen):,} · chunks {len(chunks):,} · pretrain pairs {len(pairs):,}")
    for az, ar in pairs[:8]:
        print("  ", az, "->", ar)


if __name__ == "__main__":
    main()
