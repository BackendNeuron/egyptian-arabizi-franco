"""Builds data/train.tsv and data/test.tsv (columns: arabizi, arabic).

- Hand-written seeds (data/seed/*.tsv): about 15% go to the test set as they are, human spellings only, never
  varied and never seen in training. The rest go to training with several spelling variants each.
- Template sentences (tools/templates.py) go to training only, with variants.
Inputs are lowercased: case carries nothing in Arabizi, and the model sees text the way it will at inference.
Run: python tools/build_dataset.py [--variants 4] [--seed 0]"""

import argparse
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from arabizi.variants import vary, word_table  # noqa: E402
from templates import generate  # noqa: E402


def load_seeds():
    """(arabic, arabizi, topic): the topic is the seed file's name, 02_daily_life.tsv -> daily life."""
    pairs = []
    for f in sorted(glob.glob(str(ROOT / "data/seed/*.tsv"))):
        topic = Path(f).stem.split("_", 1)[-1].replace("_", " ")
        for line in Path(f).read_text(encoding="utf-8").splitlines()[1:]:
            if "\t" in line:
                ar, az = line.split("\t", 1)
                pairs.append((ar.strip(), az.strip(), topic))
    return pairs


def is_test(arabic: str) -> bool:  # stable split: the same sentence always lands on the same side
    return int(hashlib.md5(arabic.encode()).hexdigest(), 16) % 100 < 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template-ratio", type=float, default=1.0,
                    help="template sentences per hand-written training sentence; templates must not drown the real data")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    seeds = load_seeds()
    test = [(az.lower(), ar, topic) for ar, az, topic in seeds if is_test(ar)]
    test_ar = {t[1] for t in test}
    train_src = [(ar, az) for ar, az, _ in seeds if not is_test(ar)]
    n_seed = len(train_src)
    templates = [(ar, az) for ar, az in generate(rnd) if ar not in test_ar]
    rnd.shuffle(templates)
    train_src += templates[: int(n_seed * a.template_ratio)]
    # single words from the hand-written sentences (ezayak -> ازيك): short pairs that teach the letter mapping
    # directly, so the model converts what it reads instead of recalling whole sentences it has seen
    words = word_table([(ar, az) for ar, az in train_src[:n_seed]])
    word_pairs = sorted({(ar, sp) for ar, spellings in words.items() for sp in spellings})
    train_src += word_pairs
    n_words = len(word_pairs)

    train, human = set(), set()
    for ar, az in train_src:
        keep = {w.lower() for w in re.findall(r"[A-Za-z]+", ar)}  # English words stay as they are
        human.add((az.lower(), ar))
        for s in [az] + vary(az, rnd, a.variants, keep):
            train.add((s.lower(), ar))
    train = sorted(train)
    rnd.shuffle(train)
    with open(ROOT / "data/train.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("arabizi\tarabic\n")
        f.writelines(f"{az}\t{ar}\n" for az, ar in train)
    with open(ROOT / "data/test.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("arabizi\tarabic\ttopic\n")
        f.writelines(f"{az}\t{ar}\t{tp}\n" for az, ar, tp in test)
    # what the training set is made of, for tools/plots.py
    stats = dict(seed_sentences=n_seed, template_sentences=len(train_src) - n_seed - n_words, single_words=n_words,
                 human_spellings=len(human), generated_variants=len(train) - len(human), test=len(test))
    (ROOT / "data/stats.json").write_text(json.dumps(stats, indent=1))
    print(f"seeds {len(seeds)} · train sentences {len(train_src)} · train pairs {len(train)} · test {len(test)}")


if __name__ == "__main__":
    main()
