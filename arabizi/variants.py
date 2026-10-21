"""Realistic Arabizi spelling variants.

`vary(arabizi)` takes one human spelling and returns others a person might type for the same words: number
letters swapped for letters (3 -> a, 7 -> h), doubled or single consonants, vowel swaps, dropped vowels, case and
stretched letters. `from_arabic(text)` writes Arabizi straight from Arabic, for sentences that come without a
human spelling (templates, generated text). English words (Latin letters in the Arabic) pass through unchanged."""

import random
import re

# one Arabic letter -> the ways people write it, most common first (weights fall off down the list)
LETTER = {
    "ا": ["a", "a", "e", ""], "أ": ["a", "2a", "e"], "إ": ["e", "i", "2e"], "آ": ["a", "aa", "2a"],
    "ء": ["2", "'", ""], "ؤ": ["2", "o"], "ئ": ["2", "e", "y"],
    "ب": ["b"], "ت": ["t"], "ث": ["s", "th", "t"], "ج": ["g", "g", "j"], "ح": ["7", "7", "h"],
    "خ": ["5", "kh", "7'"], "د": ["d"], "ذ": ["z", "d", "th"], "ر": ["r"], "ز": ["z"], "س": ["s"],
    "ش": ["sh", "sh", "ch"], "ص": ["s", "s", "9"], "ض": ["d", "d", "D"], "ط": ["t", "t", "6"],
    "ظ": ["z", "z", "th"], "ع": ["3", "3", "a", ""], "غ": ["gh", "gh", "8", "3'"], "ف": ["f"],
    "ق": ["2", "2", "'", "k", "q"], "ك": ["k"], "ل": ["l"], "م": ["m"], "ن": ["n"],
    "ه": ["h"], "ة": ["a", "a", "ah", "e"], "و": ["w", "o", "ou", "u"], "ي": ["y", "i", "e", "ee"],
    "ى": ["a", "y"],
}
VOWELS = "aeiou"


def _pick(options: list[str], rnd: random.Random) -> str:
    # first options are likelier: weights 1, 1/2, 1/3 ...
    return rnd.choices(options, weights=[1 / (i + 1) for i in range(len(options))])[0]


def _arabic_word(word: str, rnd: random.Random) -> str:
    if re.search(r"[A-Za-z]", word):  # an English word kept in Latin (maybe with ال attached)
        return ("el " if word.startswith("ال") else "") + re.sub(r"^ال", "", word)
    out, i = [], 0
    if word.startswith("ال"):
        out.append(_pick(["el ", "el", "al ", "il "], rnd))
        i = 2
    prev_cons = False
    for ch in word[i:]:
        w = _pick(LETTER.get(ch, [ch]), rnd)
        is_vowel_letter = ch in "اوي"
        # Arabic doesn't write short vowels; Arabizi usually does: sometimes put one between two consonants
        if prev_cons and not is_vowel_letter and w and rnd.random() < 0.45:
            out.append(rnd.choice("aeo"))
        out.append(w)
        prev_cons = bool(w) and not is_vowel_letter
    return "".join(out)


def from_arabic(text: str, rnd: random.Random | None = None, words: dict[str, list[str]] | None = None) -> str:
    """`words`: Arabic word -> human spellings seen in the hand-written pairs; used first, letters as a fallback."""
    rnd = rnd or random.Random()
    words = words or {}
    out = []
    for w in text.split():
        if w in words:
            out.append(rnd.choice(words[w]))
        elif re.search(r"[؀-ۿ]", w):
            out.append(_arabic_word(w, rnd))
        else:
            out.append(w)
    return " ".join(out)


def aligned(pairs: list[tuple[str, str]]):
    """(arabic words, arabizi words) for each pair whose two sides line up word for word."""
    for ar, az in pairs:
        a, z = ar.split(), re.sub(r"\b(el|al|il)\s+(?=\S)", r"\1_", az).split()
        if len(a) == len(z):
            yield a, [w.replace("_", " ") for w in z]


def word_table(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Arabic word -> Arabizi spellings, from pairs whose two sides have the same number of words."""
    table: dict[str, list[str]] = {}
    for ar, az in pairs:
        # "el flous" is one Arabic word (الفلوس): join the article to its word before lining the two sides up
        a, z = ar.split(), re.sub(r"\b(el|al|il)\s+(?=\S)", r"\1_", az).split()
        z = [w.replace("_", " ") for w in z]
        if len(a) == len(z):
            for x, y in zip(a, z):
                if re.search(r"[؀-ۿ]", x) and y not in table.setdefault(x, []):
                    table[x].append(y)
    return table


SWAPS = [  # (pattern, replacements) applied to a human spelling, each with some chance
    (r"3", ["a", "", "3"]), (r"7", ["h", "7"]), (r"2", ["'", "a", "", "2"]), (r"5", ["kh", "5"]),
    (r"kh", ["5", "kh"]), (r"sh", ["ch", "sh"]), (r"gh", ["8", "gh"]),
    (r"ee", ["i", "e", "ee"]), (r"oo", ["o", "u", "oo"]), (r"ou", ["o", "u"]), (r"aw", ["ow", "aw"]),
    (r"e(?=[^aeiou ])", ["i", "e", "a"]), (r"o(?=[^aeiou ])", ["u", "o"]),
    (r"^el$", ["el", "il", "al", "l"]), (r"^msh$", ["mesh", "mish", "msh"]),
    (r"^awy$", ["awi", "awii", "awy"]), (r"^eh$", ["eih", "ey", "eh"]), (r"^w$", ["we", "wi", "w"]),
]


def _vary_once(w: str, rnd: random.Random) -> str:
    """One word, respelled the way someone else might type it."""
    for pat, reps in SWAPS:
        if rnd.random() < 0.35:
            w = re.sub(pat, lambda m: rnd.choice(reps), w)
    if rnd.random() < 0.15:  # double a consonant
        w = re.sub(r"([bdfklmnrstz])(?=[aeiou])", lambda m: m.group(1) * (2 if rnd.random() < 0.3 else 1), w)
    if rnd.random() < 0.15:  # drop a vowel inside the word
        w = re.sub(r"(?<=[a-z])[aeiou](?=[a-z])", lambda m: "" if rnd.random() < 0.25 else m.group(0), w)
    return w


def vary(arabizi: str, rnd: random.Random | None = None, n: int = 3, keep: set[str] = frozenset()) -> list[str]:
    """n distinct variants of a human spelling (the original is not included). Words in `keep` (the English
    words, as they appear in the Arabic target) are never changed."""
    rnd = rnd or random.Random()
    seen, out = {arabizi}, []
    for _ in range(n * 6):
        s = " ".join(w if w.lower() in keep else _vary_once(w, rnd) for w in arabizi.split())
        if rnd.random() < 0.1:  # stretch the last vowel: "awyyy"
            s = re.sub(r"([aeiouy])(\W*)$", lambda m: m.group(1) * rnd.randint(2, 4) + m.group(2), s)
        if rnd.random() < 0.15:
            s = s.capitalize() if rnd.random() < 0.7 else s.upper()
        s = re.sub(r" +", " ", s).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) == n:
            break
    return out
