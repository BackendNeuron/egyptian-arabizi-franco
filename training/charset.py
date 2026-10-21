"""The fixed character set. Defined once, before the first training run, so later runs can continue from a
saved model: a model can only continue training if its characters never change. Anything outside it maps to
<unk> (emojis, rare symbols) and is copied through unchanged at inference."""

ARABIC = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789٠١٢٣٤٥٦٧٨٩"
PUNCT = " .,!?;:'\"()-_/@#&%+=*~؟،؛…"

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
CHARS = SPECIALS + list(dict.fromkeys(ARABIC + LATIN + DIGITS + PUNCT))
INDEX = {c: i for i, c in enumerate(CHARS)}
PAD, BOS, EOS, UNK = 0, 1, 2, 3


def encode(text: str) -> list[int]:
    return [INDEX.get(c, UNK) for c in text]


def decode(ids: list[int]) -> str:
    return "".join(CHARS[i] for i in ids if i > UNK)
