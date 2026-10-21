"""Egyptian Arabizi (Franco-Arabic) -> Arabic, offline, on a CPU.

    from egyptian_arabizi_franco import convert
    convert("ezayak 3amel eh")              # 'ازيك عامل ايه'
    convert(["yalla bina", "el meeting bokra"])

The first call loads the model (a few MB, bundled with the package); later calls reuse it."""

from pathlib import Path

from .engine import Engine
from .pipeline import assemble, load_english, plan

__version__ = "0.1.0"
__all__ = ["convert", "Converter", "__version__"]

HERE = Path(__file__).parent


class Converter:
    """Holds the loaded model. Use one for the life of your program; `convert()` uses a shared one."""

    def __init__(self, model_dir=None, threads=0, raw=False):
        """model_dir: a folder with encoder.onnx, decoder.onnx and vocab.json (default: the bundled model).
        raw: skip the pipeline and give the model the text as it is (for testing the model alone)."""
        self.engine = Engine(Path(model_dir) if model_dir else HERE / "models", threads=threads)
        self.english = load_english(HERE / "english_words.txt")
        self.raw = raw

    def convert(self, text):
        """A string or a list of strings, Arabizi -> Arabic; returns the same shape it was given."""
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        if self.raw:
            out = self.engine.transliterate(texts)
        else:
            plans = [plan(t, self.english) for t in texts]
            runs = [" ".join(words) for p in plans for kind, words in p if kind == "convert"]
            done = iter(self.engine.transliterate(runs))
            out = [assemble(p, [next(done) for kind, _ in p if kind == "convert"]) for p in plans]
        return out[0] if single else out


_shared = None


def convert(text):
    """Arabizi -> Arabic with the bundled model: a string or a list of strings."""
    global _shared
    if _shared is None:
        _shared = Converter()
    return _shared.convert(text)
