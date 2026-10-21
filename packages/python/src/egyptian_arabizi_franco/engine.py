"""Runs the exported model with onnxruntime: greedy decoding, one character at a time, a batch at once."""

import json
from pathlib import Path

import numpy as np


class Engine:
    def __init__(self, folder, encoder="encoder.onnx", decoder="decoder.onnx", threads=0):
        import onnxruntime as ort

        folder = Path(folder)
        v = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        self.chars, self.max_len = v["chars"], v["max_len"]
        self.pad, self.bos, self.eos, self.unk = v["pad"], v["bos"], v["eos"], v["unk"]
        self.index = {c: i for i, c in enumerate(self.chars)}
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        cpu = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(folder / encoder), opts, providers=cpu)
        self.dec = ort.InferenceSession(str(folder / decoder), opts, providers=cpu)

    def _ids(self, text):
        return [self.index.get(c, self.unk) for c in text.lower()][: self.max_len]

    def transliterate(self, texts):
        """Arabizi strings in, Arabic strings out (no pre- or post-processing: see Converter for that)."""
        if not texts:
            return []
        ids = [self._ids(t) or [self.unk] for t in texts]
        width = max(len(s) for s in ids)
        src = np.array([s + [self.pad] * (width - len(s)) for s in ids], dtype=np.int64)
        memory = self.enc.run(None, {"src": src})[0]
        ys = np.full((len(texts), 1), self.bos, dtype=np.int64)
        done = np.zeros(len(texts), dtype=bool)
        for _ in range(min(self.max_len, width + 20)):
            logits = self.dec.run(None, {"ys": ys, "memory": memory, "src": src})[0]
            nxt = np.where(done, self.pad, logits.argmax(-1))
            ys = np.concatenate([ys, nxt[:, None]], axis=1)
            done |= nxt == self.eos
            if done.all():
                break
        out = []
        for row in ys[:, 1:].tolist():
            row = row[: row.index(self.eos)] if self.eos in row else row
            out.append("".join(self.chars[i] for i in row if i > self.unk))
        return out
