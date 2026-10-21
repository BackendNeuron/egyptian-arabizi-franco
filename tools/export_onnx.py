"""Exports a trained checkpoint to ONNX for the pip and npm packages, and checks the export.

  python tools/export_onnx.py runs/best.pt --out export/

Writes three files:
  encoder.onnx  Arabizi character ids -> encoded input (run once per text)
  decoder.onnx  the Arabic written so far + the encoded input -> scores for the next character (run in a loop)
  vocab.json    the character list and special ids, so the packages map characters exactly as training did
Both models are quantised to int8 (about a quarter of the size). Then the export is checked: the test set is
converted by the original PyTorch model and by the ONNX files, and the script reports how often they agree and
the word accuracy of each. Needs: torch, onnx, onnxruntime, numpy."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.charset import BOS, CHARS, EOS, PAD, UNK  # noqa: E402
from training.model import MAX_LEN, Transliterator  # noqa: E402

sys.path.insert(0, str(ROOT / "packages/python/src"))
from egyptian_arabizi_franco.engine import Engine  # noqa: E402  (the same runner the pip package uses)


# PyTorch's nn.Transformer does shape checks and fused fast paths that neither ONNX exporter can follow, so the
# export runs this plain re-implementation of the same layers (pre-norm, ReLU) on the trained weights.
# check() below proves it gives the same output as the original model on the whole test set.
NEG = -1e9


def attention(q_in, kv_in, mha: nn.MultiheadAttention, key_pad=None, causal=None):
    e, h = mha.embed_dim, mha.num_heads
    w, b = mha.in_proj_weight, mha.in_proj_bias
    q = (q_in @ w[:e].T + b[:e]).reshape(q_in.shape[0], -1, h, e // h).transpose(1, 2)
    k = (kv_in @ w[e:2 * e].T + b[e:2 * e]).reshape(kv_in.shape[0], -1, h, e // h).transpose(1, 2)
    v = (kv_in @ w[2 * e:].T + b[2 * e:]).reshape(kv_in.shape[0], -1, h, e // h).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / (e // h) ** 0.5
    if key_pad is not None:
        scores = scores.masked_fill(key_pad[:, None, None, :], NEG)
    if causal is not None:
        scores = scores.masked_fill(causal[None, None], NEG)
    out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(q_in.shape[0], -1, e)
    return mha.out_proj(out)


def embed(m, ids):
    pos = torch.ones_like(ids[0]).cumsum(0) - 1  # 0..len-1, from ids so the length stays dynamic
    return m.emb(ids) * m.scale + m.pos(pos)


def feed_forward(x, layer):
    return layer.linear2(torch.relu(layer.linear1(x)))


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, src):
        pad = src == PAD
        x = embed(self.m, src)
        for layer in self.m.tf.encoder.layers:
            x = x + attention(layer.norm1(x), layer.norm1(x), layer.self_attn, key_pad=pad)
            x = x + feed_forward(layer.norm2(x), layer)
        return self.m.tf.encoder.norm(x)


class Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ys, memory, src):
        pad = src == PAD
        n = torch.ones_like(ys[0], dtype=torch.long).cumsum(0)  # 1..t, built from ys so its length stays dynamic
        causal = n[None, :] > n[:, None]
        x = embed(self.m, ys)
        for layer in self.m.tf.decoder.layers:
            y = layer.norm1(x)
            x = x + attention(y, y, layer.self_attn, causal=causal)
            x = x + attention(layer.norm2(x), memory, layer.multihead_attn, key_pad=pad)
            x = x + feed_forward(layer.norm3(x), layer)
        return self.m.out(self.m.tf.decoder.norm(x)[:, -1])


def export(ck_path: Path, out: Path):
    torch.backends.mha.set_fastpath_enabled(False)  # the fused fast path does not export
    ck = torch.load(ck_path, map_location="cpu")
    assert ck.get("chars", CHARS) == CHARS, "this checkpoint was trained with a different character set"
    model = Transliterator(**ck["config"])
    model.load_state_dict(ck["model"])
    model.eval()
    out.mkdir(parents=True, exist_ok=True)

    # example inputs with different batch and length sizes, so nothing gets fixed to one shape
    src = torch.randint(4, 40, (2, 9), dtype=torch.long)
    with torch.no_grad():
        mem = Encoder(model)(src)
        ys = torch.randint(4, 40, (2, 5), dtype=torch.long)
        axes = {"src": {0: "b", 1: "s"}, "memory": {0: "b", 1: "s"}, "ys": {0: "b", 1: "t"}, "logits": {0: "b"}}
        common = dict(opset_version=17, dynamo=False, do_constant_folding=True)
        torch.onnx.export(Encoder(model), (src,), out / "encoder.fp32.onnx", input_names=["src"],
                          output_names=["memory"], dynamic_axes={k: axes[k] for k in ("src", "memory")}, **common)
        torch.onnx.export(Decoder(model), (ys, mem, src), out / "decoder.fp32.onnx",
                          input_names=["ys", "memory", "src"], output_names=["logits"],
                          dynamic_axes={k: axes[k] for k in ("ys", "memory", "src", "logits")}, **common)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    for name in ("encoder", "decoder"):
        quantize_dynamic(out / f"{name}.fp32.onnx", out / f"{name}.onnx", weight_type=QuantType.QInt8)
    vocab = dict(chars=CHARS, pad=PAD, bos=BOS, eos=EOS, unk=UNK, max_len=MAX_LEN,
                 trained_epochs=ck.get("epoch", -1) + 1)
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return model


def word_acc(got, wants):
    words = [pw == tw for p, t in zip(got, wants) for pw, tw in zip(p.split(), t.split())]
    return sum(words) / len(words)


def check(model, out: Path) -> dict:
    """Word accuracy on the test set of the PyTorch model and of each ONNX version."""
    rows = [r.split("	") for r in (ROOT / "data/test.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    srcs, wants = [r[0] for r in rows], [r[1] for r in rows]
    ref = model.transliterate(srcs)
    acc = {"pytorch": word_acc(ref, wants)}
    print(f"PyTorch: word accuracy {acc['pytorch']:.1%}")
    for label, enc, dec in (("fp32", "encoder.fp32.onnx", "decoder.fp32.onnx"), ("int8", "encoder.onnx", "decoder.onnx")):
        got = Engine(out, encoder=enc, decoder=dec).transliterate(srcs)
        acc[label] = word_acc(got, wants)
        agree = sum(a == b for a, b in zip(ref, got)) / len(ref)
        print(f"{label}: same output as PyTorch {agree:.1%} · word accuracy {acc[label]:.1%}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="export")
    a = ap.parse_args()
    out = Path(a.out)
    model = export(Path(a.checkpoint), out)
    acc = check(model, out)
    # ship the small int8 files unless they cost accuracy; then ship the full-precision ones under the same names
    if acc["int8"] + 1e-9 < acc["fp32"]:
        print("int8 loses accuracy: shipping the full-precision model")
        for name in ("encoder", "decoder"):
            (out / f"{name}.fp32.onnx").replace(out / f"{name}.onnx")
    else:
        print("int8 keeps the accuracy: shipping the int8 model")
        for f in out.glob("*.fp32.onnx"):
            f.unlink()
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<16} {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
