"""A tiny character-level transformer: Arabizi characters in, Arabic characters out (about 3M parameters)."""

import math

import torch
from torch import nn

from .charset import BOS, CHARS, EOS, PAD, decode, encode

MAX_LEN = 160


class Transliterator(nn.Module):
    def __init__(self, d=192, heads=4, layers=3, ff=768, dropout=0.1):
        super().__init__()
        self.config = dict(d=d, heads=heads, layers=layers, ff=ff, dropout=dropout)
        self.emb = nn.Embedding(len(CHARS), d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN + 2, d)
        self.tf = nn.Transformer(d, heads, layers, layers, ff, dropout, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, len(CHARS))
        self.scale = math.sqrt(d)

    def _embed(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device)
        return self.emb(ids) * self.scale + self.pos(pos)

    def forward(self, src, tgt_in):
        mask = self.tf.generate_square_subsequent_mask(tgt_in.size(1), device=src.device)
        h = self.tf(self._embed(src), self._embed(tgt_in), tgt_mask=mask,
                    src_key_padding_mask=src == PAD, tgt_key_padding_mask=tgt_in == PAD,
                    memory_key_padding_mask=src == PAD)
        return self.out(h)

    @torch.no_grad()
    def transliterate(self, texts: list[str]) -> list[str]:
        """Greedy decoding, a batch at a time."""
        self.eval()
        dev = next(self.parameters()).device
        src = batchify([encode(t.lower())[:MAX_LEN] for t in texts]).to(dev)
        mem = self.tf.encoder(self._embed(src), src_key_padding_mask=src == PAD)
        ys = torch.full((len(texts), 1), BOS, dtype=torch.long, device=dev)
        done = torch.zeros(len(texts), dtype=torch.bool, device=dev)
        for _ in range(min(MAX_LEN, src.size(1) + 20)):
            mask = self.tf.generate_square_subsequent_mask(ys.size(1), device=dev)
            h = self.tf.decoder(self._embed(ys), mem, tgt_mask=mask, memory_key_padding_mask=src == PAD)
            nxt = self.out(h[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        out = []
        for row in ys.tolist():
            row = row[1:]
            out.append(decode(row[: row.index(EOS)] if EOS in row else row))
        return out


def batchify(seqs: list[list[int]]) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (n - len(s)) for s in seqs], dtype=torch.long)
