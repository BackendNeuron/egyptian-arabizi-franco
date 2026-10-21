"""Train the transliterator. Runs on CPU or a GPU (Colab), whichever is there.

  python train.py --new                         start a fresh model
  python train.py --resume                      carry on from runs/last.pt (after a cut-off session)
  python train.py --continue-from runs/v1.pt    train an existing model further on new data, at a lower rate

Checkpoints go to runs/ (last.pt every epoch, best.pt by word accuracy on the test set).
Data: data/train.tsv and data/test.tsv, made by tools/build_dataset.py."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from arabizi.charset import BOS, CHARS, EOS, PAD, encode
from arabizi.model import MAX_LEN, Transliterator, batchify

RUNS = Path("runs")


def read(path):
    rows = Path(path).read_text(encoding="utf-8").splitlines()[1:]
    return [tuple(r.split("\t")[:2]) for r in rows if "\t" in r]


def cer(a: str, b: str) -> float:
    """Character error rate: edit distance over the reference's length."""
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1] / max(len(b), 1)


def evaluate(model, test, show=0):
    preds = []
    for i in range(0, len(test), 64):
        preds += model.transliterate([s for s, _ in test[i:i + 64]])
    pairs = list(zip(preds, test))
    exact = sum(p == t for p, (_, t) in pairs) / len(test)
    words = [pw == tw for p, (_, t) in pairs for pw, tw in zip(p.split(), t.split())]
    c = sum(cer(p, t) for p, (_, t) in pairs) / len(test)
    for p, (s, t) in pairs[:show]:
        print(f"   {s}  ->  {p}   {'ok' if p == t else 'want ' + t}")
    return exact, sum(words) / max(len(words), 1), c


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--new", action="store_true")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--continue-from")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="default 5e-4 for a new model, 1.5e-4 when continuing")
    ap.add_argument("--patience", type=int, default=6, help="stop after this many epochs without improvement")
    ap.add_argument("--train-file", default="data/train.tsv", help="data/pretrain.tsv for the pre-training stage")
    ap.add_argument("--name", default="", help="checkpoint prefix: 'pretrain-' saves runs/pretrain-best.pt")
    ap.add_argument("--stage", default="train", help="label for this run in the history and the charts")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RUNS.mkdir(exist_ok=True)
    train, test = read(a.train_file), read("data/test.tsv")
    last_pt, best_pt = RUNS / f"{a.name}last.pt", RUNS / f"{a.name}best.pt"

    start, best, ck = 0, -1.0, None
    if a.new:
        model, lr = Transliterator(), a.lr or 5e-4
    else:
        ck = torch.load(last_pt if a.resume else a.continue_from, map_location="cpu")
        assert ck["chars"] == CHARS, "the character set changed, so this model cannot continue"
        model = Transliterator(**ck["config"])
        model.load_state_dict(ck["model"])
        lr = a.lr or (ck["lr"] if a.resume else 1.5e-4)
        if a.resume:
            start, best = ck["epoch"] + 1, ck["best"]
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if a.resume:
        opt.load_state_dict(ck["opt"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.05)
    n = sum(p.numel() for p in model.parameters())
    print(f"{dev} · {n / 1e6:.1f}M parameters · {len(train)} train · {len(test)} test · lr {lr}")

    # every epoch's numbers, for tools/plots.py (a fresh model starts a new history; continuing appends)
    hist_path = RUNS / "history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() and not a.new else []
    run = len({h["run"] for h in history})
    stale = 0
    for ep in range(start, a.epochs):
        model.train()
        random.shuffle(train)
        t0, tot = time.time(), 0.0
        for i in range(0, len(train), a.batch):
            chunk = train[i:i + a.batch]
            src = batchify([encode(s)[:MAX_LEN] for s, _ in chunk]).to(dev)
            tgt = batchify([[BOS] + encode(t)[:MAX_LEN] + [EOS] for _, t in chunk]).to(dev)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(chunk)
        exact, word, c = evaluate(model, test)
        print(f"epoch {ep + 1:>3} · loss {tot / len(train):.3f} · test: sentences {exact:.1%} · words {word:.1%}"
              f" · char errors {c:.1%} · {time.time() - t0:.0f}s", flush=True)
        history.append(dict(run=run, stage=a.stage, epoch=len(history) + 1, loss=tot / len(train), sentences=exact, words=word,
                            cer=c, lr=lr, seconds=time.time() - t0, train_pairs=len(train), device=dev))
        hist_path.write_text(json.dumps(history, indent=1))
        state = dict(model=model.state_dict(), opt=opt.state_dict(), config=model.config, chars=CHARS,
                     epoch=ep, best=max(best, word), lr=lr)
        torch.save(state, last_pt)
        if word > best:
            best, stale = word, 0
            torch.save(state, best_pt)
        else:
            stale += 1
            if stale >= a.patience:
                print(f"no improvement for {a.patience} epochs: stopping")
                break
    model.load_state_dict(torch.load(best_pt, map_location=dev)["model"])
    print("best model on the test set:")
    evaluate(model, test, show=12)


if __name__ == "__main__":
    main()
