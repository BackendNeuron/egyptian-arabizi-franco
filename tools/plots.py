"""Charts for the README, in light and dark versions: docs/charts/<name>-light.png and <name>-dark.png.

  python tools/plots.py                 training curves (runs/history.json) and the dataset (data/)
  python tools/plots.py --model runs/best.pt   also: accuracy by topic and the most common mistakes

The README shows the right version for the reader's theme with a <picture> element. Colours are the validated
two-series palette (blue, orange); single-series charts use the blue alone."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/charts"
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#33332f", s1="#3987e5", s2="#d95926"),
}
PCT = FuncFormatter(lambda v, _: f"{v:.0%}")


def figure(t, title, subtitle=None, size=(7.2, 3.6)):
    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=8, length=0, pad=6)
    ax.grid(True, axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.95, title, color=t["ink"], fontsize=11, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.02, 0.885, subtitle, color=t["ink2"], fontsize=8.5, va="top")
    fig.subplots_adjust(left=0.1, right=0.8, top=0.78, bottom=0.16)  # room for end labels on the right
    return fig, ax


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def end_label(ax, t, x, y, text, color_key="ink2"):
    """A direct label at a line's last point, in text ink (identity comes from the line beside it)."""
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                color=t[color_key])


def run_breaks(ax, t, hist):
    """A faint vertical line where a continued run starts, so you can see what new data did."""
    for prev, cur in zip(hist, hist[1:]):
        if cur["run"] != prev["run"]:
            ax.axvline(cur["epoch"] - 0.5, color=t["ink2"], linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.6)
            ax.annotate(cur.get("stage", f"run {cur['run'] + 1}"), (cur["epoch"] - 0.5, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -10), textcoords="offset points", fontsize=7.5, color=t["ink2"])


def training(hist):
    ep = [h["epoch"] for h in hist]
    for name, t in THEMES.items():
        fig, ax = figure(t, "Training loss", "Lower is better · one point per epoch")
        ax.plot(ep, [h["loss"] for h in hist], color=t["s1"], linewidth=2, marker="o", markersize=4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch", color=t["ink2"], fontsize=8)
        run_breaks(ax, t, hist)
        save(fig, "loss", name)

        fig, ax = figure(t, "Accuracy on the held-out test set",
                         "Human-written Arabizi the model never saw in training")
        for key, label, col in (("words", "words", "s1"), ("sentences", "whole sentences", "s2")):
            ys = [h[key] for h in hist]
            ax.plot(ep, ys, color=t[col], linewidth=2, marker="o", markersize=4, label=label)
            end_label(ax, t, ep[-1], ys[-1], f"{label} {ys[-1]:.0%}")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PCT)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch", color=t["ink2"], fontsize=8)
        leg = ax.legend(loc="lower right", frameon=False, fontsize=8)
        for txt in leg.get_texts():
            txt.set_color(t["ink2"])
        run_breaks(ax, t, hist)
        save(fig, "accuracy", name)

        fig, ax = figure(t, "Character error rate", "Share of characters wrong · lower is better")
        ys = [h["cer"] for h in hist]
        ax.plot(ep, ys, color=t["s1"], linewidth=2, marker="o", markersize=4)
        end_label(ax, t, ep[-1], ys[-1], f"{ys[-1]:.1%}")
        ax.set_ylim(0, max(ys) * 1.15)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epoch", color=t["ink2"], fontsize=8)
        run_breaks(ax, t, hist)
        save(fig, "cer", name)


def hbars(t, title, subtitle, labels, values, fmt, size=(7.2, 3.2)):
    fig, ax = figure(t, title, subtitle, size)
    ax.grid(True, axis="x", color=t["grid"], linewidth=0.8)
    ax.grid(False, axis="y")
    y = range(len(labels))[::-1]
    ax.barh(list(y), values, color=t["s1"], height=0.62, edgecolor=t["surface"], linewidth=2)
    ax.set_yticks(list(y), labels)
    for yi, v in zip(y, values):
        ax.annotate(fmt(v), (v, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
                    color=t["ink2"])
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.28, right=0.9)
    return fig, ax


def dataset():
    stats = json.loads((ROOT / "data/stats.json").read_text())
    rows = Path(ROOT / "data/train.tsv").read_text(encoding="utf-8").splitlines()[1:]
    lengths = [len(r.split("\t")[0]) for r in rows]
    for name, t in THEMES.items():
        labels = ["hand-written sentences", "single words", "template sentences", "generated spellings",
                  "test sentences"]
        values = [stats["seed_sentences"], stats.get("single_words", 0), stats["template_sentences"],
                  stats["generated_variants"], stats["test"]]
        fig, ax = hbars(t, "What the data is made of", "Training rows by source · test is kept apart",
                        labels, values, lambda v: f"{v:,}")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        save(fig, "dataset", name)

        fig, ax = figure(t, "Input length", "Characters per Arabizi input in the training set")
        ax.hist(lengths, bins=range(0, max(lengths) + 4, 3), color=t["s1"], edgecolor=t["surface"], linewidth=1)
        ax.set_xlabel("characters", color=t["ink2"], fontsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        save(fig, "lengths", name)


def by_model(path):
    sys.path.insert(0, str(ROOT))
    import torch

    from training.model import Transliterator

    ck = torch.load(path, map_location="cpu")
    model = Transliterator(**ck["config"])
    model.load_state_dict(ck["model"])
    rows = [r.split("\t") for r in (ROOT / "data/test.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    preds = model.transliterate([r[0] for r in rows])
    per, wrong = {}, Counter()
    for (src, tgt, topic), p in zip(rows, preds):
        ok = [pw == tw for pw, tw in zip(p.split(), tgt.split())] or [False]
        per.setdefault(topic, []).extend(ok)
        for pw, tw in zip(p.split(), tgt.split()):
            if pw != tw:
                wrong[f"{tw}  →  {pw}"] += 1
    topics = sorted(per, key=lambda k: -sum(per[k]) / len(per[k]))
    acc = [sum(per[k]) / len(per[k]) for k in topics]
    for name, t in THEMES.items():
        fig, ax = hbars(t, "Word accuracy by topic", "Held-out test set · topics come from the seed files",
                        topics, acc, lambda v: f"{v:.0%}")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PCT)
        save(fig, "topics", name)
    # the most common mistakes, as a table for the README (Arabic text reads better as text than in a chart)
    lines = ["| expected → got | times |", "|---|---|"] + [f"| {k} | {v} |" for k, v in wrong.most_common(15)]
    (OUT / "mistakes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    a = ap.parse_args()
    hist_path = ROOT / "runs/history.json"
    if hist_path.exists():
        training(json.loads(hist_path.read_text()))
    if (ROOT / "data/stats.json").exists():
        dataset()
    if a.model:
        by_model(a.model)
    print("charts in", OUT)


if __name__ == "__main__":
    main()
