// Runs the exported model with onnxruntime-web (Node and browsers): greedy decoding, one character at a time,
// a batch at once. Same algorithm as packages/python/src/egyptian_arabizi_franco/engine.py.

import * as ort from "onnxruntime-web";

export class Engine {
  static async create(files, { wasmPaths, threads } = {}) {
    if (wasmPaths) ort.env.wasm.wasmPaths = wasmPaths;
    if (threads) ort.env.wasm.numThreads = threads;
    const e = new Engine();
    const v = files.vocab;
    Object.assign(e, { chars: v.chars, maxLen: v.max_len, pad: v.pad, bos: v.bos, eos: v.eos, unk: v.unk });
    e.index = new Map(v.chars.map((c, i) => [c, i]));
    e.enc = await ort.InferenceSession.create(files.encoder);
    e.dec = await ort.InferenceSession.create(files.decoder);
    return e;
  }

  ids(text) {
    const out = [];
    for (const c of text.toLowerCase()) out.push(this.index.get(c) ?? this.unk);
    return out.slice(0, this.maxLen);
  }

  /** Arabizi strings in, Arabic strings out (no pre- or post-processing: see Converter for that). */
  async transliterate(texts) {
    if (!texts.length) return [];
    const ids = texts.map((t) => {
      const s = this.ids(t);
      return s.length ? s : [this.unk];
    });
    const b = texts.length;
    const width = Math.max(...ids.map((s) => s.length));
    const srcData = new BigInt64Array(b * width).fill(BigInt(this.pad));
    ids.forEach((s, r) => s.forEach((x, c) => (srcData[r * width + c] = BigInt(x))));
    const src = new ort.Tensor("int64", srcData, [b, width]);
    const { memory } = await this.enc.run({ src });

    const rows = ids.map(() => [this.bos]);
    const done = new Array(b).fill(false);
    const steps = Math.min(this.maxLen, width + 20);
    for (let step = 0; step < steps; step++) {
      const t = rows[0].length;
      const ysData = new BigInt64Array(b * t);
      rows.forEach((r, i) => r.forEach((x, j) => (ysData[i * t + j] = BigInt(x))));
      const { logits } = await this.dec.run({ ys: new ort.Tensor("int64", ysData, [b, t]), memory, src });
      const v = logits.dims[1];
      for (let i = 0; i < b; i++) {
        let best = 0;
        for (let k = 1; k < v; k++) if (logits.data[i * v + k] > logits.data[i * v + best]) best = k;
        const next = done[i] ? this.pad : best;
        rows[i].push(next);
        if (next === this.eos) done[i] = true;
      }
      if (done.every(Boolean)) break;
    }
    return rows.map((r) => {
      const body = r.slice(1);
      const end = body.indexOf(this.eos);
      return (end >= 0 ? body.slice(0, end) : body)
        .filter((i) => i > this.unk)
        .map((i) => this.chars[i])
        .join("");
    });
  }
}
