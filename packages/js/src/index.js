// Egyptian Arabizi (Franco-Arabic) -> Arabic, offline: in Node and in the browser.
//
//   import { convert } from "egyptian-arabizi-franco";
//   await convert("ezayak 3amel eh");               // "ازيك عامل ايه"
//   await convert(["yalla bina", "el meeting bokra"]);
//
// The first call loads the model (a few MB, shipped with the package); later calls reuse it.

import { Engine } from "./engine.js";
import { assemble, parseEnglish, plan } from "./pipeline.js";

export const VERSION = "0.1.0";

const asset = (name) => new URL(`../${name}`, import.meta.url);

// Node reads the files from disk; a browser (or a bundler that copies the files) fetches them.
async function load(url, kind) {
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    const buf = await readFile(url);
    return kind === "bytes" ? new Uint8Array(buf) : buf.toString("utf8");
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`egyptian-arabizi-franco: could not load ${url} (${res.status})`);
  return kind === "bytes" ? new Uint8Array(await res.arrayBuffer()) : res.text();
}

export class Converter {
  /**
   * options.modelUrl: a folder URL holding encoder.onnx, decoder.onnx and vocab.json (default: the bundled model)
   * options.wasmPaths: where onnxruntime-web finds its .wasm files, if your bundler does not serve them
   * options.raw: skip the pipeline and give the model the text as it is (for testing the model alone)
   */
  static async create(options = {}) {
    const base = options.modelUrl ? new URL(options.modelUrl.replace(/\/?$/, "/"), asset("")) : asset("models/");
    const [encoder, decoder, vocab, english] = await Promise.all([
      load(new URL("encoder.onnx", base), "bytes"),
      load(new URL("decoder.onnx", base), "bytes"),
      load(new URL("vocab.json", base), "text"),
      load(asset("english_words.txt"), "text"),
    ]);
    const c = new Converter();
    c.engine = await Engine.create({ encoder, decoder, vocab: JSON.parse(vocab) }, options);
    c.english = parseEnglish(english);
    c.raw = Boolean(options.raw);
    return c;
  }

  /** A string or an array of strings, Arabizi -> Arabic; resolves to the same shape it was given. */
  async convert(text) {
    const single = typeof text === "string";
    const texts = single ? [text] : [...text];
    let out;
    if (this.raw) out = await this.engine.transliterate(texts);
    else {
      const plans = texts.map((t) => plan(t, this.english));
      const runs = plans.flatMap((p) => p.filter(([k]) => k === "convert").map(([, words]) => words.join(" ")));
      const done = await this.engine.transliterate(runs);
      let i = 0;
      out = plans.map((p) => assemble(p, p.filter(([k]) => k === "convert").map(() => done[i++])));
    }
    return single ? out[0] : out;
  }
}

let shared;

/** Arabizi -> Arabic with the bundled model: a string or an array of strings. */
export async function convert(text) {
  shared ??= Converter.create();
  return (await shared).convert(text);
}
