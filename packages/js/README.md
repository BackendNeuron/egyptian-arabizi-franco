# egyptian-arabizi-franco

Egyptian **Arabizi** (Franco-Arabic, "Franco") → **Arabic**, offline, in **Node and the browser**:

```
ezayak 3amel eh        →  ازيك عامل ايه
3andy el meeting bokra →  عندي الmeeting بكرة
```

A small character-level transformer (about 3M parameters, shipped with the package, runs locally with
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web)) plus a few rules around it: English work and tech
words stay in Latin letters, numbers, punctuation and emojis pass through, and stretched letters (`awyyyy`) are
shortened before conversion. No server, no API key.

## Install

```bash
npm install egyptian-arabizi-franco
```

## Use

```js
import { convert } from "egyptian-arabizi-franco";

await convert("ezayak 3amel eh");                     // "ازيك عامل ايه"
await convert(["yalla bina", "el meeting bokra"]);    // an array in, an array out (faster than one by one)
```

The first call loads the model; later calls reuse it. To control loading, keep your own converter:

```js
import { Converter } from "egyptian-arabizi-franco";

const c = await Converter.create({
  // wasmPaths: "/ort/",   // where onnxruntime-web's .wasm files are served, if your bundler doesn't copy them
  // modelUrl: "/models/", // a folder with encoder.onnx, decoder.onnx and vocab.json, to host the model yourself
  // raw: true,            // the model alone, without the rules around it
});
await c.convert("msh fahem 7aga");
```

### In a plain HTML page (no build step)

```html
<script type="importmap">
  { "imports": { "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.min.mjs" } }
</script>
<script type="module">
  import { convert } from "https://cdn.jsdelivr.net/npm/egyptian-arabizi-franco@0.1.0/src/index.js";
  document.body.textContent = await convert("ezayak 3amel eh");
</script>
```

The package's own files are loaded straight from the CDN (not a bundled `+esm` build), so it can find its model
next to them.

Node 18+, and current Chrome, Edge, Firefox and Safari. The model is a few MB, downloaded once by the browser.

## How good is it?

On held-out, human-written Egyptian Arabizi: about **74% of words** and **37% of whole sentences** exactly right
(version 0.1.0). Typical mistakes are long vowels (`fahem` → فهم instead of فاهم) and letters Arabizi writes the
same way (`s` for both س and ص). Results, charts and the training code:
[github.com/BackendNeuron/egyptian-arabizi-franco](https://github.com/BackendNeuron/egyptian-arabizi-franco).

## Licence

MIT. The model was pre-trained on text from the Egyptian Arabic Wikipedia (CC BY-SA 4.0).
