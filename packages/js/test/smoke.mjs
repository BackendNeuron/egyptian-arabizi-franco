// npm test: checks the pipeline rules (no model needed), then converts a few lines if a model is bundled.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { assemble, parseEnglish, plan } from "../src/pipeline.js";

const english = parseEnglish(readFileSync(new URL("../english_words.txt", import.meta.url), "utf8"));
const shape = (t) => {
  const p = plan(t, english);
  return assemble(p, p.filter(([k]) => k === "convert").map(([, w]) => `[${w.join(" ")}]`));
};
assert.equal(shape("ezayak 3amel ehhh??"), "[ezayak 3amel eh]??");
assert.equal(shape("3andy el meeting bokra 😂"), "[3andy] الmeeting [bokra] 😂");
assert.equal(shape("el password ghalat!"), "الpassword [ghalat]!");
assert.equal(shape("2026 hayeb2a 7elw"), "2026 [hayeb2a 7elw]");
assert.equal(shape("Wa7ashtni AWYYYY"), "[wa7ashtni awy]");
console.log("pipeline rules: ok");

if (existsSync(new URL("../models/encoder.onnx", import.meta.url))) {
  const { convert } = await import("../src/index.js");
  for (const t of ["ezayak 3amel eh", "eh el a5bar", "3andy el meeting bokra"]) console.log(t, "→", await convert(t));
} else {
  console.log("no bundled model yet (models/ is filled by tools/sync_packages.py): skipped the model check");
}
