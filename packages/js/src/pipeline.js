// The steps around the model, so it only ever sees what it was trained on. Same rules as the Python package
// (packages/python/src/egyptian_arabizi_franco/pipeline.py): keep the two in step.
//
// 1. split the text into words, spaces and everything else (punctuation, emojis, text already in Arabic)
// 2. lowercase words and shorten letters stretched three or more times ("ehhh" -> "eh")
// 3. pass through unchanged: numbers on their own, and English work/tech words; "el meeting" -> "الmeeting"
// 4. convert each run of Arabizi words with the model, at most MAX_SPAN words at a time
// 5. put everything back in its original order and spacing

const TOKEN = /[A-Za-z0-9'’]+|\s+|[^\sA-Za-z0-9'’]+/g;
const STRETCH = /([a-z])\1{2,}/g;
const ARTICLE = new Set(["el", "al", "il", "l"]);
export const MAX_SPAN = 6;

export function parseEnglish(text) {
  return new Set(
    text.split(/\r?\n/).map((w) => w.trim()).filter((w) => w && !w.startsWith("#")),
  );
}

/** The text as parts: ["keep", text] is copied as it is, ["convert", words] is one run of Arabizi. */
export function plan(text, english) {
  const items = []; // [kind, original, normalised]: kind is "space", "ar", "en" or "keep"
  for (const tok of text.match(TOKEN) ?? []) {
    if (/^\s+$/.test(tok)) {
      items.push(["space", tok, tok]);
      continue;
    }
    const w = tok.toLowerCase().replace(STRETCH, "$1");
    if (!/^[a-z0-9]/.test(w) || /^\d+$/.test(w)) items.push(["keep", tok, tok]);
    else if (english.has(w)) items.push(["en", tok, w]);
    else items.push(["ar", tok, w]);
  }

  // "el meeting" -> one kept part "الmeeting"
  const merged = [];
  for (let i = 0; i < items.length; i++) {
    const [k, orig, w] = items[i];
    if (k === "ar" && ARTICLE.has(w) && items[i + 1]?.[0] === "space" && items[i + 2]?.[0] === "en") {
      merged.push(["keep", "ال" + items[i + 2][2]]);
      i += 2;
      continue;
    }
    merged.push(k === "en" ? ["keep", orig] : [k, orig, w]);
  }

  const parts = [];
  let run = [];
  const flush = () => {
    for (let j = 0; j < run.length; j += MAX_SPAN) {
      if (j) parts.push(["keep", " "]);
      parts.push(["convert", run.slice(j, j + MAX_SPAN)]);
    }
    run = [];
  };
  merged.forEach(([k, orig, w], j) => {
    if (k === "ar") run.push(w);
    else if (k === "space" && run.length && merged[j + 1]?.[0] === "ar") return; // stays inside the run
    else {
      flush();
      parts.push(["keep", orig]);
    }
  });
  flush();
  return parts;
}

/** Puts the model's outputs (one per "convert" part, in order) back between the kept parts. */
export function assemble(parts, converted) {
  let i = 0;
  return parts.map(([kind, val]) => (kind === "convert" ? converted[i++] : val)).join("");
}
