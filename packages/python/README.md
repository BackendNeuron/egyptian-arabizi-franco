# egyptian-arabizi-franco

Egyptian **Arabizi** (Franco-Arabic, "Franco") → **Arabic**, offline, on a CPU:

```
ezayak 3amel eh       →  ازيك عامل ايه
3andy el meeting bokra →  عندي الmeeting بكرة
```

A small character-level transformer (about 3M parameters, bundled with the package, no internet needed) plus a
few rules around it: English work and tech words stay in Latin letters, numbers, punctuation and emojis pass
through, and stretched letters (`awyyyy`) are shortened before conversion.

## Install

```bash
pip install egyptian-arabizi-franco
```

Python 3.9+. Depends only on `onnxruntime` and `numpy` (no PyTorch).

## Use

```python
from egyptian_arabizi_franco import convert

convert("ezayak 3amel eh")                       # 'ازيك عامل ايه'
convert(["yalla bina", "el meeting bokra"])      # a list in, a list out (faster than one by one)
```

The first call loads the model (a second or so); later calls reuse it. For more control, keep your own
converter:

```python
from egyptian_arabizi_franco import Converter

c = Converter(threads=2)            # limit CPU threads
c.convert("msh fahem 7aga")

raw = Converter(raw=True)           # the model alone, without the rules around it
```

From the command line:

```bash
egyptian-arabizi-franco "ezayak 3amel eh"
cat chat.txt | egyptian-arabizi-franco          # one line in, one line out
```

## In a FastAPI backend

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
from egyptian_arabizi_franco import Converter

app = FastAPI()
converter = Converter()  # load the model once, when the server starts


class ConvertRequest(BaseModel):
    texts: list[str] = Field(max_length=100)  # e.g. {"texts": ["ezayak 3amel eh", "eh el a5bar"]}


@app.post("/convert")
def convert(req: ConvertRequest):  # plain def: FastAPI runs it in a worker thread
    return {"arabic": converter.convert([t[:500] for t in req.texts])}
```

```bash
pip install egyptian-arabizi-franco fastapi uvicorn
uvicorn main:app
curl -X POST localhost:8000/convert -H "Content-Type: application/json" -d '{"texts": ["eh el a5bar"]}'
# {"arabic": ["ايه الاخبار"]}
```

The `curl` line is for bash (macOS, Linux, Git Bash). In Windows PowerShell, `curl` means a different built-in
command: use `curl.exe` with the JSON in a file (`-d "@body.json"`), or open http://localhost:8000/docs and
try the endpoint there.

Tips for a backend:

- **Create the `Converter` once**, at module level or at startup. Loading it takes around a second, so doing it on
  every request would add that to each one.
- **Use `def`, not `async def`, for the endpoint.** The conversion is CPU work, so FastAPI should run it in its
  thread pool instead of blocking the event loop. One converter can be used from several threads at once.
- **Send lists when you have several texts.** One call with 50 texts is much faster than 50 calls with one.
- **Limit the input size** (the example allows 100 texts of up to 500 characters) so one huge request can't hog
  the CPU.
- **Speed is a rough guide, not a promise:** in our tests on an ordinary laptop CPU, a short sentence took
  somewhere around 50 to 100 ms on its own, and less per sentence when sent in a batch. Your numbers will depend
  on the machine, the text length and how many requests arrive together, so measure on your own server.

## How good is it?

On held-out, human-written Egyptian Arabizi: **75% of words** and **41% of whole sentences** exactly right
(the 0.1.x model, with the rules above). Typical mistakes are long vowels (`fahem` → فهم instead of فاهم) and letters Arabizi writes the
same way (`s` for both س and ص). Results, charts and the training code:
[github.com/BackendNeuron/egyptian-arabizi-franco](https://github.com/BackendNeuron/egyptian-arabizi-franco).

## Licence

MIT. The model was pre-trained on text from the [Egyptian Arabic Wikipedia](https://arz.wikipedia.org), by its
contributors, used under the
[Creative Commons Attribution-ShareAlike 4.0 licence](https://creativecommons.org/licenses/by-sa/4.0/). No Wikipedia
text is included in this package, only the trained model.

Built with [PyTorch](https://pytorch.org) (training) and [ONNX Runtime](https://onnxruntime.ai) (running the
model).
