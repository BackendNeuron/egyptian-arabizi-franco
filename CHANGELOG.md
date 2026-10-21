# Changelog

Versions of the `egyptian-arabizi-franco` Python package.

## 0.1.1
- Package page: a FastAPI backend example and tips for running it on a server. No change to the model or the
  conversion.

## 0.1.0
- First release. The model pre-trained on the Egyptian Arabic Wikipedia and fine-tuned on hand-written pairs
  (ONNX, full precision, about 13 MB), plus the rules around it: English work/tech words and numbers pass
  through, stretched letters are shortened. 75.0% of words and 40.8% of whole sentences right on the test set.
