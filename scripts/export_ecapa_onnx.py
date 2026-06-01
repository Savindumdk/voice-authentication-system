"""
Export the speaker-embedding model to ONNX for faster, portable inference.

Run on a machine where the SpeechBrain model loads (ideally the GPU host):

    python scripts/export_ecapa_onnx.py --output models/ecapa.onnx

Then serve it:

    EMBEDDING_BACKEND=onnx
    ONNX_MODEL_PATH=models/ecapa.onnx
    pip install onnxruntime-gpu   # or onnxruntime on CPU

VALIDATE before trusting in production: this script prints a parity cosine
between the PyTorch and ONNX embeddings — it should be > 0.999. SpeechBrain's
full `encode_batch` (features + norm + embedding net) occasionally hits ops
ONNX can't trace; if export fails, export only the embedding network and run
Fbank feature extraction in Python before the ONNX session.

TensorRT: once you have a validated ONNX file, build an engine with
`trtexec --onnx=models/ecapa.onnx --fp16 --saveEngine=models/ecapa.trt`
(or torch-tensorrt) and serve via onnxruntime's TensorRT execution provider.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="models/ecapa.onnx")
    ap.add_argument("--source", default=None, help="HF model id (default: config)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()

    from config import settings
    from speechbrain.inference.classifiers import EncoderClassifier

    source = args.source or settings.SPEAKER_VERIFIER_MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {source} on {device} ...")
    enc = EncoderClassifier.from_hparams(source=source, run_opts={"device": device})

    class Wrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, wav):
            return self.encoder.encode_batch(wav)

    model = Wrapper(enc).eval().to(device)
    dummy = torch.randn(1, int(args.seconds * args.sample_rate), device=device)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print("Exporting to ONNX ...")
    torch.onnx.export(
        model,
        (dummy,),
        args.output,
        input_names=["waveform"],
        output_names=["embedding"],
        dynamic_axes={"waveform": {0: "batch", 1: "samples"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"✅ Saved {args.output}")

    # Parity check.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        ref = model(dummy).detach().cpu().numpy().reshape(-1)
        out = sess.run(None, {"waveform": dummy.cpu().numpy()})[0].reshape(-1)
        cos = float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-9))
        print(f"Parity cosine(PyTorch, ONNX) = {cos:.5f} (want > 0.999)")
        if cos < 0.999:
            print("⚠️ Parity below 0.999 — do NOT use this export until investigated.")
    except Exception as exc:  # noqa: BLE001
        print(f"(skipped parity check: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
