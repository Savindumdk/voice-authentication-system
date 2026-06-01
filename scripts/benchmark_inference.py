"""
Benchmark fp32 vs fp16 (USE_AMP) embedding inference: latency + accuracy drift.

Run on the GPU host:

    python scripts/benchmark_inference.py --iters 50 --seconds 4

Reports mean/p95 latency for fp32 and fp16 and the cosine drift between their
embeddings. Use it to decide whether to set USE_AMP=true and whether the
verification threshold needs recalibrating (drift < 0.001 ≈ safe to enable).
"""

import argparse
import statistics
import time

import torch


def _bench(extractor, signal, iters):
    for _ in range(3):  # warmup
        extractor.extract(signal)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    emb = None
    for _ in range(iters):
        t0 = time.perf_counter()
        emb = extractor.extract(signal)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return emb, times


def _report(name, times):
    ordered = sorted(times)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    print(f"{name}: mean={statistics.mean(times):.2f}ms  p95={p95:.2f}ms  min={min(times):.2f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--sample-rate", type=int, default=16000)
    args = ap.parse_args()

    import config
    import embeddings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠️ No CUDA detected — AMP is a no-op; this benchmark is only meaningful on GPU.")
    signal = torch.randn(1, int(args.seconds * args.sample_rate), device=device)

    config.settings.USE_AMP = False
    embeddings._backend = None
    emb32, t32 = _bench(embeddings.get_backend(), signal, args.iters)

    config.settings.USE_AMP = True
    embeddings._backend = None
    emb16, t16 = _bench(embeddings.get_backend(), signal, args.iters)

    _report("fp32", t32)
    _report("fp16", t16)
    print(f"speedup (fp32/fp16) = {statistics.mean(t32) / max(statistics.mean(t16), 1e-6):.2f}x")

    a = emb32.squeeze().float().flatten()
    b = emb16.squeeze().float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    print(f"cosine(fp32, fp16 embedding) = {cos:.5f} (1.0 = no drift)")
    if cos < 0.999:
        print("⚠️ Non-trivial drift — recalibrate VERIFICATION_THRESHOLD before enabling AMP.")


if __name__ == "__main__":
    main()
