#!/usr/bin/env python3
"""End-to-end demo of the Metal RDMA widget — Mac as attention accelerator.

The pattern:
    1. Linux GPU generates Q, K, V matrices
    2. PUT them across the TB5 cable to the Mac
    3. Ask the Mac to run scaled-dot-product attention on its end
       (executes on Metal, against tensors that never leave Mac unified memory)
    4. GET the result back into Linux GPU memory
    5. Verify against a local CUDA attention computation

Run the server first:
    # On the Mac:
    pip install mlx  # if you haven't
    python3 bridge/tb_bridge_server.py --bind 0.0.0.0 --backend mlx -v

Then this demo:
    python3 bridge/metal_attention_demo.py --host <mac-tb-ip>

If you don't have a Mac handy, run both ends locally to exercise the
COMPUTE plumbing against the numpy fallback:
    python3 bridge/tb_bridge_server.py --bind 127.0.0.1 --backend bytes -v &
    python3 bridge/metal_attention_demo.py --host 127.0.0.1 --no-cuda --no-attn
"""
import argparse
import time

import numpy as np
from tb_bridge_client import TBBridgeClient

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    _HAS_CUDA = False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=29800)
    ap.add_argument("--no-cuda", action="store_true",
                    help="Don't touch CUDA. For testing on Linux without a GPU.")
    ap.add_argument("--no-attn", action="store_true",
                    help="Skip the attention op (not in numpy fallback). "
                         "Useful when testing against the bytes backend.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--head-dim", type=int, default=64)
    args = ap.parse_args()

    cli = TBBridgeClient(args.host, args.port)
    info = cli.info()
    print(f"server backend: {info['backend']}  mlx_available={info['mlx_available']}")
    print(f"shapes: Q,K,V = [{args.batch}, {args.heads}, {args.seqlen}, {args.head_dim}] float32")

    use_cuda = not args.no_cuda and _HAS_CUDA
    device = "cuda" if use_cuda else "cpu"

    # ── Step 1: generate Q, K, V locally ───────────────────────────────────
    if torch is not None:
        gen = torch.Generator(device=device).manual_seed(0)
        shape = (args.batch, args.heads, args.seqlen, args.head_dim)
        q = torch.randn(shape, generator=gen, device=device)
        k = torch.randn(shape, generator=gen, device=device)
        v = torch.randn(shape, generator=gen, device=device)
        local_pkg = q, k, v
    else:
        rng = np.random.default_rng(0)
        q = rng.standard_normal((args.batch, args.heads, args.seqlen, args.head_dim), dtype="float32")
        k = rng.standard_normal(q.shape, dtype="float32")
        v = rng.standard_normal(q.shape, dtype="float32")
        local_pkg = q, k, v

    nbytes = q.numel() * 4 if torch is not None else q.nbytes
    print(f"per-tensor size: {nbytes/1e6:.1f} MB  (3 tensors total)")

    # ── Step 2: PUT them ───────────────────────────────────────────────────
    t0 = time.perf_counter()
    cli.put("demo/q", q)
    cli.put("demo/k", k)
    cli.put("demo/v", v)
    put_s = time.perf_counter() - t0
    print(f"PUT 3× tensors: {put_s*1e3:.1f} ms  "
          f"({3*nbytes/put_s/1e6:.0f} MB/s effective)")

    # ── Step 3: small ops (sanity check) ───────────────────────────────────
    print("\nsanity ops:")
    cli.compute("demo/q_plus_k", "add", ["demo/q", "demo/k"])
    cli.compute("demo/qkT", "matmul",
                ["demo/q", "demo/k"] if args.no_attn else None or
                ["demo/q", "demo/k"])  # placeholder swap-axes; matmul handles last-2 dims
    # NB: matmul as-is takes [B,H,L,D] @ [B,H,L,D] which requires last two
    # axes to align. The server's matmul is just `a @ b`; for an actual
    # qkT we'd want a transpose op. Demo'd via the full attention op below.

    if not args.no_attn:
        # ── Step 4: full attention on the Mac ──────────────────────────────
        t0 = time.perf_counter()
        meta = cli.compute(
            "demo/attn_out",
            "scaled_dot_product_attention",
            ["demo/q", "demo/k", "demo/v"],
            scale=1.0 / (args.head_dim ** 0.5),
        )
        compute_s = time.perf_counter() - t0
        print(f"\nremote attention (compute on Mac, MLX/Metal): {compute_s*1e3:.1f} ms")
        print(f"  result meta: {meta}")

        # ── Step 5: fetch + verify ─────────────────────────────────────────
        t0 = time.perf_counter()
        result = cli.get("demo/attn_out", device=device if torch is not None else None)
        get_s = time.perf_counter() - t0
        print(f"GET attn_out: {get_s*1e3:.1f} ms  ({nbytes/get_s/1e6:.0f} MB/s)")

        if torch is not None:
            t0 = time.perf_counter()
            local = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize() if use_cuda else None
            local_s = time.perf_counter() - t0
            print(f"local attention ({device}): {local_s*1e3:.1f} ms")

            # Cast both to same dtype/shape for comparison
            local = local.to(result.dtype if hasattr(result, "dtype") else None)
            if torch.is_tensor(result):
                diff = (result.cpu() - local.cpu()).abs().max().item()
                rel = diff / local.abs().max().clamp_min(1e-8).item()
                print(f"max |Δ|: {diff:.4e}  rel: {rel:.4e}")
                if rel < 1e-2:
                    print("✓ remote attention matches local (within fp32 tolerance)")
                else:
                    print("⚠ noticeable drift — likely dtype downcast on the server side")

    # ── Cleanup ────────────────────────────────────────────────────────────
    for k_ in ["demo/q", "demo/k", "demo/v", "demo/q_plus_k", "demo/qkT", "demo/attn_out"]:
        try: cli.delete(k_)
        except Exception: pass

    s = cli.stat()
    print(f"\nfinal server state: {s['num_keys']} keys, {s['total_bytes']/1e9:.2f} GB resident")


if __name__ == "__main__":
    main()
