# TB-Bridge — userspace tensor offload + Metal compute over Thunderbolt IP

A pure-Python client/server that lets a Linux CUDA training host offload
tensors to a Mac (or any peer) across a Thunderbolt 5 cable. Two modes:

- **Storage** (PUT/GET/DEL/LIST/STAT): bytes move across the cable, tensors
  live on the peer. Works against any peer.
- **Compute** (COMPUTE op, MLX backend): tensors land in **Apple unified
  memory** addressable by Metal. The Linux side can request server-side
  ops (`matmul`, `softmax`, `scaled_dot_product_attention`, ...) and the
  Mac runs them on Metal against tensors that never leave its memory.
  Linux only fetches the result. *This is the "Metal RDMA widget" — the
  Mac becomes a compute peer, not just a storage tier.*

Backend selection is automatic: MLX-backed if `pip install mlx` is
available (Apple Silicon), plain bytes otherwise. Same protocol, same
COMPUTE op — the bytes backend has a numpy-fallback implementation for
matmul/softmax/rms_norm/add/mul so the plumbing tests work on Linux.

## What this is — and isn't

**It is**: a working userspace bridge usable today. ~25–40 Gbps over TB5
IP, ~50–200 µs round-trip latency. Suitable as an offload tier for spilled
tensors when the training GPU's VRAM isn't big enough.

**It isn't**: GPU-direct RDMA. Data goes through host RAM on both ends.
For zero-copy GPU-to-GPU over Thunderbolt you need OdinLink's kernel
driver path (Linux↔Linux working today, Mac interop work-in-progress;
see [`../docs/MAC_PROTOCOL_CAPTURE.md`](../docs/MAC_PROTOCOL_CAPTURE.md)
and [`../docs/REMOTE_TENSORS.md`](../docs/REMOTE_TENSORS.md)).

## Quick start

### On the Mac (or remote host)

```bash
# Find your TB-net IP — Apple assigns one when the cable is connected:
ifconfig | grep -A 3 bridge   # look for an "inet" line under a TB-named iface
# Or check System Settings → Network → "Thunderbolt Bridge"

pip install mlx                # enables Metal-backed storage + compute
python3 bridge/tb_bridge_server.py --bind 0.0.0.0 --max-gb 96 -v
# [startup] backend=mlx  mlx_available=True  max=96.0 GB
```

### On the Linux training host

```bash
# Quick benchmark (transfer-only)
python3 bridge/benchmark.py --host <mac-tb-ip> --cuda

# Storage round-trip
python3 -c '
import torch
from bridge.tb_bridge_client import TBBridgeClient

cli = TBBridgeClient("10.0.0.2")
print(cli.info())                          # {"backend": "mlx", ...}
x = torch.randn(1024, 4096, dtype=torch.bfloat16, device="cuda")
cli.put("layer.42.kv", x)                  # offload to mac unified memory
y = cli.get("layer.42.kv", device="cuda")  # fetch back
print("round-trip OK")
'

# Mac-as-attention-accelerator demo
python3 bridge/metal_attention_demo.py --host <mac-tb-ip>
```

### COMPUTE op (Metal RDMA widget)

```python
import torch
from bridge.tb_bridge_client import TBBridgeClient
cli = TBBridgeClient("10.0.0.2")

# Push Q, K, V to mac unified memory (one-time, ~ms on TB5)
B, H, L, D = 1, 8, 4096, 64
q = torch.randn(B, H, L, D, device="cuda")
k = torch.randn(B, H, L, D, device="cuda")
v = torch.randn(B, H, L, D, device="cuda")
cli.put("q", q); cli.put("k", k); cli.put("v", v)

# Run attention on the Mac (MLX/Metal). Inputs never leave the Mac.
cli.compute("attn_out", "scaled_dot_product_attention",
            ["q", "k", "v"], scale=1.0 / D**0.5)

# Fetch only the result back to Linux GPU
result = cli.get("attn_out", device="cuda")
```

Supported ops in the **MLX backend**: `matmul`, `softmax`, `rms_norm`,
`scaled_dot_product_attention`, `add`, `mul`.
**Numpy fallback** (bytes backend): `matmul`, `softmax`, `rms_norm`,
`add`, `mul` — enough to test plumbing.

## Wire protocol

Length-prefixed binary over TCP, one request per connection. All integers
big-endian.

```
Request:
  u8   op           PUT=1, GET=2, DEL=3, LIST=4, STAT=5, COMPUTE=6, INFO=7
  u32  key_len
  u8[] key          (≤256 B utf-8; for COMPUTE this is the OUTPUT key)
  PUT only:
    u32  meta_len
    u8[] meta_json  (dtype, shape, kind: numpy|torch|mlx)
    u64  data_len
    u8[] data       (raw tensor bytes)
  COMPUTE only:
    u32  expr_len
    u8[] expr_json  ({"op": "matmul", "args": [...], "kwargs": {...}})

Response:
  u8   status       0=OK, 1=NOT_FOUND, 2=BAD_OP, 3=OOM, 4=PROTOCOL, 6=COMPUTE_FAIL
  GET ok:
    u32  meta_len + meta_json + u64 data_len + data
  LIST ok:
    u32  n  + n × ( u32 keylen + key )
  STAT ok:
    u64  total_bytes + u32 num_keys
  COMPUTE ok:
    u32  meta_len + meta_json   (no data — fetch via GET)
  INFO ok:
    u32  info_len + info_json   ({"backend": "mlx"|"bytes", "mlx_available": bool, ...})
```

## bfloat16 caveat

NumPy has no native bf16. The client carries bf16 tensors as `uint16` on
the wire and rehydrates them with `tensor.view(torch.bfloat16)` on the
receive side. Lossless.

## Performance ceiling

Theoretical TB5 IP layer: 80 Gbps raw → ~10 GB/s payload after framing.
Real-world over Apple's bridge: 25–40 Gbps (~3–5 GB/s). Latency floor:
the TCP stack + scheduler costs ~50 µs each direction. So this is best
for tensors ≥ 16 MB; smaller payloads spend most of their time in
syscall overhead.

For comparison: a single PCIe Gen4 x16 link is ~64 GB/s, so the bridge
is roughly 5–15× slower than local VRAM movement. The win is *capacity*,
not bandwidth: it lets you treat the Mac's 96–192 GB unified memory as
an extension of the training host's spill pool.

## What's missing

- **Connection reuse**: every operation opens a fresh socket. Fine for
  bulk transfers, painful for tight loops with sub-MB tensors.
- **Compression**: bf16 + lz4 would help if you're often offloading
  near-zero activations. Out of scope for v1.
- **Encryption**: zero. Run it on a private link only.
- **MLX zero-copy**: Mac side should be able to import the resident buffer
  into an `mlx.array` without a copy (unified memory). Hook stub exists
  but not yet wired (`mlx_helpers.py`).
