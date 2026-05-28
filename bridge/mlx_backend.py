"""
MLX backend for the tb_bridge tensor store.

Stores incoming tensors as `mlx.core.array` objects. On Apple Silicon
those live in unified memory, which means a Metal kernel can read them
without a copy. The widget half of "Metal RDMA": once a tensor lands
here, the Mac can compute on it via MLX/Metal without round-tripping
to the requesting machine.

On non-Apple hosts (Linux dev box), import of `mlx.core` fails; the
factory returns a `BytesBackend` that stores raw bytes — same protocol,
just no Metal-backed storage. Lets us unit-test the server end-to-end
without a Mac in the loop.

Protocol additions vs. v1 bridge:

  Stored value metadata gains a `kind: "mlx"` variant for arrays
  that were promoted into MLX. Bytes-on-wire format unchanged —
  client serialises into raw bytes, server hydrates into mlx.array
  on receive, dehydrates on send.

  New op: COMPUTE (=6). Server-side execution against stored MLX
  arrays. Operands are referenced by key; result is stored under
  a new key (no automatic streaming back — fetch with GET).
    Request:
      u8   op = 6
      u32  out_key_len + utf8 out_key
      u32  expr_len    + utf8 expr_json
    Response:
      u8   status      (0=OK, 1=NOT_FOUND, 6=COMPUTE_FAIL)
      For OK: u32 meta_len + meta_json   (the resulting array's meta)
"""
import json
import threading
from typing import Any, Optional, Tuple

import numpy as np

# ── MLX availability detection ───────────────────────────────────────────────
try:
    import mlx.core as mx  # type: ignore
    _MLX_OK = True
except Exception:
    mx = None  # type: ignore
    _MLX_OK = False


# ── Metadata helpers ────────────────────────────────────────────────────────
# Dtype mapping: meta carries an unambiguous string; the backend decides how
# to materialise. bfloat16 is the awkward case — numpy doesn't have it, MLX
# does. We carry bf16 bytes as uint16 on the wire and view-cast on rehydrate.

_TORCH_DTYPE_NUMPY = {
    "torch.float32": "float32",
    "torch.float16": "float16",
    "torch.bfloat16": "uint16",
    "torch.float64": "float64",
    "torch.int8": "int8",
    "torch.int16": "int16",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.uint8": "uint8",
    "torch.bool": "bool",
}

_MLX_DTYPE_BY_NAME = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float64": "float32",   # mlx doesn't have float64; downcast silently
    "int8":    "int8",
    "int16":   "int16",
    "int32":   "int32",
    "int64":   "int64",
    "uint8":   "uint8",
    "bool":    "bool",
}


def _resolve_dtype_str(meta: dict) -> str:
    """Return a canonical dtype name (numpy-compatible string for bytes,
    or 'bfloat16' if the source was torch bf16)."""
    kind = meta.get("kind")
    if kind == "torch":
        dt = meta["dtype"]
        if dt == "torch.bfloat16":
            return "bfloat16"
        return _TORCH_DTYPE_NUMPY.get(dt, "float32").removeprefix("torch.")
    if kind == "numpy":
        return str(meta["dtype"])
    if kind == "mlx":
        return str(meta["dtype"])
    return "float32"


def _bytes_to_numpy(data: bytes, meta: dict) -> np.ndarray:
    dt = _resolve_dtype_str(meta)
    if dt == "bfloat16":
        # carry as uint16; caller may view-cast if it wants bf16 semantics
        arr = np.frombuffer(data, dtype="uint16").reshape(meta["shape"]).copy()
    else:
        arr = np.frombuffer(data, dtype=dt).reshape(meta["shape"]).copy()
    return arr


def _numpy_to_bytes(arr: np.ndarray) -> bytes:
    return arr.tobytes()


# ── Backend ABC + two implementations ───────────────────────────────────────
class _BaseBackend:
    """Common protocol: put/get/delete/keys/stat/compute, with raw bytes
    in/out. Subclasses decide internal representation."""

    def put(self, key: str, meta: dict, data: bytes) -> None: ...
    def get(self, key: str) -> Optional[Tuple[dict, bytes]]: ...
    def delete(self, key: str) -> bool: ...
    def keys(self) -> list: ...
    def stat(self) -> Tuple[int, int]: ...
    def compute(self, out_key: str, expr: dict) -> Tuple[bool, Optional[dict]]: ...
    def info(self) -> dict: ...


class BytesBackend(_BaseBackend):
    """Plain in-memory bytes store. Used when MLX isn't available (Linux dev
    box) or when the user explicitly disables MLX."""

    def __init__(self, max_bytes: int):
        self._store: dict[str, Tuple[dict, bytes]] = {}
        self._bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def put(self, key, meta, data):
        with self._lock:
            old = self._store.get(key)
            new_bytes = self._bytes - (len(old[1]) if old else 0) + len(data)
            if new_bytes > self._max_bytes:
                raise MemoryError("backend full")
            self._store[key] = (meta, data)
            self._bytes = new_bytes

    def get(self, key):
        with self._lock:
            return self._store.get(key)

    def delete(self, key):
        with self._lock:
            old = self._store.pop(key, None)
            if old is None:
                return False
            self._bytes -= len(old[1])
            return True

    def keys(self):
        with self._lock:
            return list(self._store.keys())

    def stat(self):
        with self._lock:
            return self._bytes, len(self._store)

    def compute(self, out_key, expr):
        # BytesBackend can do a small numpy-shaped subset for testing without
        # MLX. Keeps server-side compute API uniform across backends.
        op = expr.get("op")
        args = [self._store.get(k) for k in expr.get("args", [])]
        if any(a is None for a in args):
            return False, None
        try:
            np_args = [_bytes_to_numpy(d, m) for (m, d) in args]
            result_arr, result_meta = _compute_numpy(op, np_args, expr.get("kwargs", {}))
        except Exception:
            return False, None
        self.put(out_key, result_meta, _numpy_to_bytes(result_arr))
        return True, result_meta

    def info(self):
        return {"backend": "bytes", "mlx_available": _MLX_OK,
                "max_bytes": self._max_bytes, "bytes": self._bytes}


class MLXBackend(_BaseBackend):
    """Tensors land in MLX-managed unified memory. Metal kernels see them
    without a copy. Falls back to bytes for dtypes MLX can't represent."""

    def __init__(self, max_bytes: int):
        if not _MLX_OK:
            raise RuntimeError("mlx.core not importable — use BytesBackend on Linux")
        # Two stores: arrays we successfully promoted into MLX, and a raw
        # bytes shadow for things MLX rejects (rare for our dtypes, but
        # keeps the protocol total).
        self._mlx: dict[str, Tuple[dict, "mx.array"]] = {}
        self._bytes_fallback: dict[str, Tuple[dict, bytes]] = {}
        self._mlx_bytes = 0
        self._fb_bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _bytes_to_mlx(data: bytes, meta: dict) -> Optional["mx.array"]:
        """Try to materialise into MLX. Return None if dtype/shape is bad."""
        if not _MLX_OK:
            return None
        try:
            dt_str = _resolve_dtype_str(meta)
            np_arr = _bytes_to_numpy(data, meta)
            # bf16: numpy carried it as uint16; MLX has a real bf16
            if dt_str == "bfloat16":
                # MLX from_buffer / view doesn't expose bf16-from-uint16 directly,
                # so go via numpy float32 and downcast. Costs a copy but correct.
                f32 = np.empty(np_arr.shape, dtype="float32")
                # bf16 bit pattern in uint16: high 16 bits of fp32
                f32.view("uint32")[...] = np_arr.astype("uint32") << 16
                arr = mx.array(f32).astype(mx.bfloat16)
                return arr
            mlx_dt = _MLX_DTYPE_BY_NAME.get(dt_str)
            if mlx_dt is None:
                return None
            arr = mx.array(np_arr).astype(getattr(mx, mlx_dt))
            return arr
        except Exception:
            return None

    @staticmethod
    def _mlx_to_bytes(arr: "mx.array") -> Tuple[bytes, dict]:
        # bf16 → numpy: go via float32 then take the high u16 bits
        dt = str(arr.dtype).split(".")[-1]
        if dt == "bfloat16":
            f32 = np.array(arr.astype(mx.float32))
            u32 = f32.view("uint32")
            u16 = (u32 >> 16).astype("uint16")
            return u16.tobytes(), {
                "kind": "mlx", "dtype": "bfloat16", "shape": list(arr.shape)
            }
        np_arr = np.array(arr)
        return np_arr.tobytes(), {
            "kind": "mlx", "dtype": dt, "shape": list(arr.shape)
        }

    # ── interface ───────────────────────────────────────────────────────────
    def put(self, key, meta, data):
        with self._lock:
            self._evict_key(key)
            total = self._mlx_bytes + self._fb_bytes + len(data)
            if total > self._max_bytes:
                raise MemoryError("backend full")
            arr = self._bytes_to_mlx(data, meta)
            if arr is not None:
                meta = dict(meta); meta["kind"] = "mlx"; meta["dtype"] = str(arr.dtype).split(".")[-1]
                meta["shape"] = list(arr.shape)
                self._mlx[key] = (meta, arr)
                self._mlx_bytes += int(arr.size * arr.dtype.size) if hasattr(arr.dtype, "size") else len(data)
            else:
                self._bytes_fallback[key] = (meta, data)
                self._fb_bytes += len(data)

    def _evict_key(self, key):
        if key in self._mlx:
            meta, arr = self._mlx.pop(key)
            self._mlx_bytes -= int(arr.size * arr.dtype.size) if hasattr(arr.dtype, "size") else 0
        if key in self._bytes_fallback:
            meta, data = self._bytes_fallback.pop(key)
            self._fb_bytes -= len(data)

    def get(self, key):
        with self._lock:
            if key in self._mlx:
                meta, arr = self._mlx[key]
                data, meta_out = self._mlx_to_bytes(arr)
                return meta_out, data
            if key in self._bytes_fallback:
                return self._bytes_fallback[key]
            return None

    def delete(self, key):
        with self._lock:
            if key not in self._mlx and key not in self._bytes_fallback:
                return False
            self._evict_key(key)
            return True

    def keys(self):
        with self._lock:
            return list(set(self._mlx) | set(self._bytes_fallback))

    def stat(self):
        with self._lock:
            return (self._mlx_bytes + self._fb_bytes,
                    len(self._mlx) + len(self._bytes_fallback))

    def compute(self, out_key, expr):
        """Execute an MLX op against stored arrays. Result is stored under
        out_key; client GETs it back. Keeps inputs in unified memory the
        whole time — Linux only sees the final result if it asks for it."""
        with self._lock:
            try:
                op = expr.get("op")
                args = []
                for k in expr.get("args", []):
                    if k not in self._mlx:
                        return False, None
                    args.append(self._mlx[k][1])
                result = _compute_mlx(op, args, expr.get("kwargs", {}))
                if result is None:
                    return False, None
                # Force eval so the buffer is materialised before we measure size
                mx.eval(result)
                meta = {"kind": "mlx",
                        "dtype": str(result.dtype).split(".")[-1],
                        "shape": list(result.shape)}
                self._evict_key(out_key)
                self._mlx[out_key] = (meta, result)
                self._mlx_bytes += int(result.size * result.dtype.size) if hasattr(result.dtype, "size") else 0
                return True, meta
            except Exception:
                return False, None

    def info(self):
        return {"backend": "mlx", "mlx_available": True,
                "max_bytes": self._max_bytes,
                "mlx_bytes": self._mlx_bytes,
                "fallback_bytes": self._fb_bytes}


# ── Compute primitives ──────────────────────────────────────────────────────
def _compute_mlx(op: str, args, kwargs):
    if op == "matmul":
        a, b = args[0], args[1]
        return a @ b
    if op == "softmax":
        x = args[0]
        dim = kwargs.get("dim", -1)
        return mx.softmax(x, axis=dim)
    if op == "rms_norm":
        x = args[0]
        eps = kwargs.get("eps", 1e-5)
        var = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(var + eps)
    if op == "scaled_dot_product_attention":
        q, k, v = args[0], args[1], args[2]
        scale = kwargs.get("scale", 1.0 / (q.shape[-1] ** 0.5))
        # q,k,v: [B, H, L, D]
        scores = (q @ mx.swapaxes(k, -1, -2)) * scale
        attn = mx.softmax(scores, axis=-1)
        return attn @ v
    if op == "add":
        return args[0] + args[1]
    if op == "mul":
        return args[0] * args[1]
    return None


def _compute_numpy(op: str, args, kwargs):
    """Numpy fallback for the BytesBackend. Returns (array, meta)."""
    if op == "matmul":
        out = args[0] @ args[1]
    elif op == "softmax":
        x = args[0]; dim = kwargs.get("dim", -1)
        x = x - x.max(axis=dim, keepdims=True)
        e = np.exp(x)
        out = e / e.sum(axis=dim, keepdims=True)
    elif op == "rms_norm":
        x = args[0]; eps = kwargs.get("eps", 1e-5)
        var = (x * x).mean(axis=-1, keepdims=True)
        out = x / np.sqrt(var + eps)
    elif op == "add":
        out = args[0] + args[1]
    elif op == "mul":
        out = args[0] * args[1]
    else:
        raise ValueError(f"numpy backend doesn't implement op {op!r}")
    meta = {"kind": "numpy", "dtype": str(out.dtype), "shape": list(out.shape)}
    return out, meta


# ── Public factory ──────────────────────────────────────────────────────────
def make_backend(backend: str = "auto", max_bytes: int = 64 * (1 << 30)) -> _BaseBackend:
    """Return the requested backend. 'auto' picks MLX on Apple Silicon,
    BytesBackend elsewhere."""
    if backend == "bytes":
        return BytesBackend(max_bytes)
    if backend == "mlx":
        return MLXBackend(max_bytes)  # raises if MLX not importable
    if backend == "auto":
        return MLXBackend(max_bytes) if _MLX_OK else BytesBackend(max_bytes)
    raise ValueError(f"unknown backend {backend!r}")
