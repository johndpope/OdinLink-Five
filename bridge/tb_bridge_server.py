#!/usr/bin/env python3
"""
TB-Bridge server — runs on the Mac (or any host with extra RAM/VRAM).

Holds tensor blobs in unified memory and serves push/pull over a length-
prefixed TCP protocol. Intended to run on a Mac across a Thunderbolt 5
cable from the training host; the TB-net link gives 25-40 Gbps at the
IP layer without any kernel-driver work.

Why this exists: RDMA-level Mac↔Linux interop in OdinLink-Five still
requires Apple-protocol XDomain login work that hasn't been verified
against real hardware. This bridge is the userspace fallback that
works today — slower than RDMA but immediately usable.

Storage backend selection:
- On Apple Silicon with `pip install mlx`, tensors land as mlx.core.array
  in unified memory — Metal kernels can read them zero-copy. Adds the
  COMPUTE op so the Linux client can offload work (matmul/softmax/
  attention/...) and only fetch the result.
- Elsewhere, falls back to a plain bytes store. Same protocol, same
  COMPUTE op (numpy-backed subset), so unit tests work on Linux too.

Wire format (per request, all big-endian on the wire):
    u8   op     (PUT=1, GET=2, DEL=3, LIST=4, STAT=5, COMPUTE=6, INFO=7)
    u32  key_len
    u8[] key      (UTF-8 string, ≤ 256 bytes)
    PUT:
        u32  meta_len
        u8[] meta_json   (dtype, shape, device, etc.)
        u64  data_len
        u8[] data        (raw tensor bytes)
    COMPUTE:
        u32  expr_len
        u8[] expr_json   ({"op": "matmul", "args": [...], "kwargs": {...}})
        key field above is the OUTPUT key — operand keys are in expr.args
    Response:
        u8   status (0=OK, 1=NOT_FOUND, 2=BAD_OP, 3=OOM, 4=PROTOCOL,
                     6=COMPUTE_FAIL)
        GET ok:     u32 meta_len + meta_json + u64 data_len + data
        LIST ok:    u32 count + count × (u32 keylen + keystr)
        STAT ok:    u64 total_bytes + u32 num_keys
        COMPUTE ok: u32 meta_len + meta_json   (no data — fetch via GET)
        INFO ok:    u32 info_len + info_json
"""
import argparse
import json
import os
import socket
import struct
import sys
import threading
import time

from mlx_backend import make_backend, _MLX_OK

OP_PUT = 1
OP_GET = 2
OP_DEL = 3
OP_LIST = 4
OP_STAT = 5
OP_COMPUTE = 6
OP_INFO = 7

STATUS_OK = 0
STATUS_NOT_FOUND = 1
STATUS_BAD_OP = 2
STATUS_OOM = 3
STATUS_PROTOCOL = 4
STATUS_COMPUTE_FAIL = 6


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError(f"peer closed with {remaining}/{n} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_exact(sock: socket.socket, buf):
    view = memoryview(buf)
    while view:
        sent = sock.send(view)
        view = view[sent:]


def handle_client(sock, addr, backend, verbose):
    """One client = one request/response cycle, then close."""
    try:
        op = _recv_exact(sock, 1)[0]
        key_len = struct.unpack(">I", _recv_exact(sock, 4))[0]
        if key_len > 256:
            sock.sendall(bytes([STATUS_PROTOCOL]))
            return
        key = _recv_exact(sock, key_len).decode("utf-8") if key_len else ""

        if op == OP_PUT:
            meta_len = struct.unpack(">I", _recv_exact(sock, 4))[0]
            meta = json.loads(_recv_exact(sock, meta_len).decode("utf-8"))
            data_len = struct.unpack(">Q", _recv_exact(sock, 8))[0]
            data = _recv_exact(sock, data_len)
            try:
                backend.put(key, meta, data)
                sock.sendall(bytes([STATUS_OK]))
                if verbose:
                    print(f"[PUT] {key} meta_kind={meta.get('kind')} {data_len/1e6:.1f}MB")
            except MemoryError:
                sock.sendall(bytes([STATUS_OOM]))

        elif op == OP_GET:
            entry = backend.get(key)
            if entry is None:
                sock.sendall(bytes([STATUS_NOT_FOUND]))
                return
            meta, data = entry
            meta_bytes = json.dumps(meta).encode("utf-8")
            hdr = (bytes([STATUS_OK])
                   + struct.pack(">I", len(meta_bytes)) + meta_bytes
                   + struct.pack(">Q", len(data)))
            _send_exact(sock, hdr)
            _send_exact(sock, data)
            if verbose:
                print(f"[GET] {key} -> {len(data)/1e6:.1f}MB")

        elif op == OP_DEL:
            ok = backend.delete(key)
            sock.sendall(bytes([STATUS_OK if ok else STATUS_NOT_FOUND]))
            if verbose:
                print(f"[DEL] {key} ok={ok}")

        elif op == OP_LIST:
            keys = backend.keys()
            parts = [bytes([STATUS_OK]), struct.pack(">I", len(keys))]
            for k in keys:
                kb = k.encode("utf-8")
                parts.append(struct.pack(">I", len(kb)))
                parts.append(kb)
            sock.sendall(b"".join(parts))
            if verbose:
                print(f"[LIST] -> {len(keys)} keys")

        elif op == OP_STAT:
            total, n = backend.stat()
            sock.sendall(bytes([STATUS_OK]) + struct.pack(">QI", total, n))
            if verbose:
                print(f"[STAT] {total/1e9:.2f}GB across {n} keys")

        elif op == OP_COMPUTE:
            expr_len = struct.unpack(">I", _recv_exact(sock, 4))[0]
            expr = json.loads(_recv_exact(sock, expr_len).decode("utf-8"))
            ok, result_meta = backend.compute(key, expr)
            if not ok:
                sock.sendall(bytes([STATUS_COMPUTE_FAIL]))
                if verbose:
                    print(f"[COMPUTE] {expr.get('op')} -> FAIL")
                return
            meta_bytes = json.dumps(result_meta).encode("utf-8")
            sock.sendall(bytes([STATUS_OK]) + struct.pack(">I", len(meta_bytes)) + meta_bytes)
            if verbose:
                print(f"[COMPUTE] {expr.get('op')}({','.join(expr.get('args', []))}) "
                      f"-> {key}  shape={result_meta.get('shape')} dtype={result_meta.get('dtype')}")

        elif op == OP_INFO:
            info = backend.info()
            ib = json.dumps(info).encode("utf-8")
            sock.sendall(bytes([STATUS_OK]) + struct.pack(">I", len(ib)) + ib)
            if verbose:
                print(f"[INFO] {info}")

        else:
            sock.sendall(bytes([STATUS_BAD_OP]))

    except (ConnectionError, OSError) as e:
        if verbose:
            print(f"[client {addr}] dropped: {e}")
    finally:
        try:
            sock.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--bind", default="0.0.0.0",
                    help="Bind address. For TB-net, set to the thunderbolt0 IP "
                         "(or 0.0.0.0 to listen on all interfaces).")
    ap.add_argument("--port", type=int, default=29800)
    ap.add_argument("--max-gb", type=float, default=64.0,
                    help="Max total tensor bytes held in memory (default 64 GB).")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "bytes"],
                    help="Storage backend. 'auto' picks MLX on Apple Silicon, "
                         "'bytes' for raw-bytes (works everywhere).")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    backend = make_backend(args.backend, max_bytes=int(args.max_gb * (1 << 30)))
    print(f"[startup] backend={backend.info().get('backend')}  "
          f"mlx_available={_MLX_OK}  max={args.max_gb} GB")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 << 20)
    except OSError:
        pass

    sock.bind((args.bind, args.port))
    sock.listen(64)
    print(f"tb_bridge_server listening on {args.bind}:{args.port}")
    sys.stdout.flush()

    try:
        while True:
            client, addr = sock.accept()
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t = threading.Thread(target=handle_client,
                                 args=(client, addr, backend, args.verbose),
                                 daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
