#!/usr/bin/env python3
"""FoundationPose 추론 서버 — **컨테이너 안에서** 실행된다.

왜 서버로 쪼갰나: FoundationPose 는 nvdiffrast/PyTorch3D 를 nvcc 로 소스 빌드해야 하는데
이 PC 에는 CUDA 툴킷이 없다. 호스트에 CUDA 를 새로 깔면 이미 잘 돌고 있는
torch 2.6+cu124 / SAM2 환경을 건드리게 된다. 그래서 추론만 공식 도커 이미지 안에
가두고, ROS2·SAM2 는 호스트에 그대로 둔 뒤 TCP 로 잇는다.

프로토콜: 길이(8B, little-endian) + pickle 페이로드. 양쪽 다 파이썬이라 이걸로 충분하다.
    요청  {"cmd": "register"|"track"|"reset"|"ping",
           "rgb": HxWx3 uint8, "depth": HxW float32 [m], "K": 3x3 float64,
           "mask": HxW bool (register 에만)}
    응답  {"ok": bool, "pose": 4x4 float64 | None, "ms": float, "err": str|None}

실행(컨테이너 내부):
    python3 /workspace/fp/fp_server.py --mesh /workspace/fp/assets/orange.obj --port 5577
"""
from __future__ import annotations

import argparse
import os
import pickle
import socket
import struct
import sys
import time
import traceback

import numpy as np

HDR = struct.Struct("<Q")


def recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket):
    head = recv_exact(sock, HDR.size)
    if head is None:
        return None
    (n,) = HDR.unpack(head)
    body = recv_exact(sock, n)
    return None if body is None else pickle.loads(body)


def send_msg(sock: socket.socket, obj) -> None:
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(HDR.pack(len(body)) + body)


class Engine:
    """FoundationPose 를 감싸 register/track 상태를 들고 있는다."""

    def __init__(self, mesh_file: str, est_iter: int, track_iter: int, debug_dir: str):
        # FoundationPose 저장소 코드는 sys.path 에 /workspace/FoundationPose 가 있어야 import 된다.
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

        self.est_iter = est_iter
        self.track_iter = track_iter

        self.mesh = trimesh.load(mesh_file, force="mesh")
        print(f"[fp_server] mesh: {mesh_file}  v={len(self.mesh.vertices)} "
              f"f={len(self.mesh.faces)}  extents={self.mesh.extents}", flush=True)

        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=ScorePredictor(),
            refiner=PoseRefinePredictor(),
            debug_dir=debug_dir,
            debug=0,
            glctx=dr.RasterizeCudaContext(),
        )
        self.registered = False
        print("[fp_server] FoundationPose 준비 완료", flush=True)

    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        t0 = time.perf_counter()

        if cmd == "ping":
            # extents 를 같이 준다 — 노드가 /fruit/size 를 메시 실제 크기로 발행하도록
            return {"ok": True, "pose": None, "ms": 0.0, "err": None,
                    "registered": self.registered,
                    "extents": [float(x) for x in self.mesh.extents]}

        if cmd == "reset":
            # 다음 프레임에서 마스크를 받아 다시 register 한다
            self.registered = False
            return {"ok": True, "pose": None, "ms": 0.0, "err": None}

        if cmd == "set_mesh":
            # 물체가 바뀌면 메시도 바뀌어야 한다(model-based 라서).
            # reset_object 는 지름·복셀크기·회전격자를 새 메시로 다시 만든다.
            import trimesh
            path = req["mesh"]
            if not os.path.isfile(path):
                return {"ok": False, "pose": None, "ms": 0.0,
                        "err": f"메시 파일 없음: {path}"}
            m = trimesh.load(path, force="mesh")
            self.est.reset_object(model_pts=m.vertices, model_normals=m.vertex_normals,
                                  mesh=m)
            self.mesh = m
            self.registered = False
            print(f"[fp_server] 메시 교체: {path} (v={len(m.vertices)}, "
                  f"extents={m.extents})", flush=True)
            return {"ok": True, "pose": None, "ms": (time.perf_counter() - t0) * 1e3,
                    "err": None, "extents": [float(x) for x in m.extents]}

        rgb = req["rgb"]
        depth = np.ascontiguousarray(req["depth"], dtype=np.float32)
        K = np.ascontiguousarray(req["K"], dtype=np.float64)

        if cmd == "register" or not self.registered:
            mask = req.get("mask")
            if mask is None:
                return {"ok": False, "pose": None, "ms": 0.0,
                        "err": "register 에는 mask 가 필요합니다"}
            pose = self.est.register(K=K, rgb=rgb, depth=depth,
                                     ob_mask=np.asarray(mask).astype(bool),
                                     iteration=self.est_iter)
            self.registered = True
        elif cmd == "track":
            pose = self.est.track_one(rgb=rgb, depth=depth, K=K,
                                      iteration=self.track_iter)
        else:
            return {"ok": False, "pose": None, "ms": 0.0, "err": f"알 수 없는 cmd: {cmd}"}

        return {"ok": True, "pose": np.asarray(pose, dtype=np.float64),
                "ms": (time.perf_counter() - t0) * 1e3, "err": None}


def main():
    ap = argparse.ArgumentParser(description="FoundationPose TCP 추론 서버 (컨테이너용)")
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5577)
    ap.add_argument("--est-refine-iter", type=int, default=5)
    ap.add_argument("--track-refine-iter", type=int, default=2)
    ap.add_argument("--debug-dir", default="/tmp/fp_debug")
    ap.add_argument("--fp-root", default="/workspace/FoundationPose",
                    help="FoundationPose 저장소 경로 (import 용)")
    a = ap.parse_args()

    if os.path.isdir(a.fp_root):
        sys.path.insert(0, a.fp_root)
        os.chdir(a.fp_root)          # 저장소가 상대경로로 가중치를 찾는다
    os.makedirs(a.debug_dir, exist_ok=True)

    engine = Engine(a.mesh, a.est_refine_iter, a.track_refine_iter, a.debug_dir)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.host, a.port))
    srv.listen(1)
    print(f"[fp_server] listening on {a.host}:{a.port}", flush=True)

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[fp_server] 접속: {addr}", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                try:
                    rep = engine.handle(req)
                except Exception as e:                       # noqa: BLE001
                    traceback.print_exc()
                    rep = {"ok": False, "pose": None, "ms": 0.0, "err": repr(e)}
                send_msg(conn, rep)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            conn.close()
            print("[fp_server] 접속 종료 — 다음 클라이언트 대기", flush=True)


if __name__ == "__main__":
    main()
