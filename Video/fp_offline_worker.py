#!/usr/bin/env python3
"""fp_offline_worker.py — 녹화 데이터에 FoundationPose 를 오프라인으로 돌려 자세를 뽑는다.

conda env `foundationpose` 의 python 으로 실행해야 한다 (make_demo_video.py 가
subprocess 로 부른다 — 직접 실행할 일은 없음):

  CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/foundationpose/bin/python \
      fp_offline_worker.py --job job.npz --out poses.npz

job.npz:
  mesh_paths  : (S,) 구간별 과일 CAD 경로
  K           : (3,3) 카메라 내참
  rgb_paths   : (N,) 프레임 jpg 경로 (전 구간 연결)
  depth_paths : (N,) 정렬 depth png 경로 (16UC1, mm)
  seg_bounds  : (S,2) 각 inhand 구간의 [시작, 끝) 인덱스 (rgb_paths 기준)
  masks       : (S,H,W) bool — 구간별 SAM2 초기 마스크 (구간 첫 프레임에서 등록)

poses.npz:
  poses (N,4,4) float64 · valid (N,) bool · mesh_bounds (2,3) — trimesh AABB
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(HERE, "..", "skill-set", "in-hand-reorientation",
                      "kist_deploy_pkg", "2_fruit_pose", "foundation_pose")
FP_DIR = os.path.normpath(FP_DIR)
FP_ROOT = os.path.join(FP_DIR, "FoundationPose")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    job = np.load(a.job, allow_pickle=True)
    mesh_paths = [str(p) for p in job["mesh_paths"]]
    K = np.asarray(job["K"], dtype=np.float64)
    rgb_paths = [str(p) for p in job["rgb_paths"]]
    depth_paths = [str(p) for p in job["depth_paths"]]
    seg_bounds = np.asarray(job["seg_bounds"], dtype=int)
    masks = np.asarray(job["masks"]).astype(bool)

    # fp_server 와 동일한 임포트 준비 (estimater 는 FP_ROOT 기준 상대경로 가중치를 쓴다)
    sys.path.insert(0, FP_ROOT)
    sys.path.insert(0, FP_DIR)
    os.chdir(FP_ROOT)
    import fp_server                                   # noqa: E402

    print(f"[worker] 모델 로드 (mesh={os.path.basename(mesh_paths[0])}) ...", flush=True)
    eng = fp_server.Engine(mesh_paths[0], est_iter=5, track_iter=2, debug_dir="/tmp/fp_offline")
    import trimesh
    bounds = np.stack([trimesh.load(p, force="mesh").bounds for p in mesh_paths])  # (S,2,3)

    n = len(rgb_paths)
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    for si, (s, e) in enumerate(seg_bounds):
        print(f"[worker] 구간 {si + 1}/{len(seg_bounds)}: 프레임 {s}..{e - 1} "
              f"(mesh={os.path.basename(mesh_paths[si])})", flush=True)
        if si > 0 and mesh_paths[si] != mesh_paths[si - 1]:
            eng.handle({"cmd": "set_mesh", "mesh": mesh_paths[si]})
        t0 = time.perf_counter()
        for i in range(s, e):
            bgr = cv2.imread(rgb_paths[i], cv2.IMREAD_COLOR)
            dep = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED)
            if bgr is None or dep is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_m = dep.astype(np.float32) / 1000.0          # mm → m
            try:
                if i == s:                                     # 구간 첫 프레임 = 등록
                    rep = eng.handle({"cmd": "register", "rgb": rgb, "depth": depth_m,
                                      "K": K, "mask": masks[si]})
                else:
                    rep = eng.handle({"cmd": "track", "rgb": rgb, "depth": depth_m,
                                      "K": K})
            except Exception as ex:                            # noqa: BLE001
                print(f"[worker]   frame {i} 실패: {ex}", flush=True)
                eng.registered = False                          # 다음 프레임 재등록 유도
                continue
            if rep.get("ok") and rep.get("pose") is not None:
                poses[i] = np.asarray(rep["pose"], dtype=np.float64)
                valid[i] = True
        # 구간 종료 → 다음 구간은 새로 등록
        eng.registered = False
        dt = time.perf_counter() - t0
        nf = e - s
        print(f"[worker]   완료: {nf}프레임 / {dt:.1f}s ({dt / max(1, nf) * 1e3:.0f}ms/f), "
              f"valid {int(valid[s:e].sum())}", flush=True)

    np.savez_compressed(a.out, poses=poses, valid=valid, mesh_bounds=bounds,
                        seg_bounds=seg_bounds)
    print(f"[worker] 저장: {a.out}", flush=True)


if __name__ == "__main__":
    main()
