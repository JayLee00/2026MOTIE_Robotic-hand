#!/usr/bin/env python3
"""make_demo_video.py — 녹화(h5 + 미디어)에서 데모 영상을 만든다.

레이아웃 (live_viz 와 동일 구성):
  ┌───────────────┬───────────────┐
  │ RealSense RGB │ PaXini 촉각 3D │   ← FP 버전은 inhand 구간에만 과일 3D bbox+축
  ├───────────────┴───────────────┤
  │ Hand FT 스트립차트 (4손가락)     │
  └────────────────────────────────┘

두 버전을 한 번에 만든다:
  demoN_base.mp4 : FoundationPose 없음
  demoN_fp.mp4   : inhand(seq 2 RUNNING) 구간에만 — 구간 첫 프레임에서 SAM2 를
                   화면 중앙 클릭 1회로 시딩해 과일 seg → FoundationPose 등록/추적
                   → 3D bounding box + 오리엔테이션 축 표시

사용 (호스트 /usr/bin/python3, `rs` 환경 불필요 — ROS 안 씀):
  cd ~/prime/Jaesung_Lee/RobotAgentSystem
  /usr/bin/python3 Video/make_demo_video.py                 # 두 데모 × 두 버전 전부
  /usr/bin/python3 Video/make_demo_video.py --demo 0        # Demo_0 만
  /usr/bin/python3 Video/make_demo_video.py --no-fp         # base 버전만 (빠름)
  /usr/bin/python3 Video/make_demo_video.py --mesh peach    # FP 과일 CAD 지정

FoundationPose 추론은 conda env `foundationpose` 의 fp_offline_worker.py 가
subprocess 로 수행한다 (GPU0). SAM2 시딩은 호스트 python (GPU1).
"""
import argparse
import faulthandler
import os
import subprocess
import sys
import time
from pathlib import Path

faulthandler.enable()                 # 네이티브 크래시(segfault) 도 트레이스 출력

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEF_H5 = ROOT / "logs/h5_all/exp_20260817_024901_final_ver_3_success.h5"
DEF_MEDIA = ROOT / "logs/h5_all/exp_20260817_024901_media_final_ver_3_success"
FP_DIR = ROOT / ("skill-set/in-hand-reorientation/kist_deploy_pkg/2_fruit_pose/"
                 "foundation_pose")
VIZ_DIR = ROOT / ("skill-set/in-hand-reorientation/kist_deploy_pkg/3_visualization/"
                  "Visualization")
SAM2_CKPT = ROOT / "skill-set/in-hand-reorientation/kist_deploy_pkg/2_fruit_pose/sam2/sam2.1_hiera_tiny.pt"
CONDA_FP_PY = Path.home() / "miniconda3/envs/foundationpose/bin/python"

# ── 레이아웃 ──
PANE_W, PANE_H = 640, 480          # 좌 RGB / 우 촉각
STRIP_H = 300                       # 하단 FT
CANVAS_W, CANVAS_H = PANE_W * 2, PANE_H + STRIP_H
TRAIL_S = 6.0                       # FT 표시 구간 [s]
SEED_OFFSET_FRAMES = 15             # inhand 시작 후 이 프레임 뒤에 SAM2 시딩 (~0.5s)
MIN_SEG_FRAMES = 30                 # 이보다 짧은 inhand 조각은 무시

BG = (24, 26, 28)
FT_COLORS = [(200, 224, 94), (76, 162, 244), (216, 107, 200)]   # BGR: Tx, Ty, Fz
FT_LABEL = ["Tx", "Ty", "Fz"]
FINGERS = ["index", "middle", "ring", "thumb"]
BOX_COLOR = (80, 220, 80)


# ───────────────────────────────────────────────────────────────────────────
def load_demo(h5, media, demo):
    d = h5[f"Demo_{demo}"]
    t = d["90_real_time_demo"][:]
    seq = d["64_seq_shm_state"][:, :2]
    kin = d["07_R_hand_j_kin"][:]                    # (T,12) = 4손가락 × (Fz,Tx,Ty)
    hand_tar = d["04_R_hand_j_tar"][:]               # (T,16) — 정책 engage 감지용
    pax = d["12_R_paxini_raw"][:]                    # (T,1524) = 4×127×3
    rgb_t = d["71_rgb_time"][:]
    n_f = len(rgb_t)
    rgb_paths = [str(media / f"Demo_{demo}" / "rgb" / f"{i:06d}.jpg") for i in range(n_f)]
    depth_paths = [str(media / f"Demo_{demo}" / "depth" / f"{i:06d}.png") for i in range(n_f)]
    # 프레임 → 100Hz 틱 매핑 — ⚠ 71_rgb_time 은 global(perf) 클록이다.
    # demo-상대 클록(90_real_time_demo)이 아니라 91_real_time_global 에 매핑할 것.
    g = d["91_real_time_global"][:]
    tick_of = np.clip(np.searchsorted(g, rgb_t), 0, len(t) - 1)
    inhand_tick = (seq[:, 0] == 2) & (seq[:, 1] == 1)
    inhand_f = inhand_tick[tick_of]
    fps = 1.0 / max(1e-3, float(np.median(np.diff(rgb_t))))
    return dict(t=t, kin=kin, hand_tar=hand_tar, pax=pax, rgb_t=rgb_t, tick_of=tick_of,
                rgb_paths=rgb_paths, depth_paths=depth_paths,
                inhand_f=inhand_f, fps=min(60.0, fps))


def segments_of(mask):
    """bool 배열 → [(start, end)) 연속 구간 목록 (짧은 조각 제거)."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    segs = [(int(s), int(e)) for s, e in zip(idx[::2], idx[1::2])]
    return [(s, e) for s, e in segs if e - s >= MIN_SEG_FRAMES]


def seed_frame_for(data, s, e):
    """SAM2 시딩 프레임 선택 — 정책 engage(손 타겟이 100Hz 로 지속 갱신 시작) + 2초.

    구간 시작 직후는 팔이 제시 자세로 이동 중이라 과일이 화면 중앙에 없다
    (demo1 실측: +0.5s 시딩 → 테이블을 seg). engage 시점엔 팔이 반드시 정착해 있고
    과일이 손에 쥔 채 중앙 부근이다. 감지 실패 시 구간 시작 +10s 폴백."""
    tick_s = int(data["tick_of"][s])
    tick_e = int(data["tick_of"][min(e - 1, len(data["tick_of"]) - 1)])
    tar = data["hand_tar"][tick_s:tick_e]
    moving = (np.abs(np.diff(tar, axis=0)).sum(axis=1) > 0).astype(int)
    win = 100                                        # 1초 창에서 80% 이상 갱신 = 정책 구동
    k = None
    if len(moving) > win:
        csum = np.convolve(moving, np.ones(win, int), "valid")
        idx = np.flatnonzero(csum >= 80)
        if len(idx):
            target_tick = tick_s + int(idx[0]) + win + 200      # engage + 2s
            k = int(np.searchsorted(data["tick_of"], target_tick))
    if k is None:
        k = s + 300                                  # 폴백: +10s
    return int(min(max(k, s), e - 1))


# ── SAM2: 정책 engage+2s 프레임 중앙 클릭 1회 시딩 ──────────────────────────
def sam2_masks(data, segs):
    rgb_paths = data["rgb_paths"]
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    print(f"[sam2] 로드 ({dev}) ...", flush=True)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(SAM2_CKPT), device=dev)
    pred = SAM2ImagePredictor(model)
    masks = []
    for s, e in segs:
        k = seed_frame_for(data, s, e)
        bgr = cv2.imread(rgb_paths[k], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        pred.set_image(rgb)
        m, sc, _ = pred.predict(point_coords=np.array([[w // 2, h // 2]], dtype=np.float32),
                                point_labels=np.array([1]), multimask_output=True)
        best = m[int(np.argmax(sc))].astype(bool)
        print(f"[sam2] 구간 {s}..{e}: seed frame {k}, mask {int(best.sum())}px "
              f"(score {sc.max():.2f})", flush=True)
        masks.append((k, best))
    return masks


# ── FoundationPose 오프라인 (conda worker) ─────────────────────────────────
def run_fp(demo, data, segs, seg_meshes, K, tmp: Path):
    out = tmp / f"fp_poses_demo{demo}.npz"
    if out.is_file():                                   # 캐시 재사용 (합성만 다시 할 때)
        print(f"[fp] 캐시 재사용: {out}", flush=True)
        z = np.load(out)
        return z["poses"], z["valid"], z["mesh_bounds"], z["seg_bounds"]
    seeds = sam2_masks(data, segs)
    # 등록 프레임을 구간 시작으로 쓰되, 시드 프레임부터 추적하도록 구간을 시드에 맞춘다
    seg_bounds = [(k, e) for (s, e), (k, m) in zip(segs, seeds)]
    job = tmp / f"fp_job_demo{demo}.npz"
    np.savez_compressed(
        job, mesh_paths=np.array([str(p) for p in seg_meshes], dtype=object), K=K,
        rgb_paths=np.array(data["rgb_paths"], dtype=object),
        depth_paths=np.array(data["depth_paths"], dtype=object),
        seg_bounds=np.array(seg_bounds, dtype=int),
        masks=np.stack([m for _, m in seeds]))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    print(f"[fp] worker 실행 (구간 {len(seg_bounds)}개) ...", flush=True)
    r = subprocess.run([str(CONDA_FP_PY), str(HERE / "fp_offline_worker.py"),
                        "--job", str(job), "--out", str(out)], env=env)
    if r.returncode != 0 or not out.is_file():
        raise RuntimeError(f"fp_offline_worker 실패 (rc={r.returncode})")
    z = np.load(out)
    return z["poses"], z["valid"], z["mesh_bounds"], z["seg_bounds"]


# ── 그리기 ──────────────────────────────────────────────────────────────────
def draw_box(img, T, K, bounds):
    """메시 AABB 를 자세 T 로 변환해 3D bbox + XYZ 축을 그린다."""
    (x0, y0, z0), (x1, y1, z1) = bounds
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    pts = (T[:3, :3] @ corners.T).T + T[:3, 3]
    uv = (K @ pts.T).T
    ok = uv[:, 2] > 1e-3
    if not ok.all():
        return
    uv = (uv[:, :2] / uv[:, 2:3]).astype(int)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), BOX_COLOR, 2, cv2.LINE_AA)
    # 축 (중심에서 장축 절반 길이)
    c = (np.array([x0, y0, z0]) + np.array([x1, y1, z1])) / 2
    L = float(max(x1 - x0, y1 - y0, z1 - z0)) * 0.7
    axes = [c, c + [L, 0, 0], c + [0, L, 0], c + [0, 0, L]]
    p = (T[:3, :3] @ np.array(axes).T).T + T[:3, 3]
    q = (K @ p.T).T
    q = (q[:, :2] / q[:, 2:3]).astype(int)
    for i, col in ((1, (0, 0, 255)), (2, (0, 255, 0)), (3, (255, 0, 0))):   # X빨 Y초 Z파
        cv2.arrowedLine(img, tuple(q[0]), tuple(q[i]), col, 3, cv2.LINE_AA, tipLength=0.2)


def draw_strip(canvas, kin_win, t_win, t_now):
    """하단 FT 스트립차트: 손가락 4칸 × (Tx, Ty, Fz)."""
    y0 = PANE_H
    cv2.rectangle(canvas, (0, y0), (CANVAS_W, CANVAS_H), BG, -1)
    pad = 8
    col_w = (CANVAS_W - pad * 5) // 4
    for f in range(4):
        x0 = pad + f * (col_w + pad)
        cv2.rectangle(canvas, (x0, y0 + 28), (x0 + col_w, CANVAS_H - pad),
                      (40, 44, 46), -1)
        cv2.putText(canvas, f"{FINGERS[f]}  FT", (x0 + 4, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if len(t_win) < 2:
            continue
        # kin 채널 순서 (Fz,Tx,Ty) → 표시 (Tx,Ty,Fz)
        chan = [kin_win[:, f * 3 + 1], kin_win[:, f * 3 + 2], kin_win[:, f * 3 + 0]]
        allv = np.concatenate(chan)
        lo, hi = float(allv.min()), float(allv.max())
        rng = max(1e-6, hi - lo)
        h = CANVAS_H - pad - (y0 + 28)
        xs = ((t_win - (t_now - TRAIL_S)) / TRAIL_S * (col_w - 2)).astype(int) + x0 + 1
        for ci, v in enumerate(chan):
            ys = (y0 + 28 + h - 4 - ((v - lo) / rng * (h - 8))).astype(int)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, FT_COLORS[ci], 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{FT_LABEL[ci]} {v[-1]:+.0f}",
                        (x0 + 4 + ci * 100, CANVAS_H - pad - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, FT_COLORS[ci], 1, cv2.LINE_AA)


TACTILE_STEP = 3          # 촉각 렌더 간격 (프레임) ≈ 10Hz
TACTILE_CHUNK = 3600      # 워커 1회당 처리 프레임 수 (크래시 격리 단위)


def tactile_worker(h5_path, media, demo, start, end, pane_dir: Path):
    """서브프로세스: [start,end) 구간의 촉각 패널을 jpg 로 렌더 (open3d 크래시 격리)."""
    sys.path.insert(0, str(VIZ_DIR))
    from tactile_render import Scene3D                        # noqa: E402
    scene = Scene3D(width=PANE_W, height=PANE_H)
    with h5py.File(h5_path, "r") as h5:
        data = load_demo(h5, Path(media), demo)
    for i in range(start, end, TACTILE_STEP):
        out = pane_dir / f"pane_{i:06d}.jpg"
        if out.is_file():
            continue
        tac = data["pax"][data["tick_of"][i]].reshape(4, 127, 3)
        cv2.imwrite(str(out), scene.render(tac), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[tactile-worker] demo{demo} {start}..{end} 완료", flush=True)


def render_tactile_panes(h5_path, media, demo, n_frames, pane_dir: Path):
    """촉각 패널 전체를 청크 서브프로세스로 렌더 — 세그폴트 나면 그 청크만 재시도."""
    pane_dir.mkdir(parents=True, exist_ok=True)
    need = list(range(0, n_frames, TACTILE_STEP))
    for attempt in range(6):
        missing = [i for i in need if not (pane_dir / f"pane_{i:06d}.jpg").is_file()]
        if not missing:
            return
        s = missing[0]
        e = min(n_frames, s + TACTILE_CHUNK)
        print(f"[tactile] 렌더 {s}..{e} (남은 {len(missing)}장, 시도 {attempt + 1})",
              flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--tactile-worker", str(demo), str(s), str(e),
                        "--h5", str(h5_path), "--media", str(media),
                        "--out", str(pane_dir)])
    missing = [i for i in need if not (pane_dir / f"pane_{i:06d}.jpg").is_file()]
    if missing:
        raise RuntimeError(f"촉각 렌더 미완 {len(missing)}장 (예: {missing[:3]})")


def make_videos(demo, data, K, fp_result, out_dir: Path, no_fp: bool, pane_dir: Path):
    fps = data["fps"]
    names = [out_dir / f"demo{demo}_base.mp4"]
    if not no_fp:
        names.append(out_dir / f"demo{demo}_fp.mp4")
    writers = [cv2.VideoWriter(str(n), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                               (CANVAS_W, CANVAS_H)) for n in names]
    poses = valid = bounds = seg_b = None
    if fp_result is not None:
        poses, valid, bounds, seg_b = fp_result

    n = len(data["rgb_paths"])
    t0 = time.perf_counter()
    pane_t = np.full((PANE_H, PANE_W, 3), BG, np.uint8)
    for i in range(n):
        bgr = cv2.imread(data["rgb_paths"][i], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        tick = data["tick_of"][i]
        # 우: PaXini 촉각 3D — 사전 렌더된 패널 사용 (open3d 는 서브프로세스에서만)
        if i % TACTILE_STEP == 0:
            p = cv2.imread(str(pane_dir / f"pane_{i:06d}.jpg"), cv2.IMREAD_COLOR)
            if p is not None:
                pane_t = p if p.shape[:2] == (PANE_H, PANE_W) else cv2.resize(
                    p, (PANE_W, PANE_H))
        # 하단 FT 윈도우
        t_now = data["t"][tick]
        w0 = np.searchsorted(data["t"], t_now - TRAIL_S)
        kin_win = data["kin"][w0:tick + 1]
        t_win = data["t"][w0:tick + 1]

        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
        canvas[:PANE_H, PANE_W:] = pane_t
        draw_strip(canvas, kin_win, t_win, t_now)

        def put_left(img):
            canvas[:PANE_H, :PANE_W] = cv2.resize(img, (PANE_W, PANE_H))
            if data["inhand_f"][i]:
                cv2.putText(canvas, "IN-HAND MANIPULATION", (12, PANE_H - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 220, 255), 2, cv2.LINE_AA)

        # base 버전
        put_left(bgr)
        writers[0].write(canvas)
        # fp 버전 (inhand 구간 + 유효 자세일 때만 bbox)
        if len(writers) > 1:
            left = bgr.copy()
            if valid is not None and valid[i]:
                si = int(np.argmax((seg_b[:, 0] <= i) & (i < seg_b[:, 1])))
                draw_box(left, poses[i], K, bounds[si])
            put_left(left)
            writers[1].write(canvas)
        if i % 1000 == 0:
            dt = time.perf_counter() - t0
            print(f"[compose] demo{demo}: {i}/{n} ({dt:.0f}s)", flush=True)
    for w in writers:
        w.release()
    for nm in names:
        print(f"[compose] 저장: {nm} ({nm.stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(DEF_H5))
    ap.add_argument("--media", default=str(DEF_MEDIA))
    ap.add_argument("--demo", type=int, default=None, help="0/1, 생략=전부")
    ap.add_argument("--meshes", default="lemon,mandarin,peach",
                    help="inhand 구간별 FP 과일 CAD (쉼표, 부족하면 순환). "
                         "이번 데모 = lemon,tomato,peach 인데 tomato CAD 가 없어 "
                         "mandarin 을 대용으로 기본 지정")
    ap.add_argument("--no-fp", action="store_true", help="base 버전만")
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--tactile-worker", nargs=3, type=int, metavar=("DEMO", "S", "E"),
                    help=argparse.SUPPRESS)     # 내부용: 촉각 렌더 서브프로세스
    a = ap.parse_args()

    if a.tactile_worker:
        demo, s, e = a.tactile_worker
        tactile_worker(a.h5, a.media, demo, s, e, Path(a.out))
        return

    def mesh_path(name):
        p = Path(name)
        if not p.is_file():
            p = FP_DIR / "assets" / name / f"{name}.obj"
        if not p.is_file():
            sys.exit(f"메시 없음: {name} ({p})")
        return p
    mesh_names = [x.strip() for x in a.meshes.split(",") if x.strip()]
    meshes = [mesh_path(x) for x in mesh_names] if not a.no_fp else []

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tmp"
    tmp.mkdir(exist_ok=True)

    with h5py.File(a.h5, "r") as h5:
        K = np.array([[h5.attrs["rgb_fx"], 0, h5.attrs["rgb_cx"]],
                      [0, h5.attrs["rgb_fy"], h5.attrs["rgb_cy"]],
                      [0, 0, 1.0]])
        demos = [a.demo] if a.demo is not None else list(range(int(h5.attrs["n_demos"])))
        for demo in demos:
            print(f"═══ Demo_{demo} ═══", flush=True)
            data = load_demo(h5, Path(a.media), demo)
            segs = segments_of(data["inhand_f"])
            print(f"inhand 구간: {segs} (프레임)", flush=True)
            pane_dir = tmp / f"panes_demo{demo}"
            render_tactile_panes(a.h5, a.media, demo, len(data["rgb_paths"]), pane_dir)
            fp_result = None
            if not a.no_fp and segs:
                seg_meshes = [meshes[i % len(meshes)] for i in range(len(segs))]
                print("구간별 메시:", [p.stem for p in seg_meshes], flush=True)
                fp_result = run_fp(demo, data, segs, seg_meshes, K, tmp)
            make_videos(demo, data, K, fp_result, out_dir, a.no_fp, pane_dir)


if __name__ == "__main__":
    main()
