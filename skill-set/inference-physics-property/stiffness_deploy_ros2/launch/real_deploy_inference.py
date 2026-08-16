"""
real_deploy_inference.py  (실시간 강성 추론 엔진)
=================================================
★ 구조 변경: squeeze 플래그 파일/폴링 제거.
  데이터수집 때는 motion 과 logger 가 '분리된 프로세스'라 squeeze 신호를
  파일(/tmp/gen3_squeeze_on.txt)로 주고받았다. 하지만 deploy 에서는 추론을
  motion_sequence 코드 '안에서' 호출하므로, motion 이 이미 squeeze 상태를
  변수로 알고 있다. 따라서 파일 불필요 -> motion 이 직접 엔진 메서드를 호출.

사용법 (motion_sequence_A_self.py 안에서):
    from real_deploy_inference import StiffnessInferenceEngine
    engine = StiffnessInferenceEngine(model_path=..., fruit="tomato")
    ...
    # 스퀴즈+hold 루프 안에서 매 제어주기마다:
    engine.add_sample(shm, paxini)        # 현재 핸드/paxini 샘플 적재
    ...
    # 스퀴즈 끝난 직후 (파지 복귀 전):
    stiffness, cls, cname = engine.infer()  # 절대강성 + 등급 반환
    engine.reset()                          # 다음 데모 위해 버퍼 비움

데이터 소스 (학습과 동일하게 둘 다):
  핸드 SHM 0x3931: 03_joint(j_pos), 06_FT(j_kin)
  paxini SHM 0x3934: 17_tactile -> 18_resultant(=nan_to_num합) , 20_valid
전처리/마스킹/정규화/모델 전부 학습(data_preprocessing.py)과 1:1 동일.
"""
import sys, os, math, time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.shm_common import ShmAccess, SHM_MSG_KEY, Hand_DOF, Kinesthetic_Sensor_Num, Kinesthetic_Sensor_DOF
from core.paxini_shm import PaxiniShmReader, PAXINI_SHM_KEY
from model import StiffnessRegressor, MODEL_REGISTRY

# ===================== CONFIG (학습과 반드시 일치) =====================
# 모델(.pth)/라벨(yaml)은 이 레포에 번들되어 있다(절대경로 의존 제거 → 어느 머신에서든 동작).
#   stiffness_deploy_ros2/models/  ·  stiffness_deploy_ros2/labels/
_PKG_DIR    = Path(__file__).resolve().parent.parent   # stiffness_deploy_ros2/
_MODELS_DIR = _PKG_DIR / "models"
LABEL_DIR   = str(_PKG_DIR / "labels")
# 시연 모델: train.py 가 저장한 체크포인트 (.pth). 레포 내 models/ 에 번들됨.
MODEL_PATH  = str(_MODELS_DIR / "260630_1006_transformer_fruit_A_lr0.0007_h64_L3_do0.3_schCOSINE_s64.pth")

# 채널 플래그 — data_preprocessing.py 와 똑같이!
USE_JOINT     = True
# ★ joint 변위(Δjoint) 사용: 학습(data_preprocessing)과 반드시 일치.
#   buffer 첫 프레임(=스퀴즈 시작) 대비 변위로 변환 -> 파지 자세 불변.
#   ⚠ 현재 챔피언 모델(260630_1006)은 '절대 joint'로 학습됨 -> False 유지.
#   Δjoint 로 '재학습한 새 모델'로 교체할 때만 True 로 (그 땐 force_zero 도 불필요).
USE_JOINT_DELTA = False
USE_JKIN      = True
USE_RESULTANT = True
USE_TACTILE   = False
TACTILE_SUMMARY = True

JOINT_SCALE = math.pi / 4096.0
FACTOR = 10
MIN_LEN = FACTOR

# --- 파지실패 손가락 마스킹 ---
# 기본: 체크포인트의 mask_grip_fail 값을 자동으로 따름.
# 수동강제: 체크포인트 값이 부정확하다고 판단되면 아래를 True/False 로 지정.
#   MASK_OVERRIDE = None  -> 자동 (체크포인트 값 사용)
#   MASK_OVERRIDE = True  -> 강제로 마스킹 ON  (학습데이터가 마스킹됐을 때)
#   MASK_OVERRIDE = False -> 강제로 마스킹 OFF (학습데이터가 마스킹 안 됐을 때)
MASK_OVERRIDE    = None
MASK_GRIP_FAIL   = True    # (자동 모드의 기본값. 보통 체크포인트가 덮어씀)
GRIP_FAIL_THRESH = 1.4    # ★ 학습 데이터셋과 반드시 일치시킬 것
# ★ 플랜A: 학습분포를 크게 벗어난(파지 성공이라 마스킹 안 되는) 특정 관절을
#   정규화 후 0(=학습평균)으로 강제. std 극소 관절(예: ch8=finger2 joint0)이
#   deploy 자세 미세차로 z 폭발 -> 모델이 그 채널만 보고 쏠리는 것을 방지.
#   이 관절들은 학습 때 거의 안 변해(정보 거의 없음) 0 처리해도 추론 영향 미미.
FORCE_ZERO_CHANNELS = [8]      # 정규화 후 0 으로 만들 채널 인덱스 (빈 리스트면 비활성)
CLAMP_Z = 5.0                  # |z|>CLAMP_Z 인 값은 ±CLAMP_Z 로 클램프 (0이면 비활성). 안전망.
DEBUG = False                  # ★ 시연=False(깔끔). 디버깅=True(센서/입력 통계 출력)
THUMB_FINGER     = 0
SUPPORT_FINGERS  = (1, 2, 3)
NORMAL_AXIS      = 2
JOINTS_PER_FINGER = 4
FRUIT_BY_NUM = {1: "plum", 2: "kiwi", 3: "tomato", 4: "lemon"}

# ===================== 과일별 설정 =====================
# 과일 번호 선택 시 모델/포즈/채널처리가 한꺼번에 정해진다.
#   model : 그 과일 학습 체크포인트(.pth). None 이면 "아직 준비 안 됨".
#   pose  : 파지 자세 txt 파일명 (deploy_motion_sequence 가 이걸 읽어 파지).
#   force_zero : 그 과일에서 정규화 후 0 으로 만들 채널(플랜A). 과일마다 다를 수 있음.
# ★ 새 과일을 추가하려면: 학습 .pth 경로 + 포즈 txt 를 채우면 된다.
_RESULTS_DIR = str(_MODELS_DIR)   # 레포 번들 models/ (위 CONFIG 에서 정의)
FRUIT_CONFIG = {
    "tomato": {
        "model": f"{_RESULTS_DIR}/260630_1006_transformer_fruit_A_lr0.0007_h64_L3_do0.3_schCOSINE_s64.pth",
        "pose":  "tomato.txt",
        "force_zero": [8],          # finger2 joint0 (std 극소 -> 폭발) 0 처리
    },
    "plum":  {"model": None, "pose": "plum.txt",  "force_zero": []},
    "kiwi":  {"model": None, "pose": "kiwi.txt",  "force_zero": []},
    "lemon": {"model": None, "pose": "lemon.txt", "force_zero": []},
}
# =====================================================================
# =====================================================================


# ---------- 학습과 동일한 변환들 ----------
def resultant_from_tactile(tac):
    """17 (4,127,3) -> 18 (4,3). 수집코드와 동일: nan_to_num 후 점축 합."""
    return np.nan_to_num(tac, nan=0.0).sum(axis=1)


def downsample_avg(data, factor, offset=0):
    data = data[offset:]
    n = (len(data) // factor) * factor
    if n == 0:
        return data[:0]
    data = data[:n]
    return data.reshape(n // factor, factor, *data.shape[1:]).mean(axis=1)


def tactile_summary(tac):
    p = np.abs(np.nan_to_num(tac[..., 2]))
    total = p.sum(axis=2)
    active = (p > (p.max() + 1e-9) * 0.05).sum(axis=2).astype(float)
    peak = p.max(axis=2)
    idx = np.arange(p.shape[2])[None, None, :]
    w = p / (p.sum(axis=2, keepdims=True) + 1e-9)
    centroid = (w * idx).sum(axis=2)
    spread = np.sqrt((w * (idx - centroid[..., None])**2).sum(axis=2))
    return np.concatenate([total, active, peak, centroid, spread], axis=1)


def detect_failed_fingers(finger_normal_series):
    fmax = np.abs(finger_normal_series).max(axis=0)
    failed = {fi for fi in SUPPORT_FINGERS if fmax[fi] < GRIP_FAIL_THRESH}
    # === DEBUG: 손가락별 최대 법선력 + 마스킹 판정 ===
    if DEBUG:
        print(f"[DEBUG] 손가락별 최대법선력 f0={fmax[0]:.2f} f1={fmax[1]:.2f} "
              f"f2={fmax[2]:.2f} f3={fmax[3]:.2f} | 임계={GRIP_FAIL_THRESH} -> 마스킹대상={sorted(failed)}")
    return failed


def build_sensor(buf, mask_grip):
    """버퍼(리스트들) -> (n,C). 마스킹 후 concat. 학습과 동일 순서."""
    n = len(buf["resultant"])
    arrays = {}
    if USE_JOINT:
        arrays["joint"] = np.array(buf["joint"], dtype=np.float32).reshape(n, -1)
        # ★ Δjoint: buffer 첫 프레임(=스퀴즈 시작) 대비 변위. 학습과 동일.
        if USE_JOINT_DELTA:
            arrays["joint"] = arrays["joint"] - arrays["joint"][0:1, :]
    if USE_JKIN:
        arrays["ft"] = np.array(buf["ft"], dtype=np.float32).reshape(n, 4, 3)
    if USE_RESULTANT:
        arrays["resultant"] = np.array(buf["resultant"], dtype=np.float32).reshape(n, 4, 3)
    if USE_TACTILE:
        arrays["tactile"] = np.nan_to_num(np.array(buf["tactile"], dtype=np.float32), nan=0.0)

    if mask_grip:
        if "resultant" in arrays:
            fn = arrays["resultant"][:, :, NORMAL_AXIS]
        elif "tactile" in arrays:
            fn = arrays["tactile"][:, :, :, NORMAL_AXIS].sum(axis=2)
        elif "ft" in arrays:
            fn = arrays["ft"][:, :, NORMAL_AXIS]
        else:
            fn = None
        if fn is not None:
            for fi in detect_failed_fingers(fn):
                if "joint" in arrays:
                    arrays["joint"][:, fi*JOINTS_PER_FINGER:(fi+1)*JOINTS_PER_FINGER] = 0.0
                if "ft" in arrays:        arrays["ft"][:, fi, :] = 0.0
                if "resultant" in arrays: arrays["resultant"][:, fi, :] = 0.0
                if "tactile" in arrays:   arrays["tactile"][:, fi, :, :] = 0.0

    feats = []
    if "joint" in arrays:     feats.append(arrays["joint"] * JOINT_SCALE)
    if "ft" in arrays:        feats.append(arrays["ft"].reshape(n, -1))
    if "resultant" in arrays: feats.append(arrays["resultant"].reshape(n, -1))
    if "tactile" in arrays:
        feats.append(tactile_summary(arrays["tactile"]) if TACTILE_SUMMARY
                     else arrays["tactile"].reshape(n, -1))
    return np.concatenate(feats, axis=1)


# ★ 학습(hdf5_logger)과 동일: FT(j_kin)는 SHM int16 raw 가 아니라
#   mN side-channel 파일에서 읽은 보정값이다. deploy 도 같은 소스를 써야 일치.
RAW_HAND_J_KIN_FILE = Path("/tmp/deep_ws_raw_06_hand_j_kin_mN.txt")
SIDE_CHANNEL_MAX_AGE_SEC = 1.0
_WARNED = {"ft": False}


def read_raw_hand_j_kin_mN(state_file):
    """수집과 동일: mN side-channel 파일에서 FT(4,3) 읽어 int16 화.
       학습 데이터의 06_hand_j_kin 이 바로 이 값이다."""
    try:
        if time.time() - state_file.stat().st_mtime > SIDE_CHANNEL_MAX_AGE_SEC:
            return None
        text = state_file.read_text(encoding="ascii").strip()
    except OSError:
        return None
    values = np.fromstring(text, sep=" ", dtype=np.float32)
    if values.shape != (12,):
        return None
    values = np.rint(values).clip(np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return values.astype(np.int16).reshape(4, 3)


def read_live_sample(hand_shm, paxini_reader):
    """핸드 SHM + paxini SHM 에서 한 샘플. 엔진이 매 제어주기마다 호출."""
    msg = hand_shm.read()
    joint = np.array([msg.j_pos[0][j] for j in range(Hand_DOF)], dtype=np.float32)
    # ★ FT(j_kin): 학습 데이터는 SHM int16 raw 가 아니라 mN 보정값(파일)을 썼다.
    #    deploy 도 반드시 같은 소스(mN 파일)를 읽어야 스케일이 일치한다.
    ft_mN = read_raw_hand_j_kin_mN(RAW_HAND_J_KIN_FILE)
    if ft_mN is not None:
        ft = ft_mN.reshape(-1).astype(np.float32)        # 학습과 동일 (mN→int16 변환값)
    else:
        # mN 파일이 없으면(side-channel 미실행) SHM raw 로 폴백 — 단 경고
        ft = np.array([[msg.j_kin[0][i][k] for k in range(Kinesthetic_Sensor_DOF)]
                       for i in range(Kinesthetic_Sensor_Num)], dtype=np.float32).reshape(-1)
        if not _WARNED["ft"]:
            print("⚠ [deploy] mN 파일 없음 → SHM raw j_kin 사용(학습과 불일치!). "
                  f"side-channel 실행 필요: {RAW_HAND_J_KIN_FILE}")
            _WARNED["ft"] = True
    tactile, _t, valid, _seq = paxini_reader.read()
    resultant = resultant_from_tactile(tactile).reshape(-1)
    return {"joint": joint, "ft": ft, "tactile": tactile,
            "resultant": resultant, "valid": int(np.array(valid).ravel()[0])}


# ---------- 모델/설정 로딩 ----------
def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval(), None, None, None
    if "state_dict" not in ckpt:
        raise RuntimeError(f"알 수 없는 체크포인트: {model_path}")
    cfg = ckpt.get("model_config", {})
    model_name = ckpt.get("model_name", "lstm")
    model_cls = MODEL_REGISTRY.get(model_name, StiffnessRegressor)
    model = model_cls(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt.get("norm_mean"), ckpt.get("norm_std"), ckpt.get("mask_grip_fail")


def load_fruit_config(label_dir, fruit):
    """과일별 정규화(역변환) + 경계(class) + 클래스명."""
    with open(os.path.join(label_dir, "stiffness.yaml"), encoding="utf-8") as f:
        stf = yaml.safe_load(f)
    with open(os.path.join(label_dir, "class.yaml"), encoding="utf-8") as f:
        clc = yaml.safe_load(f)
    npf = stf.get("normalize_per_fruit", {}).get(fruit)
    if npf is None:
        raise SystemExit(f"과일 '{fruit}' 정규화 파라미터 없음 (stiffness.yaml).")
    boundaries = clc.get("boundaries_per_fruit", {}).get(fruit, [])
    class_names = clc.get("class_names", ["soft", "mid", "hard"])
    return float(npf["min"]), float(npf["max"]), list(boundaries), class_names


# ==================== 추론 엔진 ====================
class StiffnessInferenceEngine:
    """motion_sequence 가 직접 호출하는 추론 엔진.
       squeeze 파일/폴링 없음: motion 이 squeeze 상태를 알고 직접 메서드 호출."""

    def __init__(self, model_path=MODEL_PATH, fruit="tomato", label_dir=LABEL_DIR,
                 device=None, force_zero=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fruit = fruit
        # 과일별 강제0 채널(플랜A). 미지정이면 전역 기본값 사용.
        self.force_zero = list(FORCE_ZERO_CHANNELS if force_zero is None else force_zero)

        self.model, self.mean, self.std, mask_flag = load_model(model_path, self.device)
        if self.mean is None or self.std is None:
            raise SystemExit("체크포인트에 정규화통계 없음. train.py 재저장 필요.")

        # 마스킹 설정 결정 (우선순위: 수동강제 > 체크포인트 > 기본값)
        global MASK_GRIP_FAIL
        if MASK_OVERRIDE is not None:
            MASK_GRIP_FAIL = bool(MASK_OVERRIDE)
            src = "수동강제(MASK_OVERRIDE)"
            if mask_flag is not None and bool(mask_flag) != bool(MASK_OVERRIDE):
                print(f"  ⚠ 마스킹 불일치: 체크포인트={mask_flag} 이지만 "
                      f"수동강제={MASK_OVERRIDE} 사용")
        elif mask_flag is not None:
            MASK_GRIP_FAIL = bool(mask_flag)
            src = "체크포인트 자동"
        else:
            src = "기본값(체크포인트에 정보 없음)"
        self.mask_grip = MASK_GRIP_FAIL
        print(f"  마스킹={MASK_GRIP_FAIL} (출처: {src}), 임계={GRIP_FAIL_THRESH}N")

        self.norm_min, self.norm_max, self.boundaries, self.class_names = \
            load_fruit_config(label_dir, fruit)

        self.reset()
        print(f"[추론엔진] fruit={fruit}, 마스킹={'ON' if self.mask_grip else 'OFF'}, "
              f"정규화[{self.norm_min},{self.norm_max}], 경계={self.boundaries}")

    def reset(self):
        """다음 데모(누름) 위해 버퍼 비움."""
        self.buf = {"joint": [], "ft": [], "resultant": [], "tactile": []}

    def add_sample(self, hand_shm, paxini_reader):
        """스퀴즈+hold 구간에서 매 제어주기마다 호출. 유효 샘플만 적재.
           (학습의 valid AND squeeze 중 squeeze 는 motion 이 호출시점으로 보장,
            valid 는 여기서 체크)"""
        s = read_live_sample(hand_shm, paxini_reader)
        if s["valid"] != 1:
            return False     # paxini 무효 프레임은 스킵 (학습의 valid 필터와 동일)
        self.buf["joint"].append(s["joint"])
        self.buf["ft"].append(s["ft"])
        self.buf["resultant"].append(s["resultant"])
        if USE_TACTILE:
            self.buf["tactile"].append(s["tactile"])
        return True

    def add_sample_arrays(self, joint, ft, tactile, valid):
        """SHM 객체 대신 이미 읽은 배열로 적재하고 싶을 때 (motion 이 이미 읽었으면).
           joint:(16,) ft:(12,) or (4,3) tactile:(4,127,3) valid:int"""
        if int(valid) != 1:
            return False
        self.buf["joint"].append(np.asarray(joint, np.float32).reshape(-1))
        self.buf["ft"].append(np.asarray(ft, np.float32).reshape(-1))
        self.buf["resultant"].append(resultant_from_tactile(np.asarray(tactile)).reshape(-1))
        if USE_TACTILE:
            self.buf["tactile"].append(np.asarray(tactile, np.float32))
        return True

    @torch.no_grad()
    def infer(self):
        """스퀴즈 끝난 직후 호출. (절대강성, class, class_name) 반환.
           샘플 부족하면 (None,None,None)."""
        n = len(self.buf["resultant"])
        if n < MIN_LEN:
            return None, None, None
        sensor_raw = build_sensor(self.buf, self.mask_grip)
        s = downsample_avg(sensor_raw, FACTOR, offset=0)
        if len(s) < 1:
            return None, None, None
        x = torch.tensor(s, dtype=torch.float32)
        x = (x - self.mean) / (self.std + 1e-8)
        # ★ 플랜A: 분포 벗어난 특정 채널을 0(=학습평균)으로 강제 + 안전 클램프
        if self.force_zero:
            x[:, self.force_zero] = 0.0
        if CLAMP_Z and CLAMP_Z > 0:
            x = torch.clamp(x, -CLAMP_Z, CLAMP_Z)
        x = x.unsqueeze(0).to(self.device)
        lengths = torch.tensor([s.shape[0]], device=self.device)

        out = self.model(x, lengths=lengths)
        norm_val = float(out.squeeze().cpu().item())     # 정규화 강성 (0~1)
        stiffness = norm_val * (self.norm_max - self.norm_min) + self.norm_min
        # === DEBUG: 모델 출력 + 분포 벗어난 채널 확인 ===
        if DEBUG:
            print(f"[DEBUG] x shape={tuple(x.shape)} mean={x.mean():.3f} std={x.std():.3f}")
            print(f"[DEBUG] norm_val(0~1)={norm_val:.4f} -> 절대강성={stiffness:.3f} "
                  f"(min={self.norm_min}, max={self.norm_max})")
            xch = x.squeeze(0).mean(dim=0).cpu().numpy()   # (40,) 채널별 평균
            names = (["joint"]*16) + (["ft"]*12) + (["resultant"]*12)
            bad = [(i, names[i], round(float(xch[i]),1)) for i in range(len(xch)) if abs(xch[i]) > 4]
            print(f"[DEBUG] 분포벗어난 채널(|z|>4): {bad}")
        cls = sum(1 for b in self.boundaries if stiffness >= b)
        cname = self.class_names[cls] if cls < len(self.class_names) else f"class{cls}"
        return stiffness, cls, cname


# ==================== 단독 실행 (테스트/데모) ====================
def resolve_fruit_config(fruit):
    """과일 이름 -> (model_path, pose_file, force_zero_channels).
       모델이 없으면(None) 안내 후 종료."""
    cfg = FRUIT_CONFIG.get(fruit)
    if cfg is None:
        raise SystemExit(f"'{fruit}' 설정 없음. FRUIT_CONFIG 에 추가 필요.")
    if cfg.get("model") is None:
        raise SystemExit(
            f"'{fruit}' 모델이 아직 준비되지 않았습니다.\n"
            f"  → 그 과일을 학습한 .pth 경로를 real_deploy_inference.py 의 "
            f"FRUIT_CONFIG['{fruit}']['model'] 에 넣으세요.\n"
            f"  (현재 준비된 과일: "
            f"{[k for k,v in FRUIT_CONFIG.items() if v.get('model')]})")
    return cfg["model"], cfg.get("pose"), cfg.get("force_zero", [])


def ask_fruit():
    while True:
        try:
            raw = input("추론할 과일  [1]자두 [2]키위 [3]토마토 [4]레몬 : ").strip()
        except EOFError:
            raise SystemExit("입력 없음.")
        if raw in ("1", "2", "3", "4"):
            return FRUIT_BY_NUM[int(raw)]
        print("  1~4 입력.")


def standalone_demo():
    """엔진을 단독으로 띄워, motion 코드 없이 흐름만 확인하는 데모.
       실제 deploy 는 motion_sequence_A_self.py 안에서 엔진을 호출해야 함."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--label-dir", default=LABEL_DIR)
    args = ap.parse_args()

    fruit = ask_fruit()
    engine = StiffnessInferenceEngine(args.model, fruit, args.label_dir)

    hand_shm = ShmAccess(key=SHM_MSG_KEY)
    if not hand_shm.attach():
        raise SystemExit("핸드 SHM attach 실패.")
    paxini = PaxiniShmReader(key=PAXINI_SHM_KEY)
    if not paxini.attach():
        raise SystemExit("paxini SHM attach 실패.")

    print("\n[데모] 이 단독실행은 motion 통합 예시가 아님.")
    print("실제로는 motion_sequence 의 스퀴즈 루프에서 engine.add_sample(),")
    print("스퀴즈 종료 직후 engine.infer() 를 호출해야 함. (아래 INTEGRATION 참고)")


# ==================== motion_sequence 통합 예시 ====================
INTEGRATION_EXAMPLE = '''
# motion_sequence_A_self.py 안에서 (의사코드):

from real_deploy_inference import StiffnessInferenceEngine

# 시작 시 1회:
engine = StiffnessInferenceEngine(model_path=MODEL, fruit=ask_fruit())

# move_hand_to_squeeze() 의 스퀴즈+hold 루프 안 (매 제어주기):
def normal_forces_and_log():
    # 기존처럼 paxini.read() 로 force 판정도 하고,
    engine.add_sample(shm, paxini)          # 추론용 샘플도 같이 적재
    ...

# 스퀴즈+hold 가 끝나고 파지 복귀 직전:
stiffness, cls, cname = engine.infer()
if stiffness is not None:
    print(f"[추론] 강성={stiffness:.3f}  등급={cname} (class {cls})")
    # 필요하면 ROS publish 등
engine.reset()                               # 다음 데모 준비

# 핵심: squeeze_on 을 파일에 쓰지 않는다. motion 이 '지금 스퀴즈 중'을
#       이미 알고 있으니, 스퀴즈 루프 안에서 add_sample, 끝나면 infer 만 부르면 됨.
'''


if __name__ == "__main__":
    print(INTEGRATION_EXAMPLE)
    # standalone_demo()  # SHM 환경에서 테스트할 때 주석 해제