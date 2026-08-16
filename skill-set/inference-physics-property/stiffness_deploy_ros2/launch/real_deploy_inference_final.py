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

확인사항
1. 현재 코드 이름이 ros2 코드에 잘 반영되어 있나?
2. UNIFIED_MODELS에 학습한 모델 경로 (disp_X, disp_O)
3. FRUIT_ORDER가 학습과 같은 순서인지 확인

"""
import sys, os, math, time, inspect
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
LABEL_DIR   = str(_PKG_DIR / "labels/general")  # 학습/추론 공용 라벨.
# 시연 모델: train.py 가 저장한 체크포인트 (.pth). 레포 내 models/ 에 번들됨.
MODEL_PATH  = str(_MODELS_DIR / "260707_2244_lstm_m3_dX_lr0.001_do0.3_h192_L2_s42.pth")

# ===================== SOTA 앙상블 배포 설정 =====================
# deep_ws `docs/analysis/Phase_SOTA.md` §15 권고: baseline 구성(m3_dX, LSTM h192 L2
# do0.3 lr0.001)을 5개 seed로 학습해 예측을 평균하는 앙상블. 재파지·저수준 제어기
# 변경 없이 '추론측만' 바꾸는 개선이라 로봇 거동은 그대로다(단일 grasp → 5개
# 순전파 평균 → 등급). 개체등급·MAE·예측 안정성이 단일모델 대비 개선된다.
#   USE_SOTA_ENSEMBLE=False 로 두면 기존 단일모델(MODEL_PATH/UNIFIED_MODELS) 그대로.
USE_SOTA_ENSEMBLE = True
SOTA_ENSEMBLE_SEEDS = [42, 53, 64, 71, 82]
SOTA_ENSEMBLE_PATHS = [
    str(_MODELS_DIR / f"sota_m3_dX_baseline_s{s}.pth") for s in SOTA_ENSEMBLE_SEEDS
]

# 채널 플래그 — data_preprocessing.py 와 똑같이!
USE_JOINT     = True
# ★ joint 변위(Δjoint): 방식3 변위비교용. 아래 UNIFIED 설정이 자동으로 덮어씀.
USE_JOINT_DELTA = False
USE_JKIN      = True
USE_RESULTANT = True
USE_TACTILE   = False
TACTILE_SUMMARY = True

# ===================== 통합 모델(방식2/3) 배포 설정 =====================
# 통합 모델: 과일별 정규화가 아니라 '전체 공통 min/max'로 역변환.
# 과일 조건(방식3): 센서에 과일 one-hot 채널 추가. 방식2는 one-hot 없음.
#   -> 방식2/3, 변위 O/X 는 체크포인트 메타로 자동 인식(수동설정 불필요).
# 4개 모델(방식2/3 × 변위O/X)을 등록해두고 VARIANT 로 골라 비교.
USE_UNIFIED = True          # True=통합모델(방식2/3), False=기존 과일별(FRUIT_CONFIG)
_MODEL_DIR = str(_MODELS_DIR)
UNIFIED_MODELS = {
    # 방식3 (통합+조건, one-hot O)
    "m3_dX": {   # 방식3 변위X (절대 joint)
        "model": f"{_MODEL_DIR}/260707_2244_lstm_m3_dX_lr0.001_do0.3_h192_L2_s42.pth",
        "use_joint_delta": False,
        "force_zero": [8],
    },
    "m3_dO": {   # 방식3 변위O (Δjoint)
        "model": f"{_MODEL_DIR}/260707_2245_lstm_m3_dO_lr0.001_do0.3_h192_L1_s42.pth",
        "use_joint_delta": True,
        "force_zero": [],
    },
    # 방식2 (통합, one-hot X)
    "m2_dX": {   # 방식2 변위X
        "model": f"{_MODEL_DIR}/260707_2244_lstm_m2_dX_lr0.001_do0.3_h192_L2_s64.pth",
        "use_joint_delta": False,
        "force_zero": [8],
    },
    "m2_dO": {   # 방식2 변위O
        "model": f"{_MODEL_DIR}/260707_2245_lstm_m2_dO_lr0.0005_do0.3_h192_L1_s42.pth",
        "use_joint_delta": True,
        "force_zero": [],
    },
}
# ★ 어떤 모델로 deploy 할지 선택. "m3_dX"/"m3_dO"/"m2_dX"/"m2_dO" 중 하나.
#   (방식2/3, 변위O/X, one-hot, 공통정규화는 체크포인트가 자동 설정)
UNIFIED_VARIANT  = "m3_dX"
UNIFIED_NORM_MIN = 0.0           # 학습 UNIFIED_NORM 공통 min (체크포인트에 있으면 자동)
UNIFIED_NORM_MAX = 10.14         # 학습 UNIFIED_NORM 공통 max (체크포인트에 있으면 자동)
FRUIT_ORDER      = ["tomato", "kiwi", "lemon", "plum"]   # 학습 FRUIT_ORDER 순서 (자동시 덮어씀)
# 과일 one-hot 여부. 방식3=True/방식2=False. 체크포인트에 있으면 자동으로 덮어씀.
ADD_FRUIT_ONEHOT = True
USE_JKIN         = False if USE_UNIFIED else USE_JKIN    # 통합모델(방식2/3)은 FT 미사용
# =======================================================================

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
GRIP_FAIL_THRESH = 2.5    # ★ 학습 데이터셋과 반드시 일치시킬 것
# ★ 플랜A: 학습분포를 크게 벗어난(파지 성공이라 마스킹 안 되는) 특정 관절을
#   정규화 후 0(=학습평균)으로 강제. std 극소 관절(예: ch8=finger2 joint0)이
#   deploy 자세 미세차로 z 폭발 -> 모델이 그 채널만 보고 쏠리는 것을 방지.
#   이 관절들은 학습 때 거의 안 변해(정보 거의 없음) 0 처리해도 추론 영향 미미.
FORCE_ZERO_CHANNELS = [8]      # 정규화 후 0 으로 만들 채널 인덱스 (빈 리스트면 비활성)
CLAMP_Z = 5.0                  # |z|>CLAMP_Z 인 값은 ±CLAMP_Z 로 클램프 (0이면 비활성). 안전망.
DEBUG = False                  # ★ 시연=False(깔끔). 디버깅=True(센서/입력 통계 출력)

# ★ 1순위(이분산 불확실성 회귀, deep_ws/docs/MODEL_IMPROVEMENT.md 1순위) 연결.
#   이분산 체크포인트(model_config 에 heteroscedastic=True)를 로드하면 모델이
#   (μ, log σ²) 를 함께 출력 -> infer() 가 자동 감지해 σ(절대강성 단위)를 계산.
#   기존(비-이분산) 체크포인트를 쓰면 이 로직은 자동으로 비활성(σ=None, 기존과 100% 동일 동작).
#   임계값 출처: deep_ws eval.py calibration 결과 — m3_dX 3-seed 이분산모델의 σ 중앙값
#   (2026-07-09 측정, 개체 126개 기준 median≈0.6). ★ 모델 재학습/교체 시
#   eval.py --name <모델> 로 calibration 다시 뽑아서 재산정할 것.
SIGMA_CONFIDENCE_THRESHOLD = 0.6
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


# ---------- 촉각 포화 글리치 가드 (2026-08-11 추가) ----------
#   무엇: paxini 127점이 동시에 포화값 25.5 를 찍어 손가락 합력이 Σ127 = 3238.5 가
#         되는 현상(정상 |resultant| ~20 의 160배). 실측 0.4% 의 파지에서 나오고
#         대부분 2~3프레임 스파이크다.
#   왜 여기서 고치나: 학습 파이프라인에는 '글리치 비율이 구간의 1% 를 넘으면 그 구간을
#         버린다' 는 필터가 있는데 (a) 2~3프레임은 508프레임 구간에서 0.6% 라 **그
#         필터를 통과하고** (b) 배포에서는 애초에 라이브 파지를 버릴 수 없다.
#         → 버리는 대신 **이웃 정상 프레임으로 보간해 고친다**.
#
#   ★ 실측 (deep_ws/src/ecoflex2fruit phase4-1, 8시드 · 처음 보는 개체 4개):
#       클램프 없이   : 3프레임 글리치 주입 시 mass MAE 4.51 → 22.36 (8/8, p<0.001)
#       CLAMP_Z=5 켬  : mass MAE Δ −0.05 (ns) — **지금 설정에서는 이미 무해하다**
#     즉 이 가드가 당장 바꾸는 성능은 사실상 0 이다. 그래도 넣는 이유:
#       ① CLAMP_Z 는 끌 수 있는 튜너블이고(0 이면 비활성) 전 채널에 걸리는 무차별
#          안전망이다 — 특정 센서 고장 방어를 거기에 의존하면 안 된다.
#       ② 클램프는 글리치 값을 ±5σ 로 **눌러 둘 뿐** 여전히 틀린 값이 남는다.
#          이 가드는 그 프레임을 실제에 가까운 값으로 **되돌린다**.
#       ③ 학습 전처리(deep_ws config.SAT_MODE="repair")와 같은 처리를 배포에도 두어
#          두 경로가 어긋나지 않게 한다.
#   ⚠ 드리프트(파지 도중 시작하는 영점 이동)는 이 가드로 **안 막힌다** — 클램프로도
#     안 막힌다(실측 stif MAE +1.85, p<0.001). 별도 대응이 필요하다.
SAT_RESULTANT_MAX = 50.0     # 손가락별 |resultant| 성분이 이 값을 넘으면 포화로 본다
SAT_REPAIR_MAX_FRAC = 0.5    # 프레임의 이 비율을 넘게 포화면 보간이 창작이 된다 → 포기


def despike_saturation(buf, verbose=None):
    """buf["resultant"](·"tactile") 의 포화 프레임을 이웃 정상 프레임으로 선형 보간.

    반환: 고친 프레임 수 (0 이면 아무것도 안 했다).
    """
    verbose = DEBUG if verbose is None else verbose
    res = buf.get("resultant")
    if not res:
        return 0
    r = np.asarray(res, dtype=np.float32).reshape(len(res), -1)      # (n, 12)
    bad_m = np.abs(r).max(axis=1) > SAT_RESULTANT_MAX
    if not bad_m.any():
        return 0
    if bad_m.mean() > SAT_REPAIR_MAX_FRAC:               # 거의 전부 포화 → 못 고친다
        if verbose:
            print(f"[sat] 포화 {bad_m.mean()*100:.0f}% — 보간 포기(입력 그대로 진행)")
        return 0
    good = np.flatnonzero(~bad_m)
    bad = np.flatnonzero(bad_m)
    if len(good) < 2:
        return 0
    for key in ("resultant", "tactile"):
        seq = buf.get(key)
        if not seq:
            continue
        a = np.asarray(seq, dtype=np.float32)
        flat = a.reshape(len(a), -1)
        flat[bad] = np.stack([np.interp(bad, good, flat[good, c])
                              for c in range(flat.shape[1])], axis=1)
        buf[key] = list(flat.reshape(a.shape))
    if verbose:
        print(f"[sat] 포화 글리치 {len(bad)}프레임 보간 복구 (총 {len(r)}프레임)")
    return int(len(bad))


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


def build_sensor(buf, mask_grip, fruit=None):
    """버퍼(리스트들) -> (n,C). 마스킹 후 concat. 학습과 동일 순서.
       fruit: 방식3(통합+조건) 일 때 과일 one-hot 을 붙이기 위한 과일명."""
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
    sensor = np.concatenate(feats, axis=1)
    # ★ 방식3: 과일 one-hot 채널 추가 (학습 ADD_FRUIT_ONEHOT, FRUIT_ORDER 순서).
    # ★ 방식3(조건주입)만 one-hot 추가. 방식2(조건없음)는 ADD_FRUIT_ONEHOT=False -> 안 붙임.
    if ADD_FRUIT_ONEHOT and fruit is not None:
        onehot = np.zeros((n, len(FRUIT_ORDER)), dtype=np.float32)
        if fruit in FRUIT_ORDER:
            onehot[:, FRUIT_ORDER.index(fruit)] = 1.0
        else:
            print(f"[!] one-hot: 알 수 없는 과일 '{fruit}' -> 0벡터")
        sensor = np.concatenate([sensor, onehot], axis=1)
    return sensor


# ★ 학습(hdf5_logger)과 동일: FT(j_kin)는 SHM int16 raw 가 아니라
#   mN side-channel 파일에서 읽은 보정값이다. deploy 도 같은 소스를 써야 일치.
RAW_HAND_J_KIN_FILE = Path("/tmp/deep_ws_raw_06_hand_j_kin_mN.txt")
SIDE_CHANNEL_MAX_AGE_SEC = 1.0
# ★ P1#2 소스 고정: FT(j_kin) 를 mN 파일 '존재 여부'만으로 조용히 바꾸던 것을 명시 스위치로 잠근다.
#   기본 OFF = 현재 데이터셋(raw_hand_j_kin_mN_present=0, SHM raw)과 일치 → 파일이 생겨도 무시.
#   수집이 mN 로 이뤄졌다면 배포도 USE_MN_SIDE_CHANNEL=1 로 켜야 하고, 그때 파일이 없으면 실행을 거부한다.
USE_MN_SIDE_CHANNEL = os.environ.get("USE_MN_SIDE_CHANNEL", "0").strip().lower() in ("1", "true", "yes", "on")
_WARNED = {"ft": False, "ignored_mn": False}


def assert_jkin_source_pinned():
    """배포 시작 시 FT(j_kin) 소스를 고정·검증(P1#2). 파일 유무로 스케일이 바뀌는 조용한 불일치 차단.
       - 스위치 ON  + 파일 없음/만료 → 실행 거부(학습과 다른 스케일로 폴백 금지).
       - 스위치 OFF + 파일 존재      → 무시하고 SHM raw 로 고정(경고만)."""
    src = "mN_side_channel" if USE_MN_SIDE_CHANNEL else "SHM_raw"
    print(f"[deploy] FT(j_kin) 소스 고정: {src} (USE_MN_SIDE_CHANNEL={int(USE_MN_SIDE_CHANNEL)})")
    if USE_MN_SIDE_CHANNEL:
        if read_raw_hand_j_kin_mN(RAW_HAND_J_KIN_FILE) is None:      # 없음/만료
            raise SystemExit(
                f"[deploy] USE_MN_SIDE_CHANNEL=1 이지만 mN 파일 없음/만료: {RAW_HAND_J_KIN_FILE}\n"
                "  → 수집 당시 소스와 일치시키거나(side-channel 실행) 스위치를 끄세요. "
                "학습과 다른 스케일로 조용히 폴백하지 않고 실행을 거부합니다.")
    elif RAW_HAND_J_KIN_FILE.exists():
        print(f"  ⚠ mN 파일이 있으나 스위치 OFF → 무시하고 SHM raw 로 고정: {RAW_HAND_J_KIN_FILE}")


def current_jkin_source() -> str:
    return "mN_side_channel" if USE_MN_SIDE_CHANNEL else "SHM_raw"


def ckpt_jkin_source(ckpt_meta) -> str | None:
    """체크포인트가 밝힌 학습 데이터의 FT(j_kin) 출처. 라벨이 없으면 None."""
    if not ckpt_meta:
        return None
    v = ckpt_meta.get("jkin_source")
    if isinstance(v, str) and v.strip():
        return v.strip()
    p = ckpt_meta.get("raw_hand_j_kin_mN_present")      # 수집 HDF5 attr 이름 그대로
    if p is not None:
        return "mN_side_channel" if int(p) else "SHM_raw"
    return None


def assert_jkin_source_matches_ckpt(ckpt_meta):
    """P1#2(b) provenance 대조: 학습 데이터의 kin 출처 ↔ 지금 배포가 읽는 출처.

    (a) 소스 고정만으로는 **반대 방향 불일치**를 막지 못한다 — mN 로 수집·학습한 뒤
    배포를 기본값(OFF=SHM_raw)으로 띄우면 가드는 통과하고 조용히 다른 스케일을 먹는다.
    체크포인트 라벨과 대조해 그 경우를 실행 거부로 바꾼다.

    라벨이 없는 기존 체크포인트는 현 데이터셋 기준(raw_hand_j_kin_mN_present=0 = SHM_raw)
    으로 가정한다 → 기본(OFF) 배포는 그대로 통과하고, 스위치를 켠 배포만 거부된다.
    """
    cur = current_jkin_source()
    ck = ckpt_jkin_source(ckpt_meta)
    if ck is None:
        ck = "SHM_raw"
        print(f"  ℹ 체크포인트에 kin 소스 라벨 없음 → 데이터셋 기준 '{ck}' 으로 가정 "
              "(학습 스크립트가 ckpt['jkin_source'] 또는 "
              "ckpt['raw_hand_j_kin_mN_present'] 를 실어주면 정확히 대조됨)")
    if ck != cur:
        raise SystemExit(
            f"[deploy] FT(j_kin) 출처 불일치 — 학습='{ck}' / 배포='{cur}'.\n"
            f"  → USE_MN_SIDE_CHANNEL={int(USE_MN_SIDE_CHANNEL)} 을 학습 당시와 맞추세요"
            f"({'1' if ck == 'mN_side_channel' else '0'}). 스케일이 다른 입력으로 "
            "조용히 추론하지 않고 실행을 거부합니다.")
    print(f"[deploy] FT(j_kin) provenance 대조 OK: 학습=배포='{cur}'")


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
    # ★ P1#2: 소스는 명시 스위치가 결정한다(파일 존재로 조용히 바뀌지 않음).
    #   스위치 OFF → 파일을 아예 읽지 않고 SHM raw 고정. 시작 가드에서 이미 검증됨.
    ft_mN = read_raw_hand_j_kin_mN(RAW_HAND_J_KIN_FILE) if USE_MN_SIDE_CHANNEL else None
    if ft_mN is not None:
        ft = ft_mN.reshape(-1).astype(np.float32)        # 학습과 동일 (mN→int16 변환값)
    else:
        ft = np.array([[msg.j_kin[0][i][k] for k in range(Kinesthetic_Sensor_DOF)]
                       for i in range(Kinesthetic_Sensor_Num)], dtype=np.float32).reshape(-1)
        # 스위치 ON 인데 실행 중 파일이 stale/none 이 된 경우에만 경고(시작 시 없으면 이미 거부됨).
        if USE_MN_SIDE_CHANNEL and USE_JKIN and not _WARNED["ft"]:
            print("⚠ [deploy] USE_MN_SIDE_CHANNEL=1 인데 mN 파일 stale/none → 이번 tick SHM raw. "
                  f"side-channel 확인: {RAW_HAND_J_KIN_FILE}")
            _WARNED["ft"] = True
    tactile, _t, valid, _seq = paxini_reader.read()
    resultant = resultant_from_tactile(tactile).reshape(-1)
    return {"joint": joint, "ft": ft, "tactile": tactile,
            "resultant": resultant, "valid": int(np.array(valid).ravel()[0])}


# ---------- 모델/설정 로딩 ----------
def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval(), None, None, None, {}
    if "state_dict" not in ckpt:
        raise RuntimeError(f"알 수 없는 체크포인트: {model_path}")
    cfg = ckpt.get("model_config", {})
    model_name = ckpt.get("model_name", "lstm")
    model_cls = MODEL_REGISTRY.get(model_name, StiffnessRegressor)
    # ★ 새 학습본(deep_ws phase_sota)은 model_config 에 이 배포 model.py 가 모르는
    #   키(pooling/moe_lite/moe_trailing_extra 등)를 담을 수 있다. 생성자가 받는
    #   인자만 남겨 안전 로드 — 빠지는 키는 전부 기본값=OFF 라 동작 동일.
    accepted = set(inspect.signature(model_cls.__init__).parameters)
    cfg = {k: v for k, v in cfg.items() if k in accepted}
    model = model_cls(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    # ★ 입력 표현 메타 (deploy 자동 설정용)
    meta = {
        "use_joint_delta": ckpt.get("use_joint_delta", None),
        "unified_norm": ckpt.get("unified_norm", None),
        "unified_norm_min": ckpt.get("unified_norm_min", None),
        "unified_norm_max": ckpt.get("unified_norm_max", None),
        "add_fruit_onehot": ckpt.get("add_fruit_onehot", None),
        "fruit_order": ckpt.get("fruit_order", None),
        # ★ P1#2(b) provenance: 학습 데이터의 FT(j_kin) 출처. 학습 스크립트가 수집 HDF5 의
        #   root attr 을 그대로 실어주면 된다 — 둘 중 아무 이름이나 인식한다:
        #     ckpt["jkin_source"] = "SHM_raw" | "mN_side_channel"
        #     ckpt["raw_hand_j_kin_mN_present"] = 0 | 1        (수집 attr 이름 그대로)
        "jkin_source": ckpt.get("jkin_source", None),
        "raw_hand_j_kin_mN_present": ckpt.get("raw_hand_j_kin_mN_present", None),
    }
    return model, ckpt.get("norm_mean"), ckpt.get("norm_std"), ckpt.get("mask_grip_fail"), meta


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
        assert_jkin_source_pinned()      # P1#2: FT(j_kin) 소스 고정·검증(불일치/누락 시 실행 거부)

        # ★ model_path 가 리스트/튜플이면 SOTA 앙상블(여러 체크포인트 예측 평균),
        #   문자열이면 단일 모델(기존 동작). 앙상블 멤버는 같은 데이터·분할로 학습돼
        #   정규화통계(mean/std)·입력표현 메타가 동일하므로 첫 체크포인트 기준으로 잡는다.
        _paths = list(model_path) if isinstance(model_path, (list, tuple)) else [model_path]
        self.models = []
        self.mean = self.std = mask_flag = ckpt_meta = None
        for _i, _p in enumerate(_paths):
            _m, _mean, _std, _mf, _cm = load_model(_p, self.device)
            self.models.append(_m)
            if _i == 0:
                self.mean, self.std, mask_flag, ckpt_meta = _mean, _std, _mf, _cm
        self.model = self.models[0]           # 하위호환(기존 self.model 참조 대비)
        self.n_ensemble = len(self.models)
        # P1#2(b): 체크포인트가 밝힌 학습 데이터의 kin 출처 ↔ 지금 배포 소스 대조(불일치=거부).
        # 위 assert_jkin_source_pinned() 은 '배포 쪽 고정'만, 이건 '학습과의 일치'를 본다.
        assert_jkin_source_matches_ckpt(ckpt_meta)
        if self.n_ensemble > 1:
            print(f"[추론엔진] SOTA 앙상블 {self.n_ensemble}-seed 로딩 완료 "
                  f"(단일 grasp → {self.n_ensemble}개 순전파 평균)")
        # ★ 체크포인트 메타로 입력표현 자동 설정 (수동 UNIFIED_VARIANT/USE_JOINT_DELTA 실수 방지).
        #   체크포인트에 값이 있으면 그것을 신뢰(전역 설정을 덮어씀). 없으면(구모델) 전역 유지.
        global USE_JOINT_DELTA, USE_UNIFIED, FRUIT_ORDER, ADD_FRUIT_ONEHOT
        if ckpt_meta.get("use_joint_delta") is not None:
            if bool(ckpt_meta["use_joint_delta"]) != USE_JOINT_DELTA:
                print(f"  [자동] 변위(Δjoint)={ckpt_meta['use_joint_delta']} "
                      f"(체크포인트 기준, 전역설정 덮어씀)")
            USE_JOINT_DELTA = bool(ckpt_meta["use_joint_delta"])
        if ckpt_meta.get("unified_norm") is not None:
            USE_UNIFIED = bool(ckpt_meta["unified_norm"])
        if ckpt_meta.get("add_fruit_onehot") is not None:
            if bool(ckpt_meta["add_fruit_onehot"]) != ADD_FRUIT_ONEHOT:
                print(f"  [자동] one-hot(과일조건)={ckpt_meta['add_fruit_onehot']} "
                      f"(체크포인트 기준) -> 방식{'3' if ckpt_meta['add_fruit_onehot'] else '2'}")
            ADD_FRUIT_ONEHOT = bool(ckpt_meta["add_fruit_onehot"])
        if ckpt_meta.get("fruit_order"):
            FRUIT_ORDER = list(ckpt_meta["fruit_order"])

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

        # 정규화·경계: 방식3 통합이면 공통 min/max(정규화만), 경계는 과일별.
        if USE_UNIFIED:
            # 체크포인트에 공통 min/max 가 있으면 그것을 사용(자동), 없으면 수동 전역값.
            cmin = ckpt_meta.get("unified_norm_min")
            cmax = ckpt_meta.get("unified_norm_max")
            if cmin is not None and cmax is not None:
                self.norm_min, self.norm_max = float(cmin), float(cmax)
                print(f"  [자동] 공통정규화 min/max=[{self.norm_min:.3f},{self.norm_max:.3f}] "
                      f"(체크포인트 기준)")
            else:
                self.norm_min, self.norm_max = float(UNIFIED_NORM_MIN), float(UNIFIED_NORM_MAX)
                print(f"  [수동] 공통정규화 min/max=[{self.norm_min},{self.norm_max}] "
                      f"(체크포인트에 없음 -> 전역설정)")
            _, _, self.boundaries, self.class_names = load_fruit_config(label_dir, fruit)
        else:
            self.norm_min, self.norm_max, self.boundaries, self.class_names = \
                load_fruit_config(label_dir, fruit)

        self.last_sigma = None   # 직전 infer() 의 σ(절대강성 단위). 이분산 모델일 때만 채워짐.
        self.reset()
        print(f"[추론엔진] fruit={fruit}, 마스킹={'ON' if self.mask_grip else 'OFF'}, "
              f"정규화[{self.norm_min},{self.norm_max}], 경계={self.boundaries}")

    def reset(self):
        """다음 데모(누름) 위해 버퍼 비움."""
        self.buf = {"joint": [], "ft": [], "resultant": [], "tactile": []}
        self.last_desat = 0          # 직전 infer 에서 보간 복구한 포화 프레임 수

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
        #   ★ 포화 글리치 복구 — build_sensor 앞이어야 한다. downsample_avg(FACTOR=10)
        #     가 평균을 내기 전에 고쳐야 하고(평균에 섞이면 한 칸이 통째로 오염된다),
        #     detect_failed_fingers 의 손가락 판정도 성한 값으로 해야 한다.
        self.last_desat = despike_saturation(self.buf)
        sensor_raw = build_sensor(self.buf, self.mask_grip, fruit=self.fruit)
        s = downsample_avg(sensor_raw, FACTOR, offset=0)
        if len(s) < 1:
            return None, None, None
        x = torch.tensor(s, dtype=torch.float32, device=self.device)
        x = (x - self.mean) / (self.std + 1e-8)
        # ★ 플랜A: 분포 벗어난 특정 채널을 0(=학습평균)으로 강제 + 안전 클램프
        if self.force_zero:
            x[:, self.force_zero] = 0.0
        if CLAMP_Z and CLAMP_Z > 0:
            x = torch.clamp(x, -CLAMP_Z, CLAMP_Z)
        x = x.unsqueeze(0).to(self.device)
        lengths = torch.tensor([s.shape[0]], device=self.device)

        # ★ SOTA 앙상블: self.models(단일이면 1개)의 예측을 평균.
        #   1순위(이분산): 멤버가 heteroscedastic=True 면 (μ, log σ²) 튜플 반환 —
        #   μ 로 강성 계산은 기존과 동일, σ 는 절대강성 단위로 변환해 별도 보관.
        #   (스케일만 있는 affine 변환: Var(a·x)=a²Var(x) → σ_abs = σ_norm·(max-min))
        mus, sigmas_norm = [], []
        for _m in self.models:
            out = _m(x, lengths=lengths)
            if isinstance(out, tuple):
                mu, log_var = out
                mus.append(float(mu.squeeze().cpu().item()))
                sigmas_norm.append(float(torch.exp(0.5 * log_var).squeeze().cpu().item()))
            else:
                mus.append(float(out.squeeze().cpu().item()))   # 정규화 강성 (0~1)
        norm_val = float(np.mean(mus))                          # 앙상블 평균(단일이면 그 값)
        if sigmas_norm:
            # 이분산 멤버가 있으면 σ 도 평균해 신뢰도로 사용(앙상블은 σ도 안정화됨)
            self.last_sigma = float(np.mean(sigmas_norm)) * (self.norm_max - self.norm_min)
        else:
            self.last_sigma = None
        stiffness = norm_val * (self.norm_max - self.norm_min) + self.norm_min
        # === DEBUG: 모델 출력 + 분포 벗어난 채널 확인 ===
        if DEBUG:
            print(f"[DEBUG] x shape={tuple(x.shape)} mean={x.mean():.3f} std={x.std():.3f}")
            print(f"[DEBUG] norm_val(0~1)={norm_val:.4f} -> 절대강성={stiffness:.3f} "
                  f"(min={self.norm_min}, max={self.norm_max})")
            xch = x.squeeze(0).mean(dim=0).cpu().numpy()   # (40,) 채널별 평균
            names = (["joint"]*16) + (["ft"]*12 if USE_JKIN else []) \
                    + (["resultant"]*12) \
                    + (["onehot"]*len(FRUIT_ORDER) if ADD_FRUIT_ONEHOT else [])
            bad = [(i, names[i] if i < len(names) else "?", round(float(xch[i]),1))
                   for i in range(len(xch)) if abs(xch[i]) > 4]
            print(f"[DEBUG] 분포벗어난 채널(|z|>4): {bad}")
        cls = sum(1 for b in self.boundaries if stiffness >= b)
        cname = self.class_names[cls] if cls < len(self.class_names) else f"class{cls}"
        # ★ 이분산 모델이면 신뢰도 한 줄 출력 (기존 호출부 수정 없이도 바로 보임).
        #   비-이분산 체크포인트면 self.last_sigma=None 이라 조건문에서 자동 스킵.
        if self.last_sigma is not None:
            confident = self.last_sigma <= SIGMA_CONFIDENCE_THRESHOLD
            print(f"  [신뢰도] σ={self.last_sigma:.3f} "
                  f"{'(확실)' if confident else '⚠ 불확실 — 재측정 권장'} "
                  f"(임계={SIGMA_CONFIDENCE_THRESHOLD})")
        return stiffness, cls, cname

    def get_last_confidence(self):
        """직전 infer() 호출의 (σ, is_confident) 반환.
           이분산 모델이 아니면 (None, True) — 기존 흐름과 호환되도록 항상 '확실' 취급.
           motion_sequence 등에서 신뢰도에 따라 분기하고 싶을 때 선택적으로 사용:
             sigma, ok = engine.get_last_confidence()
             if not ok: ... # 재측정 유도 등"""
        if self.last_sigma is None:
            return None, True
        return self.last_sigma, self.last_sigma <= SIGMA_CONFIDENCE_THRESHOLD


# ==================== 단독 실행 (테스트/데모) ====================
def resolve_fruit_config(fruit):
    """과일 이름 -> (model_path, pose_file, force_zero_channels).
       방식3(USE_UNIFIED): 통합모델 사용(과일무관), 포즈만 과일별.
       방식1(else): 기존 FRUIT_CONFIG 과일별 모델."""
    global USE_JOINT_DELTA
    if USE_UNIFIED:
        v = UNIFIED_MODELS.get(UNIFIED_VARIANT)
        if v is None or not v.get("model") or "경로" in str(v.get("model")):
            raise SystemExit(
                f"통합모델('{UNIFIED_VARIANT}') 경로 미설정. "
                f"UNIFIED_MODELS['{UNIFIED_VARIANT}']['model'] 채우세요.")
        USE_JOINT_DELTA = bool(v.get("use_joint_delta", False))   # 변위여부 모델에 동기화
        pose = FRUIT_CONFIG.get(fruit, {}).get("pose")            # 포즈는 과일별
        print(f"[통합모델] 선택={UNIFIED_VARIANT}, 변위={USE_JOINT_DELTA}, "
              f"과일={fruit}, 포즈={pose}")
        return v["model"], pose, v.get("force_zero", [])

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