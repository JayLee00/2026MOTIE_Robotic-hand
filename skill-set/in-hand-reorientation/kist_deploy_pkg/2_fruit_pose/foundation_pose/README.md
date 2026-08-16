# foundation_pose — 과일 6DoF 자세추정 (FoundationPose 기반)

기존 파이프라인의 **orientation 이 안 잡히는 문제**를 FoundationPose 로 대체해 보는 모듈.
기존 코드(`fruit-manipulation/`, `record/`)는 **하나도 건드리지 않는다.**

- 논문: [FoundationPose (CVPR 2024)](https://arxiv.org/abs/2312.08344) · [프로젝트](https://nvlabs.github.io/FoundationPose/) · [코드](https://github.com/NVlabs/FoundationPose)

---

## 1. 왜 지금 orientation 이 안 잡히나

현재 경로는 이렇다:

```
live_bbox_gui.py (SAM2 마스크) → /inhand/bbox_corners (8코너 OBB)
        → fruit_pose_bridge.py → PCA 로 주축 뽑아 quaternion
```

`record/fruit_pose_bridge.py` 는 8코너에서 **PCA 주축**으로 방향을 만든다. 오렌지처럼
둥근 물체는 점군의 공분산이 등방(isotropic)에 가까워 **주축 방향이 프레임마다 임의로
튄다.** 알고리즘 버그가 아니라 입력이 방향 정보를 안 담고 있는 것이다.

## 2. FoundationPose 는 뭐가 다른가

깊이 점군의 모양이 아니라 **CAD 모델을 렌더링해서 RGB-D 와 정합**한다. 회전 가설을
여러 개 뿌리고(refine) 점수를 매겨(score) 고른다. 즉 **표면 텍스처가 회전을 결정한다.**

### ⚠ 그래도 남는 한계 (반드시 읽을 것)

**구에 가까운 물체의 회전은 기하학적으로 관측 불가능하다.** 깊이만 보면 오렌지는
어떻게 돌려도 같은 반구다. 그래서 회전을 잡아주는 건 오직 표면 텍스처(꼭지, 반점,
색 얼룩)뿐이다. 이건 FoundationPose 의 한계가 아니라 문제 자체의 성질이다.

| 물체 | 회전 관측 가능성 |
|---|---|
| 오렌지·사과처럼 매끈한 구 | **거의 불가** — 텍스처가 유일한 단서 |
| 꼭지·굴곡 있는 배·바나나 | 가능 (형상이 비대칭) |
| 텍스처 뚜렷한 물체 | 잘 됨 |

그래서 이 모듈은 **회전이 실제로 얼마나 안정적인지 측정하는 것**까지가 목적이다.
`--compare` 로 기존 방식과 나란히 띄워 quaternion 흔들림을 직접 비교할 수 있다.

정말로 그 오렌지의 회전이 필요하면 model-free(참조영상 ~16장 + Neural Object Field)
쪽이 맞다. 근사 구 메시로는 한계가 있다.

## 3. 세그멘테이션 — 논문은 어떻게 하나

논문 원문:

> "Given the RGBD image, the object is detected using an off-the-shelf method
> such as **Mask R-CNN or CNOS**"

그리고 BOP 리더보드에서는 CNOS 를 썼다. **핵심은 마스크가 첫 프레임에만 필요하다는
것이다.** 이후에는 직전 자세로 렌더링한 것과 현재 프레임을 함께 refiner 에 넣어
추적하므로 매 프레임 세그멘테이션이 없다.

> "at each timestamp, we send the cropped current frame and the rendering using
> the previous pose to the pose refinement module"

논문에 XMem 같은 video object segmentation 은 안 나온다.

**→ 우리는 SAM2 를 쓰므로 그대로 맞는다.** 기존 SAM2 체크포인트
(`fruit-manipulation/sam2.1_hiera_tiny.pt`)로 첫 프레임 마스크만 만들고,
그 뒤는 FoundationPose 트래킹에 넘긴다. 매 프레임 SAM2 를 돌리는 지금 방식보다
오히려 가볍다.

## 4. 구조 — 왜 도커로 쪼갰나

FoundationPose 는 nvdiffrast 와 PyTorch3D 를 **nvcc 로 소스 빌드**해야 한다.
그런데 이 PC 에는 CUDA 툴킷이 없고(`nvcc: command not found`), 호스트에 새로 깔면
이미 잘 돌고 있는 **torch 2.6+cu124 / SAM2 환경이 깨질 위험**이 있다.

그래서 추론만 공식 도커 이미지 안에 가두고, ROS2·SAM2 는 호스트에 그대로 뒀다:

```
 호스트                                   컨테이너 (wenbowen123/foundationpose)
 ─────────────────────────────           ────────────────────────────────────
 fp_ros_node.py                          fp_server.py
  · ROS2 구독 (color/depth/info)   TCP    · FoundationPose
  · 첫 프레임 SAM2 마스크        ──5577─► · nvdiffrast + PyTorch3D
  · /fruit/pose, /fruit/size 발행  ◄────  · register() / track_one()
```

컨테이너는 `--network=host` 라 `127.0.0.1:5577` 로 그냥 붙는다. `/home` 을 마운트하므로
메시·가중치 경로가 양쪽에서 동일하다. 프로토콜은 길이 접두 + pickle (양쪽 다 파이썬).

## 5. 파일

| 파일 | 역할 | 어디서 도나 |
|---|---|---|
| `setup.sh` | 이미지·저장소·가중치·확장빌드·메시 준비 (한 번만) | 호스트 |
| `run_foundation_pose.sh` | **원커맨드 런처** | 호스트 |
| `fp_server.py` | FoundationPose TCP 추론 서버 | 컨테이너 |
| `fp_ros_node.py` | ROS2 브리지 + SAM2 초기 마스크 | 호스트 |
| `fruit_label_node.py` | 과일 종류 라벨 발행 + CAD 교체 | 호스트 |
| `prepare_mesh.py` | 스캔 메시 진단·단위보정·변환 | 호스트 |
| `make_fruit_mesh.py` | 과일 근사 메시(OBJ+텍스처) 생성 | 호스트 |
| `capture_scene.py` | RGB-D 를 레포 demo_data 형식으로 녹화 | 호스트 |
| `test_symmetry_snap.py` | 대칭 스냅·부호 연속성 검증 (카메라 불필요) | 호스트 |
| `fruits.yaml` | 과일 카탈로그 (id·이름·CAD·공칭크기) | — |
| `FoundationPose/` | 업스트림 저장소 (클론, git 제외) | — |
| `weights/`, `assets/` | 가중치·메시 (git 제외) | — |

## 5-2. ROS2 토픽

기존 `/fruit/*` 네임스페이스를 그대로 쓴다. 30~32 는 레코더에 이미 있고, 라벨만 새로 붙었다.

| 토픽 | 타입 | 내용 | 레코더 필드 |
|---|---|---|---|
| `/fruit/pose` | PoseStamped | 위치 + 방향 (FoundationPose) | `30_fruit_pos`(3), `31_fruit_quat`(4) |
| `/fruit/size` | Float32MultiArray[3] | **비전 실측** 축 길이 [m] | `32_fruit_size`(3) |
| `/fruit/type` | Int32 | 과일 종류 id (레몬=1, 자두=2 …) | `33_fruit_type`(1) ← **추가 필요** |
| `/fruit/type_name` | String | 표시용 이름 | (기록 안 함) |
| `/fruit/set_type` | String | **입력** — 종류 전환 | — |
| `/fruit/reset` | String | **입력** — 재세그/CAD 교체 | — |

`/fruit/type` 은 TRANSIENT_LOCAL 이라 나중에 뜬 레코더도 마지막 라벨을 받는다.

레코더에 넣으려면 `record/ros2_hdf5_recorder.py` 의 필드 표에 한 줄 추가하면 된다
(원본이라 여기서 건드리지 않았다):

```python
("33_fruit_type",  "/fruit/type",  Int32, 1, lambda m: [float(m.data)]),
```

### 크기를 왜 CAD 가 아니라 비전에서 뽑나

CAD 는 **종류당 대표 1개**라 개체 크기를 모른다. 그래서 마스크+깊이를 3D 로 역투영해
PCA 축 길이를 잰다. 실측 70×55mm 레몬에서 **64.6 × 55.6 mm** 가 나왔다.

두 가지 보정이 들어간다:
- **배경 깊이 제거** — 마스크가 물체 경계를 넘으면 배경 화소가 섞여 점군이 시선
  방향으로 늘어난다. 보정 전 826mm 가 나왔다. 물체 깊이 중앙값 ±(크기×1.5) 밖은 자른다.
- **공칭 대비 3배 검사** — 그래도 이상하면 버리고 CAD 공칭치를 쓴다. 말도 안 되는
  값이 HDF5 에 들어가는 것보다 낫다.

한계: 카메라는 앞면만 보므로 시선 방향 축은 과소평가된다. 과일은 장축 둘레로 대체로
회전대칭이라 그 축을 중간축으로 대체해 `(장축, 중간축, 중간축)` 으로 낸다.
CAD 공칭치를 쓰려면 `--size-source cad`.

## 5-3. 과일 카탈로그

`fruits.yaml` 이 단일 출처다. 종류당 대표 CAD 1개 + 정수 라벨.

```yaml
- id: 1
  name: lemon
  mesh: assets/lemon.obj
  nominal: [0.070, 0.055, 0.055]
```

```bash
python3 foundation_pose/fruit_label_node.py --list   # 카탈로그 + CAD 유무 확인
```

**새 과일 추가**
1. 스캔 → `prepare_mesh.py scan.obj -o assets/plum.obj --target-extents 0.055,0.050,0.050`
2. `fruits.yaml` 에 항목 추가 (id 는 다음 번호)

⚠ **한 번 부여한 id 는 바꾸지 마세요.** HDF5 에 그대로 들어가므로 바꾸면 과거 데이터의
라벨 의미가 달라집니다.

**전환** (실행 중에 가능, CAD 도 같이 바뀐다)
```bash
ros2 topic pub --once /fruit/set_type std_msgs/String '{data: "plum"}'
```

전환하면 이 순서로 자동 진행된다:

```
/fruit/set_type "orange"
  → fruit_label_node 가 fruits.yaml 조회
  → /fruit/type = 3 발행
  → /fruit/reset 에 assets/orange.obj 경로 발행
  → fp_ros_node → 서버 set_mesh → reset_object() → 재등록
```

⚠ **CAD 가 없으면 라벨만 바뀌고 자세는 이전 메시로 계속 돈다.** 이러면 HDF5 에
"자두"라고 적힌 채 레몬 CAD 로 뽑은 자세가 들어간다. 로그에 경고가 뜬다:

```
CAD 파일이 없습니다: .../assets/plum.obj — 라벨만 발행하고 자세는 이전 메시 유지
```

수집 전에 `fruit_label_node.py --list` 로 쓸 과일의 CAD 가 ✓ 인지 확인하세요.

또 물체를 **실제로 바꿔 놓은 경우**엔 전환 후 창에서 한 번 클릭해 주는 게 안전하다.
자동 재등록은 직전 자세 위치를 시드로 쓰므로, 새 과일을 다른 자리에 놓았으면
엉뚱한 곳을 잡을 수 있다.

## 6. 사용법

```bash
# 최초 1회 (도커 이미지 ~20GB 받음)
bash foundation_pose/setup.sh

# 실행 — 이 한 줄이면 끝
bash foundation_pose/run_foundation_pose.sh --fruit lemon
```

창이 두 개 뜬다. **`FoundationPose select` 에서 과일을 클릭**하면 등록되고,
`Fruit 6DoF overlay` 에 3D 박스가 붙는다. 종료는 Ctrl+C (컨테이너·republish 까지 정리).

기본값: 클릭 선택 켬 · 연속 세그 5Hz · 자동 재등록 켬 · 대칭 스냅 켬 ·
시작 시 이전 잔재 자동 정리.

```bash
--seg-hz 0 --no-click   # 자세 주기 최대 (연속 세그·클릭 없이 자동 ROI 시드)
--seg-hz 15             # 세그 더 자주 (자세 주기는 내려감)
--fruit orange          # 다른 과일
--check                 # 전제조건만 점검
--compare               # /fruit_fp/* 로 발행해 기존 파이프라인과 동시 비교
```

기본 모드는 `/fruit/pose` · `/fruit/size` 로 발행하는 **드롭인**이라
`record/fruit_overlay.py` 를 원본 그대로 재사용한다.

### 성능 (RTX 4080 SUPER, 레몬 70×55×55, 카메라 30Hz)

| | |
|---|---|
| `/fruit/pose` | **27.2 Hz** (클릭 + 연속 세그 5Hz) |
| FoundationPose 추론 | 15~22 ms (`track_one`) |
| 최초 등록 | ~1.5~1.9 s (회전 후보 252개) |
| TCP 전송·직렬화 | 1~6 ms |
| 위치 안정도 | 정지 시 ±1 mm |

세그는 자세추정에 **안 쓰인다**(`track_one` 에 마스크 인자가 없다). 오버레이와
크기 측정용이므로, 자세가 최우선이면 `--seg-hz 0` 이 가장 빠르고 정확하다.

메시 크기를 실측에 맞추려면:

```bash
python3 foundation_pose/make_fruit_mesh.py --diameter 0.075 -o foundation_pose/assets/orange.obj
```

## 7. 실측 결과 — CAD 가 전부다

같은 코드로 두 과일을 재보면 **CAD 품질이 성능을 가른다**는 게 분명하다.

| | 오렌지 (생성한 구 + 가짜 텍스처) | **레몬 (실제 스캔 + 4096² 텍스처)** |
|---|---|---|
| 초기 수렴 | ~26초 방황 | **1회 등록에 바로 락** |
| 회전 | 임의값에 수렴 (절대값 무의미) | **안정, 움직여도 추종** |
| 위치 | ±1 mm | ±1 mm |

오렌지는 매끈한 구라 회전이 기하학적으로 관측 불가능하고, 텍스처마저 제가 만든
가짜라 RGB 정합이 단서를 못 준다. 레몬은 (1) 길쭉해서 장축이 형상만으로 관측되고
(2) 진짜 표면 텍스처가 있어 즉시 잡힌다.

**→ 새 과일을 추가할 땐 반드시 실물을 스캔하세요.** 근사 구 메시는 위치 전용이다.

오렌지 90초 측정 원본:

| 항목 | 결과 |
|---|---|
| 위치 | **[+0.023, −0.059, +0.334] m — 90초간 변동 ≤1mm** |
| 회전 | 초기 ~26초 요동 → 이후 수렴·고정 |

회전 quaternion 변화:

```
t+0  ~ t+26s   [-0.402,+0.341,-0.184,+0.830]   ← 프레임마다 완전히 다름
               [+0.515,+0.695,-0.248,-0.436]
               [+0.122,+0.961,-0.231,+0.092]      (구 대칭성 때문에 자세가 방황)
t+26 ~ t+43s   [+0.957,-0.123,+0.262,+0.026]   ← 여기서 락
               [+0.961,-0.141,+0.237,+0.023]
               [+0.965,-0.127,+0.229,+0.015]      (성분 변동 ±0.02 이내로 안정)
```

**해석**

- **위치는 그냥 쓰면 된다.** ±1mm 는 기존 PCA 중심보다 확실히 낫다.
- **회전은 "절대값"이 아니라 "일관성"으로 봐야 한다.** 수렴한 자세는 오렌지의 실제
  방향이라는 보장이 없다(구라서 기준이 없음). 하지만 한 번 락되면 계속 그 값을
  유지하므로 **상대 회전 추적**은 된다. 기존 PCA 방식은 끝까지 계속 튄다.
- 초기 수렴에 시간이 걸리는 건 구 대칭성 탓이다. 꼭지처럼 눈에 띄는 특징이 카메라를
  향하면 훨씬 빨리 잡힌다.

**다음에 해볼 것**
- 실측 지름을 맞춘 메시 (`--diameter`) — 지금은 70mm 가정
- 그 오렌지 사진으로 텍스처를 갈아끼우면 초기 수렴이 빨라진다
- 회전이 정말 중요하면 model-free(참조영상 16장 + NeRF) 경로

## 7-2. 자세가 이상할 때

**축이 갑자기 휙 바뀐다** — 레몬은 장축 둘레로 거의 회전대칭이라 여러 회전이 관측상
동등하다. `register()` 는 회전 후보 252개를 직전 자세와 무관하게 새로 뿌리므로,
재등록하면 동등하지만 다른 대표값이 나온다. 그래서 재등록 직후 **직전 자세에 가장
가까운 대칭 동등물로 스냅**한다(매 프레임이 아니다 — 그러면 진짜 spin 도 지워진다).
쿼터니언 부호(`q` vs `-q`)도 직전 값에 맞춘다. 검증: `test_symmetry_snap.py`.

그래도 spin 이 못 쓸 수준이면 `symmetry_tfs` 를 선언해 후보 자체를 합칠 수 있다
(`estimater.py:120`). 대신 장축 둘레 회전 정보를 버리게 된다.

**주기가 갑자기 떨어졌다** — 십중팔구 `republish` 가 새서 쌓인 것이다. `ros2 run` 은
래퍼라 그것만 죽으면 자식 `republish` 가 남는다. 15개까지 쌓여 같은 컬러 프레임이
15중으로 발행되고 시간동기화가 무너져 자세가 30Hz→2.5Hz 로 떨어진 적이 있다.

```bash
pgrep -fc "image_transport/republish"      # 1 이어야 정상
ros2 topic hz /front_cam/front/color/image_fast   # 카메라(30Hz)보다 높으면 중복
```

런처가 시작 시 정리하지만, 수동으로 띄웠다면 `pkill -f image_transport/republish`.

**자세가 발산해서 안 돌아온다** — 자동 재등록(`--auto-reset`, 기본 켬)이 잡아준다.
껐다면 놓친 순간 영원히 못 돌아온다(실측: z 0.31→0.11m). 반대로 이 판정이 너무
빡빡하면 1~2초마다 재등록해 오히려 자세가 망가지므로, `--check-tol` 은 발산
감지용으로 넉넉히(기본 0.15m) 잡혀 있다.

## 8. 전제조건

- 제어 PC 에서 realsense 가 떠 있고 **정렬 깊이**가 켜져 있을 것
  (`ros2 param set /front_cam/front align_depth.enable true`)
- `nvidia-container-toolkit` (setup.sh 가 확인해 준다)
- GPU: FoundationPose + SAM2 동시 상주. RTX 4080 16GB 면 충분하다.
