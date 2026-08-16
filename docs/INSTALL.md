# INSTALL — 최초 1회 설치

대상: 이 PC (`prime-ws`, Ubuntu 22.04 / ROS 2 Humble / 2x RTX 5090 32GB / `/home/user`).

무거운 자산(conda 환경 3종, HF 가중치 50GB)은 **이관 시 이미 배치·검증을 마쳤다.**
남은 것은 아래 4가지다.

| # | 항목 | 상태 |
|---|---|---|
| 1 | apt 패키지 (MoveIt 등) | **완료** |
| 2 | colcon 워크스페이스 2개 빌드 | **완료** (fr_ws 12개 + kistar_ws 7개) |
| 3 | 시스템 python `torch`(강성용) | **완료** (2.7.1+cu128) |
| 4 | `grasp_fruit` conda 환경 | **완료** |

아래는 재설치·다른 PC 이관 시의 절차 기록이다.

---

## 1. apt 패키지 (sudo)

```bash
cd ~/prime/ChanukHwang/RobotAgentSystem
tools/env/install_apt.sh
```

MoveIt / controller / joint_state / xacro 계열과 `libfranka` 의 C++ 빌드 의존
(`libpoco-dev`, `ros-humble-pinocchio`, `libeigen3-dev`, `libtinyxml2-dev`)을 넣는다.
`ros-humble-desktop`, `robot-state-publisher`, `tf2-ros`, `colcon` 은 이미 있다.

> Poco/pinocchio 가 없으면 `fr_ws` 빌드가 `Could NOT find Poco` 로 시작부터 abort 된다
> (libfranka 가 먼저 빌드되며 나머지를 끌고 내려간다).

카메라는 **Control PC 가 발행**하므로 `realsense2-camera` 는 필요 없다.
이 PC 로 카메라를 되돌릴 때만 `--with-camera` 를 붙이고, 아래 udev 규칙도 넣는다:

```bash
sudo cp tools/ros2/dex_ros/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 2. ROS2 워크스페이스 빌드

```bash
tools/ros2/build.sh
```

순서가 중요하다: `fr_ws`(franka_description) → `kistar_ws`.
kistar 의 xacro 가 `$(find franka_description)/robots/common/franka_robot.xacro` 와
fr3 yaml, 그리고 `package://franka_description/meshes/**` 를 참조하기 때문이다.

빌드되는 kistar 패키지 7개:

| 패키지 | 역할 |
|---|---|
| `franka_kistar_bringup` | MoveIt+RViz launch, 카메라 launch, `pose_commander.py` 등 |
| `franka_kistar_description` | 듀얼 FR3 + KISTAR 핸드 URDF/xacro + 메시 |
| `franka_kistar_moveit_config` | SRDF · kinematics · joint_limits · OMPL · controllers |
| `franka_kistar_isaac_moveit` | Isaac 연동 |
| `dual_arm_msgs` | `SequenceState.msg`, `RequestControl/ReleaseControl.srv` |
| `sequence_client` | `SequenceClient` — 4개 skill 이 모두 import 한다 |
| `kistar_hand_ros2` | 핸드 메시지 |

> `fr_ws` 전체 빌드가 `libfranka` 의존으로 실패하면, 이 PC 는 실기를 직접 잡지 않으므로
> 설명 패키지만 골라 빌드해도 된다:
> `cd tools/ros2/fr_ws && colcon build --symlink-install --packages-select franka_description`

> **`dual_arm_msgs` 와 `sequence_client` 는 이미 빌드해 두었다** (MoveIt 없이도 빌드되는
> 패키지라 먼저 처리했다). 시퀀스 규약 import 와 Control PC arbiter 수신을 확인한 상태다.
> 나머지 5개 패키지는 MoveIt 설치 후 빌드된다.

> ⚠ **conda 가 활성화된 셸에서 빌드하지 말 것.** cmake 가 conda python 을 잡으면
> `ModuleNotFoundError: No module named 'em'` 로 실패한다. `tools/ros2/build.sh` 는
> PATH 에서 conda 를 걷어내므로 그것을 쓰면 된다.

빌드 후 확인:

```bash
source tools/env/setup_env.sh
python3 -c "from dual_arm_msgs.msg import SequenceState; from sequence_client import SequenceClient; print('sequence OK')"
ros2 pkg list | grep franka_kistar
```

---

## 3. 시스템 python 의존 (강성 모듈)

강성 추론은 rclpy 때문에 시스템 `/usr/bin/python3`(3.10) 로 돈다. torch 는 `--user` 로 넣는다
(**conda 금지** — ABI 충돌로 종료 시 core dump).

```bash
/usr/bin/python3 -m pip install --user h5py
/usr/bin/python3 -m pip install --user torch==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

> ⚠ **버전을 반드시 핀할 것.** 핀 없이 설치하면 pip 가 최신 `torch 2.13+cu130` 을 가져오는데,
> 이 PC 의 드라이버는 CUDA 12.8 이라 `torch.cuda.is_available()` 이 False 가 된다.
> `2.7.1+cu128` 은 conda 환경 3종과 같은 버전이라 일관성도 좋다.
>
> ⚠ `--extra-index-url https://pypi.org/simple` 을 함께 주지 말 것 — 최신 setuptools(≥80)가
> `--user` 로 끌려 들어와 `colcon-core`(setuptools<80 요구)를 깨뜨린다.
> 이미 깨졌다면: `/usr/bin/python3 -m pip uninstall -y setuptools` (시스템 59.6.0 으로 복귀)

확인:
```bash
/usr/bin/python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# torch 2.7.1+cu128 True
```

> numpy / PyYAML / OpenCV / SciPy 는 ROS Humble 이 이미 시스템에 깔아 두었고,
> place skill 서버의 import 체인은 시스템 python3 에서 그대로 통과하는 것을 확인했다.

---

## 4. `grasp_fruit` conda 환경 (파지 skill)

이 환경만 이관 번들에 포함되지 않아 새로 만든다(py3.12 + torch cu128 + transformers).

```bash
cd skill-set/grasp
CONDA_BASE=$HOME/miniconda3 bash setup_pipeline_all.sh
```

확인:
```bash
~/miniconda3/envs/grasp_fruit/bin/python -c \
  "import torch; from transformers import Sam3Model, Sam3Processor; print(torch.__version__, torch.cuda.is_available())"
```

`run_scenario1_host.py` 는 이 환경의 python 절대경로를 상수로 들고 있다
(`GRASP_FRUIT_PY = ~/miniconda3/envs/grasp_fruit/bin/python`) — 환경 이름/위치를 바꿨다면
그 상수도 함께 고칠 것.

---

## 이미 끝난 것 (참고)

### conda 환경 3종 — `~/miniconda3/envs/`
`molmo` / `sam3` / `anyplace_cu128`. 원본 PC 의 `~/anaconda3` prefix 를 `~/miniconda3` 로
치환했고, `sam3` editable 설치도 새 place 경로로 재링크했다. import 검증 완료:
torch 2.7.1+cu128 / RTX 5090 인식 / `sam3` → `skill-set/place/sam3`.
자세한 내용과 재빌드 방법: [`tools/env/conda/README.md`](../tools/env/conda/README.md)

### HF 가중치 — `~/.cache/huggingface/hub/` (약 47GB)

| 모델 | 크기 | 쓰는 곳 |
|---|---|---|
| `models--allenai--Molmo2-8B` | 33G | 내려놓기 `molmo_service` |
| `models--facebook--sam3` | 6.5G | 파지의 HF `Sam3Model` + 내려놓기 `sam_service`(`sam3.pt`) |
| `models--Qwen--Qwen2.5-VL-7B-Instruct-AWQ` | 6.5G | 파지의 `--instruction` 옵션(자연어로 후보 선택). 기본 경로에서는 미사용 |
| `models--IDEA-Research--grounding-dino-tiny` | 659M | 현재 실행 경로에서 미사용. 향후 planner 지각용으로 보존 |

두 PC 의 캐시를 합쳤고 깨진 심볼릭 링크 0개를 확인했다. `facebook/sam3` 는 **HF gated** 라
재다운로드에 접근 승인이 필요하므로 이 캐시를 지우지 말 것. 실행 시
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` 을 건다(`setup_env.sh`).

### 설치하지 않아도 되는 것
**vLLM 서버와 그 전용 venv 는 필요 없다.** 현재 실행 경로에는 자연어 파싱 계층이 없고,
파지는 SAM3 를, 내려놓기는 자체 HTTP 모델 서비스 5종을 쓴다. GDINO 도 마찬가지다.
(`VLLM_USE_FLASHINFER_SAMPLER=0`, `--gpu-memory-utilization` 같은 vLLM 튜닝 항목도 무관하다.)

---

## 머신 레벨 전제 (프로젝트 밖에 있지만 없으면 안 도는 것)

이 두 가지는 저장소가 아니라 **OS/홈 디렉토리**에 있다. PC 를 새로 깔거나 홈을 초기화하면
다시 넣어야 한다.

### 1. Fast DDS 프로파일 — `~/.ros/fastdds_ros2_link.xml`

`~/.bashrc` 가 `FASTRTPS_DEFAULT_PROFILES_FILE` 로 활성화한다. NIC 가 2개(로봇망
`192.168.0.101` / 사내망 `161.122.114.73`)라 인터페이스 화이트리스트가 없으면 DDS 가 모든
인터페이스에 locator 를 광고해 사내망으로 디스커버리가 새어 나가고 Control PC 와의 연결이
불안정해진다.

⚠ 이 파일의 **`maxInitialPeersRange` 는 120 이상**이어야 한다. `useBuiltinTransports=false`
때문에 같은 호스트 디스커버리도 이 범위에 걸리며, 값이 작으면 늦게 뜨는 `move_group` 이
보이지 않아 place 서버가 죽는다. 상세: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

```bash
grep -c maxInitialPeersRange ~/.ros/fastdds_ros2_link.xml   # 1
grep maxInitialPeersRange ~/.ros/fastdds_ros2_link.xml      # 120
```

### 2. UDP 버퍼 — `/etc/sysctl.d/60-ros2-dds.conf`

카메라가 raw color + aligned depth + pointcloud 를 30Hz 로 보내 우분투 기본
`rmem_max=208KB` 로는 수신 버퍼가 넘친다(실측 수백만 건의 `receive buffer errors`).

```bash
sysctl net.core.rmem_max        # 2147483647
netstat -su | grep 'receive buffer errors'   # 증가하지 않아야 정상
```

없으면 다시 만든다:
```bash
sudo tee /etc/sysctl.d/60-ros2-dds.conf >/dev/null <<'EOF'
net.core.rmem_max = 2147483647
net.core.rmem_default = 8388608
net.core.wmem_max = 2147483647
net.core.netdev_max_backlog = 30000
EOF
sudo sysctl -p /etc/sysctl.d/60-ros2-dds.conf
```

## 설치 검증

```bash
source tools/env/setup_env.sh
tools/diagnostics/preflight.sh      # 로봇 미동작. Control PC 가 켜져 있어야 전부 통과한다.
```
