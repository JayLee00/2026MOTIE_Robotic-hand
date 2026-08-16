# TROUBLESHOOTING

증상 → 원인 → 조치. 먼저 `tools/diagnostics/preflight.sh` 를 돌려 어디서 막히는지 좁힐 것.

---

## 시퀀스 / 제어권

### 아무 단계도 시작하지 않는다 (`sequence arbiter publishers: 0`)
Control PC 의 `sequence_arbiter` 가 없다.
```bash
# [Control PC] ros2 launch trajectory_receiver control_pc.launch.py require_control:=true
```

### 첫 단계가 곧장 "성공"으로 넘어가거나 순서가 뒤엉킨다
직전 체인의 DONE 이 latched 로 남아 있다. **arbiter 를 재시작**하라(collab_policy §4).
러너 자체는 "RUNNING 을 본 뒤의 DONE" 만 인정하므로 오판하지 않지만, skill 쪽
`wait_for_previous_done` 은 latched DONE 을 그대로 통과한다.

### `seq N 실패: 하트비트 끊김 → arbiter 가 IDLE 로 회수`
그 단계 프로그램이 죽었다. `logs/run_*/skill_<stage>.log` 를 볼 것.
DONE 이 안 나갔으므로 **뒷단계는 진행되지 않는다** — 설계대로 동작한 것이다.

### 타겟을 보내는데 로봇이 안 움직인다
`require_control:=true` 모드에서는 **제어권 없는 타겟은 무시**된다(정상).
`/sequence_state` 의 `owner` 가 내 번호인지 확인.

---

## 카메라

### `ros2 topic list` 는 비었는데 `ros2 topic hz` 는 30Hz 가 나온다
**카메라는 정상이다.** `list` 계열 명령이 ros2 데몬 캐시를 거치기 때문에, 데몬이 방금 떴거나
오래된 상태면 살아 있는 토픽을 빈 목록으로 보고한다. `hz` 는 자기 노드를 만들어 직접 구독하므로
그쪽이 사실이다.

```bash
ros2 daemon stop          # 캐시 비우기 (다음 호출에서 새로 뜬다)
ros2 topic list --no-daemon
```

같은 이유로 `ros2 node list` 도 살아 있는 노드를 누락한다 — 통합 초기에 이 때문에 Control PC
노드 3개(`trajectory_receiver_node_r`, `hand_target_receiver`, `sequence_arbiter`)가 죽은 것으로
잘못 판단한 적이 있다. **모순되는 신호가 보이면 실제 데이터 쪽을 믿을 것.**

### 카메라 토픽이 안 보인다
카메라 발행자는 **Control PC** 다.
```bash
tools/camera/check_camera.sh
```
1. Control PC 에서 `realsense_front.launch.py align_depth.enable:=true` 가 떠 있는가
2. 양쪽 `ROS_DOMAIN_ID=9` 인가
3. **방화벽이 DDS 를 막고 있지 않은가** — ping 은 되는데 토픽만 안 보이면 대개 이것이다
   (`sudo ufw status`; 서브넷 허용 필요)
4. 같은 서브넷인가 (이 PC `192.168.0.101` / Control PC `192.168.0.100`)

### color/camera_info 는 오는데 `aligned_depth_to_color` 가 비어 있다
launch 에 `align_depth.enable:=true` 가 빠졌다. 파지·내려놓기 모두 정렬 depth 를 쓴다.

---

## MoveIt / 트윈

### 모든 arm move 가 실패한다
**`move_group` 이 2개**일 때의 전형적 증상이다(`/move_action` 중복).
```bash
ros2 node list --no-daemon | grep move_group     # 정확히 1개여야 한다
```
⚠ `--no-daemon` 을 빼면 데몬 캐시 때문에 **살아 있는 move_group 을 0개로 볼 수 있다.**
그 말을 믿고 하나 더 띄우면 정확히 이 버그가 만들어진다. `tools/moveit/launch_twin.sh` 는
rclpy 로 노드 그래프를 직접 보고, 검사에 실패하면 아예 기동을 거부한다.
place skill 서버 preflight 와 러너 모두 이 경우 실행을 중단한다. 여분 트윈을 종료할 것.

### 트윈이 안 뜬다 / launch 가 죽는다
`logs/run_*/twin.log`. MoveIt 패키지 미설치가 가장 흔하다 → `tools/env/install_apt.sh`.

### 먼저 이것부터 — DDS 잔재 정리

디스커버리가 이상하면(`no move_group node`, `Failed init_port`, 노드가 오락가락) 아래를 먼저 돌린다.
고아 ROS 프로세스와 스테일 공유메모리를 한 번에 정리한다.

```bash
tools/diagnostics/clean_dds.sh            # 확인만
tools/diagnostics/clean_dds.sh --apply    # 정리
```

고아 노드가 왜 문제인가: DDS participant 를 계속 점유해 새로 뜨는 `move_group` 의
participant ID 를 밀어 올린다 → 아래 `maxInitialPeersRange` 문제를 직접 유발한다.

### `[RTPS_TRANSPORT_SHM Error] Failed init_port fastrtps_portNNNN: open_and_lock_file failed`
Fast DDS 공유메모리 포트가 깨졌다. **ROS 프로세스가 살아 있는 상태에서 `/dev/shm/fastrtps_*`
를 지우면** 이렇게 된다(진단 중 실제로 발생시킨 적 있다).

```bash
tools/diagnostics/clean_dds.sh --apply    # 프로세스 정리 후에만 SHM 을 지운다
```

⚠ `/dev/shm/fastrtps_*` 를 손으로 지우지 말 것. 꼭 필요하면 **모든 ROS 프로세스를 먼저 종료**한
뒤에만 지운다. `fastdds shm clean` 은 죽은 것만 안전하게 정리한다.

### place 서버가 `no move_group node on domain 9` 로 죽는다 (트윈은 멀쩡히 떠 있는데)
가장 헷갈리는 증상이다. `ros2 node list` 로는 `/move_group` 이 보이는데 새로 뜨는 프로세스만
그것을 못 본다.

**원인: Fast DDS 프로파일의 `maxInitialPeersRange` 가 실제 participant 수보다 작다.**

이 PC 는 NIC 가 2개(로봇망 `192.168.0.101`, 사내망 `161.122.114.73`)라
`~/.ros/fastdds_ros2_link.xml` 로 인터페이스 화이트리스트를 건다(`~/.bashrc` 가 
`FASTRTPS_DEFAULT_PROFILES_FILE` 로 활성화). 그 파일은 `useBuiltinTransports=false` 라서
**기본 멀티캐스트 메타트래픽 locator 가 사라지고**, 같은 호스트 노드끼리도
`127.0.0.1` 유니캐스트로 **participant ID 0 ~ (maxInitialPeersRange-1)** 만 탐색한다.

그래서 ID 가 그 범위를 넘는 participant 는 **나중에 뜨는 프로세스에게 영영 안 보인다**:

| 왜 move_group 만 걸리나 | |
|---|---|
| 트윈의 static TF / state publisher | 즉시 뜬다 → 낮은 ID → 발견됨 |
| `move_group` | 로봇 모델·플래닝 파이프라인 로드에 20초+ → 높은 ID → **안 보임** |
| `ros2 node list` 는 왜 보이나 | 데몬이 오래 살아 있어 최초 announce 때 이미 알고 있다 |

라이브 구성은 로컬 participant 가 25~40개(트윈 ~14, place 서버, 러너, skill, 데몬)라
`10` 으로는 턱없이 부족했다. **현재 `120`** 으로 되어 있다:

```bash
grep maxInitialPeersRange ~/.ros/fastdds_ros2_link.xml     # 120 이어야 정상
```

증상이 재발하면 이 값부터 볼 것. 되돌리려면 백업이 `~/.ros/fastdds_ros2_link.xml.bak-*` 에 있다.

빠른 판정 — **데몬을 거치지 않고** 확인한다:
```bash
ros2 node list --no-daemon | grep move_group     # 비어 있으면 이 문제
```

### 트윈이 뜨자마자 전부 종료된다 — `[urdf_snapshot] ... status='mismatch'`
launch 는 생성 URDF 스냅샷(`urdf/generated/*.urdf`)과 사이드카 `*.sha256` 의 해시가 일치하는지
검사하고, 어긋나면 **즉시 전체를 shutdown 한다**(`strict_urdf_snapshot` 기본값 true).
xacro/메시를 고쳤거나 스냅샷 파일이 어떤 이유로든 바뀌면 재생성한다:

```bash
tools/urdf/regenerate.sh      # 스냅샷 + 사이드카 재생성 + place 사본 동기화
```

로그에 `[urdf_snapshot] OK — generated/*.urdf hashes match sidecars.` 가 나오면 정상이다.
급하면 `tools/moveit/launch_twin.sh strict_urdf_snapshot:=false` 로 우회할 수 있지만,
스냅샷과 xacro 가 어긋난 채 도는 것이므로 권장하지 않는다.

> dex_ros 의 `regenerate_urdf.sh` 를 직접 부르면 REPO_ROOT 자동 계산이 어긋나
> `input not found` 가 난다(원본 PC 의 디렉토리 깊이를 가정한다). `tools/urdf/regenerate.sh`
> 래퍼가 올바른 `REPO_ROOT` 를 넘겨준다.

---

## 파이썬 / 환경

### `ModuleNotFoundError: dual_arm_msgs` (또는 `sequence_client`)
kistar_ws 미빌드 또는 미소싱.
```bash
tools/ros2/build.sh
source tools/env/setup_env.sh
```

### colcon 빌드가 `ModuleNotFoundError: No module named 'em'` 로 실패한다
**셸에 conda 가 활성화된 상태로 빌드했다.** cmake 의 `find_package(Python3)` 가 conda python 을
잡아서 시스템 dist-packages 의 `em`(empy)·`lark` 를 못 본다. `tools/ros2/build.sh` 는 PATH 에서
conda 를 걷어내므로 그것을 쓰면 된다. 수동으로 돌린다면:
```bash
conda deactivate
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v miniconda3 | paste -sd: -)"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV; hash -r
source /opt/ros/humble/setup.bash && colcon build --symlink-install
```

### fr_ws 빌드가 `Could NOT find Poco` 로 실패한다
`libfranka` 의 C++ 빌드 의존이 없다. `tools/env/install_apt.sh` 가 넣어 준다
(`libpoco-dev`, `ros-humble-pinocchio`, `libeigen3-dev`, `libtinyxml2-dev`).

> 참고: 이 PC 는 실기를 직접 잡지 않으므로 활성 실행 경로에 필요한 franka 패키지는
> `franka_description` **하나뿐**이다(kistar xacro 가 `$(find franka_description)` 로 include).
> 의존을 넣기 곤란하면 그것만 빌드해도 트윈은 뜬다:
> `cd tools/ros2/fr_ws && colcon build --symlink-install --packages-select franka_description`

### colcon 이 `setuptools<80` 오류를 낸다
`pip install --user` 로 새 setuptools 가 들어왔다. 되돌리면 시스템 59.6.0 을 쓴다:
```bash
/usr/bin/python3 -m pip uninstall -y setuptools
```

### `import rclpy` 는 되는데 노드가 조용히 죽거나 종료 시 core dump
**conda python 으로 ROS2 를 돌렸다.** conda 가 번들한 libstdc++ 등이 시스템 RMW/DDS 확장
모듈과 ABI 충돌한다. rclpy 를 쓰는 코드는 반드시 `/usr/bin/python3`.
`tools/env/setup_env.sh` 가 PATH 에서 conda 를 걷어내고 `PYTHONPATH`/`CONDA_*` 를 지운다.
셸에 conda 가 활성화된 상태라면 `conda deactivate` 후 다시 source 할 것.

### 파지 단계에서 `grasp_fruit` python 을 못 찾는다
`skill-set/grasp/scripts/run_scenario1_host.py` 의 `GRASP_FRUIT_PY` 절대경로와
실제 환경 위치가 다르다. 환경 생성: `cd skill-set/grasp && bash setup_pipeline_all.sh`.

### 강성 단계에서 `ModuleNotFoundError: torch`
시스템 python 에 torch 가 없다.
```bash
/usr/bin/python3 -m pip install --user torch \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

---

## place (내려놓기)

### `place skill 서버가 READY 가 되지 못했다`
서버 preflight 3종 중 하나가 막힌 것이다 — `logs/run_*/place_server.log` 에 어느 것인지 찍힌다.
1. 모델 서비스 5종 `/health` (8810/8811/8801/8816/8815)
2. `move_group` 정확히 1개
3. 카메라 3스트림

### 모델 서비스 기동이 `python: command not found` + conda 오류 리포트로 죽는다
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
  ... conda/activate.py, in _get_deactivate_scripts(old_conda_prefix)
run_services.sh: line NN: python: command not found
```
**conda 환경 변수가 모순 상태**(= `CONDA_SHLVL=1` 인데 `CONDA_PREFIX` 없음)일 때 `conda activate`
가 이렇게 죽는다. conda 가 "활성 env 가 있으니 먼저 deactivate 하자"고 판단한 뒤 그 경로를
`None` 으로 들고 join 하기 때문이다.

두 겹으로 해결되어 있다:
1. `tools/env/setup_env.sh` 가 conda 를 벗어날 때 `CONDA_SHLVL=0` 까지 **일관되게** 맞춘다
   (`CONDA_PREFIX` 만 지우면 이 버그가 난다).
2. `run_services.sh` 는 아예 `conda activate` 를 쓰지 않고 각 env 의 인터프리터를
   **절대경로로 직접 호출**한다 (`~/miniconda3/envs/<env>/bin/python`). 세 env 모두
   `etc/conda/activate.d` 훅이 없어 건너뛰는 것이 없다.

직접 확인:
```bash
source tools/env/setup_env.sh
echo "$CONDA_SHLVL / ${CONDA_PREFIX:-unset}"      # 0 / unset 이어야 정상
bash skill-set/place/vision_pipeline/run_services.sh
```

### 모델 서비스가 안 뜬다 / VRAM 부족
`run_services.sh` 는 GPU0=Molmo(~17GB), GPU1=SAM+AnyPlace+IGR(~13GB) 로 나눈다.
다른 프로세스가 GPU 를 쓰고 있지 않은지 `nvidia-smi` 로 확인.
```bash
bash skill-set/place/vision_pipeline/run_services.sh    # 단독 기동 + 로그 직접 보기
curl -s localhost:8810/health                            # 개별 확인
```

### `igr_service` 가 시작하지 못한다
`skill-set/place/dex_ros/.../urdf/generated/dual_fr3_kistar_v2.urdf` 가 없으면 실패한다.
이 파일은 dex_ros 쪽에서 gitignore 대상이라 place 저장소가 자기 사본을 들고 있다.

### 배치 위치가 어긋난다 / 충돌한다
외부 캘리브레이션 문제일 가능성이 높다. 카메라·로봇 배치를 바꿨다면
`core/extrinsic.py` + `tf.txt` + bringup static TF 를 **함께** 갱신해야 한다
([MIGRATION.md](MIGRATION.md) §5).

### 다음 단계에서 손가락이 갑자기 크게 움직인다 / 폭주한다
**손이 Voltage 모드에 남아 있는데 Position counts 를 받은 것**이다 (수백~수천이 raw duty 로
해석된다). 정상이라면 place 가 정상·예외·중단 어느 경로로 끝나든 Position + servo OFF 로
복귀한다 — [ARCHITECTURE.md §7](ARCHITECTURE.md) 참조.

즉시 확인:
```bash
ros2 topic echo /hand/right/cmd_mode --once     # 1 이어야 정상 (0 = Voltage)
ros2 topic pub --once /hand/right/cmd_mode std_msgs/msg/Int32 "{data: 1}"   # 수동 복귀
```
place 로그에서 `[R5] release + retract complete (hand -> Position mode, servo OFF)` 가 찍혔는지
볼 것. 안 찍혔다면 그 직전 로그가 원인이다. SIGKILL·전원 차단으로 죽은 경우는 어떤 안전망도
동작할 수 없으므로 위 명령으로 수동 복귀시킨다.

### `GRIP HANDOVER: no hand target seen on /hand/right/q_target`
place 가 제어권을 받았지만 직전 단계(물성 추론)가 보내던 hand target 을 못 봤다는 뜻이다.
그립은 수신 노드가 래치한 목표로 유지되므로 보통 진행에는 문제가 없지만, 원인은 확인할 것:

1. 물성 추론이 실제로 `/hand/right/q_target` 을 발행했는가 — `ros2 topic hz /hand/right/q_target`
2. place 서버가 물성 추론보다 **늦게** 떴는가 (서버는 seq 1 시작부터 떠 있어야 정상)
3. 도메인/방화벽 문제로 그 토픽만 안 보이는가

place 는 이 경우 target 을 **지어내지 않는다** — 추측한 16-DoF 목표를 발행하면 그 자체가 파지를
다시 명령하는 것이라 물체를 놓칠 수 있다.

### `tactile triggers DISABLED` 경고
PaXini 값이 0이거나 없다. Control PC 의 `paxini_writer.py` 를 **손끝 무접촉 상태**에서
다시 켜라(시작 순간이 0점 tare 다). 이 상태에서는 T_act(Case 2) 로만 release 한다.

---

## 모델 가중치

### HF 가 네트워크에서 받으려 한다 / gated 오류
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` 이 걸려 있어야 한다(`setup_env.sh` 가 설정).
`facebook/sam3` 는 **gated** 라 재다운로드에 접근 승인이 필요하다 —
`~/.cache/huggingface/hub/models--facebook--sam3` 를 지우지 말 것.

```bash
ls ~/.cache/huggingface/hub                  # 4개 모델
find ~/.cache/huggingface/hub -xtype l       # 깨진 심볼릭 링크 (0개여야 정상)
```

---

## 그 밖에

- 각 skill 모듈은 자체 문서를 갖고 있다: `skill-set/*/README.md`, `skill-set/*/docs/`
- 시퀀스 규약(번호 배정 · state 의미 · 실패 회수)은 [ARCHITECTURE.md §2](ARCHITECTURE.md) 에 정리되어 있다
- 이관 과정에서 무엇이 어디로 갔는지: [MIGRATION.md](MIGRATION.md)
