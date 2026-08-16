#!/usr/bin/env python3
"""단일 명령 파이프라인 러너 — 물체 파지(1) → 손 안 조작(2) → 물성 추론(3) → 물체 내려놓기(4).

자연어/미션 개념은 없다 — 어떤 과일을 집을지만 인자로 받아 4단계를 순서대로 돌린다.
상위 판단(무엇을 어디에 어떤 순서로)은 추후 VLM high-level planner 의 몫이고,
planner 는 이 스크립트를 인자만 바꿔 그대로 호출하면 된다.

동작 원리 (dev/docs/collab_policy.md 시퀀스 규약):
  각 skill 프로그램은 SequenceClient(n) 으로 **직전 번호의 DONE 을 스스로 기다렸다가**
  제어권을 얻고, 정상 종료하면 DONE 을 낸다. 즉 순서는 제어 PC 의 sequence_arbiter 가
  강제한다. 이 러너가 하는 일은:
     1) 상시 서버(모델 5종 / MoveIt 트윈 / place 서버) 를 확인하고 없으면 띄운다
     2) 단계 진입 시 해당 skill 프로그램을 spawn 한다
     3) /sequence_state 를 관측해 그 단계의 성공(DONE)·실패(IDLE 회수)·타임아웃을 판정한다
     4) 실패하면 즉시 프로세스 그룹을 죽여 뒷단계가 진행되지 않게 한다

  state: 0=IDLE(대기/실패회수) · 1=RUNNING · 2=DONE(정상종료).
  **DONE 만 "성공적으로 끝났다"** 는 뜻이며, RUNNING→IDLE 은 하트비트 끊김(=실패)이다.

사용:
    ./run_fruit_demo.sh --fruit orange
    ./run_fruit_demo.sh --fruit kiwi --stiffness-fruit kiwi
    ./run_fruit_demo.sh --dry-run                 # preflight 만 (로봇 안 움직임)
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(__file__).resolve().parent / "config.yaml"

SEQ_IDLE, SEQ_RUNNING, SEQ_DONE = 0, 1, 2
STATE_NAME = {SEQ_IDLE: "IDLE", SEQ_RUNNING: "RUNNING", SEQ_DONE: "DONE"}
ORDER = ["grasp", "inhand", "stiffness", "place"]
KOREAN = {"grasp": "물체 파지", "inhand": "손 안 조작",
          "stiffness": "물성 추론", "place": "물체 내려놓기"}


# ───────────────────────────────────────────────────────────────────────────
# 로깅
# ───────────────────────────────────────────────────────────────────────────
class Log:
    def __init__(self, path: Path | None = None):
        self.t0 = time.time()
        self.fh = open(path, "a", buffering=1) if path else None

    def __call__(self, msg: str, level: str = "INFO"):
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:6.1f}s] {level:5s} {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")

    def rule(self, title: str):
        self(f"──────── {title} ────────")


# ───────────────────────────────────────────────────────────────────────────
# 프로세스 관리
# ───────────────────────────────────────────────────────────────────────────
class Proc:
    """bash -c 로 띄운 하위 프로세스. 프로세스 그룹 단위로 정리한다.

    그룹 SIGTERM 은 실질적인 정지 수단이다: skill 이 죽으면 하트비트가 끊기고
    arbiter 가 3초 안에 제어권을 IDLE 로 회수한다(collab_policy §1).
    """

    def __init__(self, name: str, command: str, log_path: Path, log: Log, cwd: Path | None = None):
        self.name = name
        self.command = command
        self.log = log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logf = open(log_path, "ab", buffering=0)
        self._logf.write(f"\n===== launch {time.strftime('%F %T')}: {command}\n".encode())
        self.log_path = log_path
        self.proc = subprocess.Popen(
            ["/bin/bash", "-c", command],
            stdin=subprocess.DEVNULL,      # 대화형 프롬프트는 EOF 로 즉시 실패시킨다
            stdout=self._logf, stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            start_new_session=True,        # 그룹 단위 kill 을 위해
        )
        log(f"spawn '{name}' (pid {self.proc.pid}) → {log_path}")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def returncode(self):
        return self.proc.poll()

    def _kill_group(self, sig) -> bool:
        """프로세스 그룹 전체에 신호. 그룹이 이미 비었으면 False."""
        try:
            os.killpg(self.proc.pid, sig)      # start_new_session=True → pid == pgid
            return True
        except ProcessLookupError:
            return False

    def stop(self, grace_s: float = 5.0):
        if self.proc.poll() is None:
            self.log(f"terminating '{self.name}' (pid {self.proc.pid})", "WARN")
            self._kill_group(signal.SIGTERM)
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                self.log(f"'{self.name}' ignored SIGTERM → SIGKILL", "ERROR")
        # ⚠ 리더가 죽어도 자식은 남을 수 있다. `ros2 launch` 는 SIGTERM 을 받으면 자식을
        #   정리하는 도중에 **자기가 먼저 빠져나가는** 경우가 있어 rviz2 같은 무거운 노드가
        #   고아로 살아남는다. 고아는 DDS participant 를 계속 점유해 다음 실행의
        #   participant ID 를 밀어 올리고(디스커버리 실패의 원인) SHM 세그먼트도 붙잡는다.
        #   그래서 리더의 종료 여부와 무관하게 그룹을 한 번 쓸어 준다.
        time.sleep(0.5)
        if self._kill_group(signal.SIGKILL):
            self.log(f"'{self.name}' 잔여 자식 프로세스 정리 (SIGKILL to group)", "WARN")
        try:
            self._logf.close()
        except Exception:
            pass

    def tail(self, n: int = 20) -> str:
        try:
            lines = self.log_path.read_bytes().decode("utf-8", "replace").splitlines()
            return "\n".join("    | " + ln for ln in lines[-n:])
        except Exception:
            return "    | (로그 없음)"


class ProcPool:
    def __init__(self):
        self.procs: list[Proc] = []

    def add(self, p: Proc):
        self.procs.append(p)
        return p

    def stop_all(self):
        for p in reversed(self.procs):
            p.stop()


# ───────────────────────────────────────────────────────────────────────────
# 시퀀스 버스 관측 (제어 PC 의 sequence_arbiter)
# ───────────────────────────────────────────────────────────────────────────
class SequenceWatcher:
    """/sequence_state (dual_arm_msgs) 와 /sequence/shm_state (Int32MultiArray) 를 함께 구독.

    dual_arm_msgs 가 아직 빌드되지 않았어도 shm 폴백으로 동작한다.
    """

    def __init__(self, cfg: dict, log: Log):
        import rclpy
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)

        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("fruit_pipeline_runner")
        self._lock = threading.Lock()
        self._state = None            # (seq_id, state, owner)
        self._latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,   # latched 샘플 수신
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.topic = cfg["ros"]["sequence_topic"]
        self.shm_topic = cfg["ros"]["sequence_shm_topic"]

        self.have_msgs = False
        try:
            from dual_arm_msgs.msg import SequenceState
            self.node.create_subscription(SequenceState, self.topic, self._on_seq,
                                          self._latched_qos)
            self.have_msgs = True
        except ImportError:
            log(f"dual_arm_msgs 미탑재 → {self.shm_topic} 폴백으로만 관측한다 "
                f"(tools/ros2/build.sh 로 kistar_ws 빌드 권장)", "WARN")

        from std_msgs.msg import Int32MultiArray
        self.node.create_subscription(Int32MultiArray, self.shm_topic, self._on_shm, 10)

        from rclpy.executors import SingleThreadedExecutor
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spinning = True
        self._thread = threading.Thread(target=self._spin, name="ros2-spin", daemon=True)
        self._thread.start()

    def _spin(self):
        while self._spinning and self.rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_seq(self, msg):
        with self._lock:
            self._state = (int(msg.seq_id), int(msg.state), int(msg.owner))

    def _on_shm(self, msg):
        if len(msg.data) >= 3:
            with self._lock:
                self._state = (int(msg.data[0]), int(msg.data[1]), int(msg.data[2]))

    def snapshot(self):
        with self._lock:
            return self._state

    def arbiter_publishers(self) -> int:
        return (self.node.count_publishers(self.topic)
                + self.node.count_publishers(self.shm_topic))

    def count_publishers(self, topic: str) -> int:
        return self.node.count_publishers(topic)

    def move_group_nodes(self) -> list[str]:
        want = "move_group"
        out = []
        for name, ns in self.node.get_node_names_and_namespaces():
            if name == want:
                out.append(f"{ns.rstrip('/')}/{name}")
        return out

    def shutdown(self):
        self._spinning = False
        self._thread.join(timeout=2.0)
        try:
            self.node.destroy_node()
        finally:
            if self.rclpy.ok():
                self.rclpy.shutdown()


# ───────────────────────────────────────────────────────────────────────────
# 헬퍼
# ───────────────────────────────────────────────────────────────────────────
def fill(template: str, ctx: dict) -> str:
    """아는 {key} 만 치환한다. str.format 과 달리 ros2 메시지 인자의 리터럴 중괄호
    (예: "{data: 1}") 를 포맷 스펙으로 오해하지 않는다."""
    out = template
    for k, v in ctx.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def http_ok(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_log_pattern(proc: Proc, pattern: str, timeout_s: float, log: Log) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not proc.alive():
            return False
        try:
            if pattern in proc.log_path.read_bytes().decode("utf-8", "replace"):
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


class StageError(RuntimeError):
    pass


class BlockerError(StageError):
    """사람이 조치해야만 풀리는 실패 — 외부 의존(제어 PC 의 arbiter·카메라)이 없거나,
    러너가 자동으로 고칠 수 없는 상태(move_group 중복). 러너가 알아서 기동해 주는 항목
    (트윈 미기동 / 모델 서비스 미기동)과 구분해 점검 결과에 다르게 표시한다."""


# ───────────────────────────────────────────────────────────────────────────
# 러너
# ───────────────────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self, cfg: dict, args, log: Log):
        self.cfg = cfg
        self.args = args
        self.log = log
        self.pool = ProcPool()
        self.early_procs: dict[str, Proc] = {}
        self.watch: SequenceWatcher | None = None
        self.logdir = ROOT / "logs" / time.strftime("run_%m%d_%H%M%S")
        self.logdir.mkdir(parents=True, exist_ok=True)

        stiff_fruit = args.stiffness_fruit or cfg["stiffness_fruit_for"].get(
            args.fruit, cfg["stiffness_fruit_for"]["default"])
        if stiff_fruit not in cfg["stiffness_fruit_numbers"]:
            raise SystemExit(
                f"--stiffness-fruit '{stiff_fruit}' 은 강성 모델에 없다. "
                f"가능: {sorted(cfg['stiffness_fruit_numbers'])}")
        self.stiff_fruit = stiff_fruit

        self.ctx = {
            "root": str(ROOT),
            "grasp_dir": str(ROOT / "skill-set" / "grasp"),
            "inhand_dir": str(ROOT / "skill-set" / "in-hand-reorientation"),
            "physics_dir": str(ROOT / "skill-set" / "inference-physics-property"),
            "place_dir": str(ROOT / "skill-set" / "place"),
            "query": args.fruit,
            "fruit_num": cfg["stiffness_fruit_numbers"][stiff_fruit],
            # 강성 결과 GUI 는 기본 ON (분산환경 때와 동일). --no-stiffness-gui 로만 끈다.
            "gui_flag": "--no-gui" if args.no_stiffness_gui else "",
            # seq 2 VTDP 정책 (kist_deploy_pkg) — 정책은 스스로 종료하지 않으므로 시간 제한
            "inhand_policy_duration": (cfg.get("inhand_policy") or {}).get("duration_s", 15),
            "inhand_policy_device": (cfg.get("inhand_policy") or {}).get("device", "cuda:1"),
        }
        self.ctx["calibration"] = args.calibration or fill(cfg["grasp"]["calibration"], self.ctx)

    # ── preflight ────────────────────────────────────────────────────────
    def _check(self, name: str, fn, auto_startable: bool = False):
        """점검 하나 실행.

        --dry-run(점검 전용)에서는 실패해도 계속 진행하고 모아서 보고한다. 이때 결과를 두
        등급으로 나눈다:
          BLOCK — 사람이 조치해야 한다 (제어 PC 의 arbiter/카메라, move_group 중복)
          AUTO  — 지금은 없지만 실제 실행 시 러너가 기동한다 (트윈, 모델 서비스)
        실제 실행에서는 등급과 무관하게 첫 실패에서 즉시 중단한다 — 로봇을 움직이기 전에.
        """
        try:
            fn()
            self.checks.append((name, "PASS", ""))
        except StageError as e:
            if not self.args.dry_run:
                raise
            # BlockerError 는 auto_startable 이어도 사람 조치가 필요하다 (예: move_group 2개)
            grade = "AUTO" if (auto_startable and not isinstance(e, BlockerError)) else "BLOCK"
            self.checks.append((name, grade, str(e)))
            self.log(f"[{name}] {e}", "WARN" if grade == "AUTO" else "ERROR")

    def preflight(self):
        self.log.rule("PREFLIGHT" + (" (점검 전용 — 아무것도 기동하지 않는다)"
                                     if self.args.dry_run else ""))
        cfg = self.cfg
        self.checks: list[tuple[str, bool, str]] = []

        self.log(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID')} "
                 f"RMW={os.environ.get('RMW_IMPLEMENTATION')}")
        if os.environ.get("ROS_DOMAIN_ID") != str(cfg["ros"]["domain_id"]):
            raise StageError(
                f"ROS_DOMAIN_ID 가 {cfg['ros']['domain_id']} 가 아니다 — "
                "tools/env/setup_env.sh 를 source 했는지 확인할 것")

        # GUI 3종(RViz / 강성 결과 창 / grid 브라우저)은 모두 DISPLAY 를 상속받아 뜬다.
        # 없으면 실행은 되지만 화면이 하나도 안 뜨므로 미리 알린다 (실패 아님 — 헤드리스 운용 가능).
        if not os.environ.get("DISPLAY"):
            self.log("DISPLAY 가 없다 — RViz · 강성 결과 GUI · grid 브라우저가 뜨지 않는다 "
                     "(헤드리스로 진행됨). 화면이 필요하면 그래픽 세션에서 실행할 것.", "WARN")
        else:
            self.log(f"DISPLAY={os.environ['DISPLAY']} — RViz · 강성 GUI · grid 브라우저 표시 가능")

        self.watch = SequenceWatcher(cfg, self.log)
        time.sleep(2.0)   # discovery

        self._check("sequence_arbiter", self._check_arbiter)                       # 제어 PC 소유
        self._check("camera", self._check_camera)                                 # 제어 PC 소유
        self._check("paxini_raw", self._check_paxini_raw)                         # 제어 PC 소유
        self._check("inhand_policy_inputs", self._check_inhand_inputs)            # 제어 PC 소유
        self._check("move_group", self.ensure_twin, auto_startable=True)          # 러너가 기동
        self._check("model_services", self.ensure_services, auto_startable=True)  # 러너가 기동

        if self.args.dry_run:
            self.log.rule("PREFLIGHT 결과")
            label = {"PASS": "PASS ", "AUTO": "AUTO ", "BLOCK": "BLOCK"}
            note = {"PASS": "", "AUTO": "  (미기동 — 실행 시 러너가 자동 기동한다)",
                    "BLOCK": "  ← 사람이 조치해야 한다"}
            for name, grade, _ in self.checks:
                self.log(f"  {label[grade]}  {name}{note[grade]}")
            blocked = [n for n, g, _ in self.checks if g == "BLOCK"]
            if blocked:
                raise StageError(f"조치 필요: {blocked} — 위 ERROR 를 참고할 것")
            auto = [n for n, g, _ in self.checks if g == "AUTO"]
            if auto:
                self.log(f"조치 필요 없음. {auto} 는 실행 시 러너가 기동한다 → "
                         "바로 ./run_fruit_demo.sh 를 돌려도 된다.")
        self.log("preflight OK")

    def _discovery_s(self) -> float:
        return float(self.cfg["timeouts_s"].get("discovery", 30))

    def _wait_until(self, probe, timeout: float):
        """probe() 가 참 같은 값을 낼 때까지 폴링. (값, 걸린 시간) 반환.

        DDS 디스커버리는 즉시 끝나지 않는다 — 고정 sleep 뒤에 한 번만 보고 판정하면
        멀쩡히 살아 있는 발행자를 '없다'고 오판한다. 보이면 바로 통과하므로 정상일 때
        추가 지연은 없다.
        """
        t0 = time.monotonic()
        while True:
            v = probe()
            if v:
                return v, time.monotonic() - t0
            if time.monotonic() - t0 >= timeout:
                return v, time.monotonic() - t0
            time.sleep(0.5)

    def _check_arbiter(self):
        cfg = self.cfg
        n, took = self._wait_until(self.watch.arbiter_publishers, self._discovery_s())
        self.log(f"sequence arbiter publishers: {n} (discovery {took:.1f}s)")
        if n == 0:
            raise BlockerError(
                "제어 PC 의 sequence_arbiter 가 보이지 않는다 "
                f"({cfg['ros']['sequence_topic']} / {cfg['ros']['sequence_shm_topic']} 발행 없음).\n"
                "  제어 PC: ros2 launch trajectory_receiver control_pc.launch.py require_control:=true")
        # latched 샘플도 기다린다. 못 받은 채로 넘어가면 아래 stale-DONE 경고가 침묵하는데,
        # 그건 "직전 체인의 DONE 이 남아 다음 체인이 대기 없이 통과하는" 실제 오동작을
        # 놓친다는 뜻이다.
        st, _ = self._wait_until(self.watch.snapshot, min(10.0, self._discovery_s()))
        self.log(f"arbiter 현재 상태: {self._fmt(st)}")
        if st is not None and st[1] == SEQ_DONE:
            self.log("이전 체인의 latched DONE 이 남아 있다. 러너는 '새 RUNNING→DONE' 전이만 "
                     "성공으로 인정하므로 오판하지 않지만, skill 쪽 wait_for_previous_done 이 "
                     "즉시 통과할 수 있다 → 체인 전체 재실행 전 arbiter 재시작 권장 "
                     "(collab_policy §4).", "WARN")

    def _check_camera(self):
        topics = self.cfg["ros"]["camera_topics"]
        _, took = self._wait_until(
            lambda: all(self.watch.count_publishers(t) > 0 for t in topics),
            self._discovery_s())
        missing = []
        for t in topics:
            c = self.watch.count_publishers(t)
            self.log(f"camera {t}: publishers={c}")
            if c == 0:
                missing.append(t)
        if not missing:
            self.log(f"  camera 3스트림 확인 (discovery {took:.1f}s)")
        if missing:
            raise BlockerError(
                f"카메라 토픽 발행자가 없다: {missing}\n"
                "  카메라는 Control PC 가 발행한다 — Control PC 담당자에게 realsense_front "
                "launch(align_depth.enable:=true) 기동을 요청할 것.\n"
                "  진단: tools/camera/check_camera.sh")

    def _check_paxini_raw(self):
        """신규 물성 추론 데모(deploy_task3_ros2_demo)는 /paxini/right/raw(4x127x3)가 필수다.

        미발행이어도 데모 코드는 경고만 찍고 힘=0 으로 DONE 까지 진행한다 — '성공했는데
        결과가 쓰레기'인 조용한 실패이므로, 로봇을 움직이기 전에 여기서 막는다.
        """
        topic = self.cfg["ros"].get("paxini_raw_topic")
        if not topic:
            return
        if "stiffness" in self.args.skip:
            self.log("stiffness 단계 skip — paxini raw 점검 생략")
            return
        n, took = self._wait_until(lambda: self.watch.count_publishers(topic),
                                   self._discovery_s())
        self.log(f"paxini raw {topic}: publishers={n} (discovery {took:.1f}s)")
        if n == 0:
            raise BlockerError(
                f"{topic} 발행자가 없다 — 신규 물성 추론 데모의 필수 입력(127점 촉각 분포).\n"
                "  제어 PC 에서 손끝 무접촉 상태로 실행할 것 (시작 순간 0점 tare):\n"
                "  python3 ~/Dual_Arm_Hand_Ctrl/tools/paxini_writer.py --hand r")

    def _check_inhand_inputs(self):
        """VTDP 정책(seq 2)의 입력 3종을 점검한다. --inhand-legacy(HDF5 재생)면 생략."""
        topics = self.cfg["ros"].get("inhand_policy_topics") or []
        if not topics:
            return
        if "inhand" in self.args.skip:
            self.log("inhand 단계 skip — 정책 입력 점검 생략")
            return
        if self.args.inhand_legacy:
            self.log("--inhand-legacy — 정책 입력 점검 생략 (HDF5 재생 경로)")
            return
        _, took = self._wait_until(
            lambda: all(self.watch.count_publishers(t) > 0 for t in topics),
            self._discovery_s())
        missing = []
        for t in topics:
            c = self.watch.count_publishers(t)
            self.log(f"inhand policy {t}: publishers={c}")
            if c == 0:
                missing.append(t)
        if missing:
            raise BlockerError(
                f"VTDP 정책 입력 토픽 발행자가 없다: {missing}\n"
                "  제어 PC 의 손 관절/촉각/카메라(compressed) 발행을 확인할 것. "
                "기존 HDF5 재생으로 돌리려면 --inhand-legacy.")

    def _fmt(self, st):
        if st is None:
            return "없음 (arbiter idle 또는 미기동)"
        return f"seq_id={st[0]} state={st[1]}({STATE_NAME.get(st[1], '?')}) owner={st[2]}"

    # ── 상시 서버 ────────────────────────────────────────────────────────
    def ensure_twin(self):
        mode = self.args.twin
        # ⚠ '없다'고 성급히 판정하면 이미 떠 있는 트윈 위에 하나 더 띄워 중복 /move_action 을
        #   만든다(= 모든 arm move 실패). 앞선 arbiter/카메라 점검이 원격 participant 까지
        #   확인하고 온 뒤라 로컬 디스커버리는 사실상 끝나 있지만, 짧게 한 번 더 확인한다.
        found, took = self._wait_until(self.watch.move_group_nodes,
                                       min(10.0, self._discovery_s()))
        self.log(f"move_group 노드: {found or '없음'}"
                 + (f" (discovery {took:.1f}s)" if found else f" ({took:.0f}s 확인)"))
        if len(found) > 1:
            raise BlockerError(
                f"move_group 이 {len(found)}개다 — 트윈은 **정확히 1개**여야 한다. "
                "중복 /move_action 은 '모든 arm move 실패' 버그의 원인이다. "
                "여분을 종료한 뒤 다시 실행할 것.")
        if len(found) == 1:
            # 이미 떠 있어도 방금 기동한 것일 수 있다 — /move_action 이 열릴 때까지 기다린다
            # (수동으로 트윈을 띄우고 곧바로 실행하는 경우의 같은 레이스).
            if self.watch.count_publishers("/move_action/_action/status") > 0:
                return
            if self.args.dry_run:
                self.log("move_group 은 있으나 /move_action 아직 미개방 (기동 중일 수 있음)", "WARN")
                return
            self.log("move_group 은 떴으나 /move_action 미개방 — 서빙 시작 대기")
            deadline = time.time() + self.cfg["timeouts_s"]["twin_ready"]
            while time.time() < deadline:
                if self.watch.count_publishers("/move_action/_action/status") > 0:
                    self.log("move_group 준비 완료 (/move_action 서빙 중)")
                    return
                time.sleep(2.0)
            raise StageError("이미 떠 있는 move_group 이 /move_action 을 열지 않는다 — "
                             "그 프로세스를 종료하고 다시 실행할 것")
        if mode == "off" or self.args.dry_run:
            raise StageError("move_group 이 없다. tools/moveit/launch_twin.sh 로 먼저 띄우거나, "
                             "러너를 --twin auto(기본)로 실행해 자동 기동시킬 것")
        self.log("move_group 이 없다 → MoveIt 트윈을 직접 기동한다")
        p = self.pool.add(Proc("twin", self.cfg["services"]["twin_launch"],
                               self.logdir / "twin.log", self.log))
        deadline = time.time() + self.cfg["timeouts_s"]["twin_ready"]
        saw_node = False
        while time.time() < deadline:
            if not p.alive():
                raise StageError(f"트윈 launch 가 죽었다 (rc={p.returncode()})\n{p.tail()}")
            if not saw_node and self.watch.move_group_nodes():
                saw_node = True
                self.log("move_group 노드 확인 — /move_action 서빙 대기")
            # ⚠ 노드 이름이 보이는 것만으로는 부족하다. move_group 은 노드를 먼저 만들고
            #   로봇 모델·플래닝 파이프라인을 20초 넘게 로드한 뒤에야 /move_action 을 연다.
            #   여기서 노드만 보고 통과하면 곧바로 spawn 되는 place 서버가 자기 preflight
            #   (정확히 1개의 /move_action 요구)에서 떨어진다. 실제 서빙 시작을 기다린다.
            #   액션 서버는 /<action>/_action/status 를 발행하므로 msg 의존 없이 확인 가능.
            if self.watch.count_publishers("/move_action/_action/status") > 0:
                self.log("move_group 준비 완료 (/move_action 서빙 중)")
                return
            time.sleep(2.0)
        why = ("move_group 노드는 떴지만 /move_action 이 열리지 않았다"
               if saw_node else "move_group 노드 미검출")
        raise StageError(f"트윈 기동 타임아웃 ({why})\n" + p.tail())

    def ensure_services(self):
        ports = self.cfg["services"]["health_ports"]
        down = [n for n, p in ports.items() if not http_ok(p)]
        for n, p in ports.items():
            self.log(f"service {n:9s} :{p}  {'DOWN' if n in down else 'UP'}")
        if not down:
            return
        if self.args.services == "off" or self.args.dry_run:
            raise StageError(f"모델 서비스 미기동: {down}. "
                             "skill-set/place/vision_pipeline/run_services.sh 로 먼저 띄우거나, "
                             "러너를 --services auto(기본)로 실행해 자동 기동시킬 것")
        self.log(f"모델 서비스 {down} 미기동 → 직접 기동한다 (Molmo 로딩에 ~1분+)")
        cmd = fill(self.cfg["services"]["launch"], self.ctx)
        p = self.pool.add(Proc("model_services", cmd, self.logdir / "model_services.log", self.log))
        deadline = time.time() + self.cfg["timeouts_s"]["services_ready"]
        while time.time() < deadline:
            if not p.alive():
                raise StageError(
                    f"모델 서비스가 뜨지 못하고 run_services.sh 가 종료됐다 "
                    f"(exit={p.returncode()}; 서비스가 하나도 살아남지 못하면 wait 가 곧바로 "
                    f"돌아오므로 exit=0 일 수 있다). 마지막 로그:\n{p.tail(40)}")
            still = [n for n, q in ports.items() if not http_ok(q)]
            if not still:
                self.log("모델 서비스 5종 모두 UP")
                return
            time.sleep(5.0)
        raise StageError(f"모델 서비스 기동 타임아웃: {still}\n{p.tail(40)}")

    def start_place_server(self):
        if "place" in self.args.skip:
            self.log("place 단계 skip — place 서버를 띄우지 않는다", "WARN")
            return None
        self.log.rule("place skill 서버 기동 (seq 4 상시 대기 + seq 1 에서 parent prewarm)")
        cmd = fill(self.cfg["services"]["place_server"], self.ctx)
        p = self.pool.add(Proc("place_server", cmd, self.logdir / "place_server.log", self.log))
        pattern = self.cfg["services"]["place_server_ready_pattern"]
        if not wait_for_log_pattern(p, pattern, self.cfg["timeouts_s"]["place_server_ready"], self.log):
            raise StageError(
                "place skill 서버가 READY 가 되지 못했다 (preflight 실패 가능: 모델 서비스 / "
                "move_group 1개 / 카메라 3스트림)\n" + p.tail(40))
        self.log("place 서버 READY — Place 차례를 대기 중")
        return p

    def start_place_logger(self):
        if not self.args.place_logger:
            return None
        cmd = fill(self.cfg["services"]["place_logger"], self.ctx)
        return self.pool.add(Proc("place_logger", cmd, self.logdir / "place_logger.log", self.log))

    def run_sync_target(self):
        """체인 시작 직전 state→target 동기화 (블로킹).

        이전 실행이 남긴 스테일 latched 타겟과 현재 자세가 크게 다르면 임피던스
        로봇이 서보-온/제어 사이클 순간 튄다. arm 은 goto_q(현재=목표 — 이동 0),
        hand 는 측정 counts 재발행으로 재앵커한다. 실패 = 점프 위험이므로 중단.
        """
        template = self.cfg["services"].get("sync_target")
        if not template:
            return
        if self.args.no_sync_target:
            self.log("--no-sync-target — state→target 동기화 생략", "WARN")
            return
        self.log.rule("state → target 동기화 (스테일 타겟 점프 방지)")
        cmd = fill(template, self.ctx)
        self.log(f"명령: {cmd}")
        p = self.pool.add(Proc("sync_target", cmd, self.logdir / "sync_target.log", self.log))
        deadline = time.time() + self.cfg["timeouts_s"].get("sync_target", 150)
        while time.time() < deadline and p.alive():
            time.sleep(0.5)
        if p.alive():
            p.stop()
            raise StageError("state→target 동기화 타임아웃 — 스테일 타겟 점프 위험, 중단\n"
                             + p.tail(20))
        if p.returncode() != 0:
            raise StageError(f"state→target 동기화 실패 (rc={p.returncode()}) — "
                             "스테일 타겟 점프 위험, 중단\n" + p.tail(20))
        self.log("state→target 동기화 완료 ✓ (arm + hand 재앵커)")

    def start_fruit_viz(self):
        """과일 6DoF 오버레이(FoundationPose 3D bbox) — seq 2 데모 화면.

        등록(과일 락)에 시간이 걸리므로 체인 시작 시 미리 띄운다. 보조 화면이므로
        실패해도 체인은 계속 간다(WARN 만). --no-fruit-viz 로 끈다.
        """
        template = self.cfg["services"].get("fruit_viz")
        if not template or self.args.no_fruit_viz:
            if self.args.no_fruit_viz:
                self.log("--no-fruit-viz — 과일 6DoF 오버레이를 띄우지 않는다", "WARN")
            return None
        self.log.rule("과일 6DoF 오버레이 기동 (FoundationPose — seq 2 화면)")
        cmd = fill(template, self.ctx)
        p = self.pool.add(Proc("fruit_viz", cmd, self.logdir / "fruit_viz.log", self.log))
        pattern = self.cfg["services"].get("fruit_viz_ready_pattern", "fp_server 준비 완료")
        if wait_for_log_pattern(p, pattern,
                                self.cfg["timeouts_s"].get("fruit_viz_ready", 300), self.log):
            self.log("fruit_viz 준비 — 오버레이 창에서 과일을 클릭하면 3D bbox 가 붙는다")
        else:
            self.log("fruit_viz 가 준비되지 않았다 — 오버레이 없이 체인을 계속한다\n"
                     + p.tail(15), "WARN")
        return p

    def _stage_template(self, stage: str):
        """단계 명령 템플릿. inhand 는 --inhand-legacy 시 HDF5 재생 경로로 전환한다."""
        st = self.cfg["stages"].get(stage) or {}
        if stage == "inhand" and self.args.inhand_legacy:
            return st.get("command_legacy") or st.get("command")
        return st.get("command")

    def spawn_early_stages(self):
        """stages.*.spawn == early 인 단계를 체인 시작 시 미리 spawn 한다.

        시퀀스 규약상 각 skill 은 wait_for_previous_done 으로 자기 차례를 스스로
        기다리므로 조기 spawn 이 순서를 깨지 않는다. VTDP 정책(seq 2)은 이 시간에
        모델 로드 + CUDA 예열을 파지(seq 1)와 겹쳐 끝낸다 (engage 전 무발행 = 안전).
        """
        self.early_procs: dict[str, Proc] = {}
        for stage in ORDER:
            if stage in self.args.skip:
                continue
            st = self.cfg["stages"].get(stage) or {}
            template = self._stage_template(stage)
            if st.get("spawn") == "early" and template:
                cmd = fill(template, self.ctx)
                self.log(f"[{stage}] 조기 spawn (프리워밍) — skill 이 직전 DONE 을 자체 대기한다")
                self.log(f"명령: {cmd}")
                self.early_procs[stage] = self.pool.add(
                    Proc(stage, cmd, self.logdir / f"skill_{stage}.log", self.log))

    # ── 단계 실행 ────────────────────────────────────────────────────────
    def run_stage(self, stage: str):
        number = self.cfg["sequence_numbers"][stage]
        timeout = self.cfg["timeouts_s"][stage] * self.args.timeout_scale
        self.log.rule(f"seq {number} — {KOREAN[stage]} ({stage})")

        proc = self.early_procs.get(stage)
        template = self._stage_template(stage)
        if proc is not None:
            if not proc.alive():
                raise StageError(
                    f"조기 spawn 된 '{stage}' 프로세스가 자기 차례 전에 죽었다 "
                    f"(rc={proc.returncode()})\n" + proc.tail(30))
            self.log(f"조기 spawn 된 '{stage}' (pid {proc.proc.pid}) 진행 관측")
        elif template:
            cmd = fill(template, self.ctx)
            self.log(f"명령: {cmd}")
            proc = self.pool.add(Proc(stage, cmd, self.logdir / f"skill_{stage}.log", self.log))
        else:
            self.log("이 단계는 상시 서버가 담당한다 — arbiter 의 DONE 만 관측한다")

        ok, why = self.wait_sequence(number, timeout, proc)
        if not ok:
            if proc:
                proc.stop()      # 하트비트 끊김 → arbiter 3초 내 IDLE 회수
                self.log("최근 로그:\n" + proc.tail(30), "ERROR")
            raise StageError(f"seq {number} ({stage}) 실패: {why}")
        self.log(f"✅ seq {number} ({stage}) DONE")
        if proc:
            # 성공한 skill 은 스스로 End 후 종료한다. 잔여 프로세스는 정리.
            for _ in range(50):
                if not proc.alive():
                    break
                time.sleep(0.1)
            if proc.alive():
                self.log(f"'{stage}' 프로세스가 아직 살아 있다 → 정리", "WARN")
                proc.stop()

    def wait_sequence(self, number: int, timeout_s: float, proc: Proc | None):
        """[number, DONE] 로의 **새로운** RUNNING→DONE 전이를 기다린다.

        latched 로 남은 이전 체인의 DONE 을 성공으로 오인하지 않도록, RUNNING 을 한 번
        본 뒤의 DONE 만 인정한다(collab_policy §4).
        """
        start = time.monotonic()
        saw_running = False
        last = None
        while time.monotonic() - start < timeout_s:
            st = self.watch.snapshot()
            if st != last:
                self.log(f"  arbiter: {self._fmt(st)}")
                last = st
            if st is not None:
                seq_id, state, _owner = st
                if seq_id == number:
                    if state == SEQ_RUNNING:
                        saw_running = True
                    elif state == SEQ_DONE and saw_running:
                        return True, "DONE"
                    elif state == SEQ_IDLE and saw_running:
                        return False, "하트비트 끊김 → arbiter 가 IDLE 로 회수 (프로그램이 죽었다)"
                elif seq_id > number and saw_running:
                    return True, "다음 시퀀스가 이어받음"
            if proc is not None and not proc.alive():
                rc = proc.returncode()
                # 정상 종료 직후 DONE 이 아직 도착하지 않았을 수 있으니 잠깐 더 본다
                grace = time.monotonic() + 5.0
                while time.monotonic() < grace:
                    st = self.watch.snapshot()
                    if st and st[0] == number and st[1] == SEQ_DONE and saw_running:
                        return True, "DONE"
                    if st and st[0] > number and saw_running:
                        return True, "다음 시퀀스가 이어받음"
                    time.sleep(0.2)
                return False, f"프로그램이 DONE 없이 종료했다 (rc={rc})"
            time.sleep(0.1)
        return False, f"{timeout_s:.0f}s 타임아웃 (마지막 상태: {self._fmt(self.watch.snapshot())})"

    # ── 전체 실행 ────────────────────────────────────────────────────────
    def run(self) -> int:
        self.log(f"로그 디렉토리: {self.logdir}")
        self.log(f"대상 과일(파지 쿼리): {self.args.fruit!r} / 강성 모델 과일: "
                 f"{self.stiff_fruit} (번호 {self.ctx['fruit_num']})")
        self.preflight()
        if self.args.dry_run:
            self.log("--dry-run: preflight 까지만 수행하고 종료한다 (로봇 미동작)")
            return 0

        # 핸드 명령 블랙박스 — sync 포함 이후 모든 핸드 명령을 기록 (사고 포렌식)
        hcl = self.cfg["services"].get("hand_cmd_logger")
        if hcl:
            self.pool.add(Proc("hand_cmd_logger", fill(hcl, self.ctx),
                               self.logdir / "hand_cmd.log", self.log))
        self.run_sync_target()
        self.start_place_server()
        self.start_place_logger()
        self.start_fruit_viz()
        self.spawn_early_stages()

        for stage in ORDER:
            if stage in self.args.skip:
                self.log(f"seq {self.cfg['sequence_numbers'][stage]} ({stage}) skip", "WARN")
                continue
            self.run_stage(stage)

        self.log.rule("전체 시퀀스 완료 ✅")
        return 0


# ───────────────────────────────────────────────────────────────────────────
def parse_args(cfg: dict):
    p = argparse.ArgumentParser(
        prog="run_fruit_demo.sh",
        description="물체 파지 → 손 안 조작 → 물성 추론 → 물체 내려놓기 를 한 번에 실행한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="⚠ 이 명령은 실제 로봇을 움직인다. E-stop 옆 인원 상주 필수.")
    p.add_argument("-f", "--fruit", default="orange",
                   help="파지할 물체명 (SAM3 텍스트 쿼리). 기본: orange")
    p.add_argument("--stiffness-fruit", default=None,
                   choices=sorted(cfg["stiffness_fruit_numbers"]),
                   help="강성 추론에 쓸 과일 모델. 미지정 시 --fruit 로부터 매핑 "
                        "(강성 모델에 orange 가 없어 lemon 으로 대용)")
    p.add_argument("--calibration", default=None, help="파지 캘리브레이션 JSON 경로 override")
    p.add_argument("--skip", default="", help="건너뛸 단계 (쉼표: grasp,inhand,stiffness,place)")
    p.add_argument("--services", choices=["auto", "off"], default="auto",
                   help="place 모델 서비스 5종: auto=없으면 기동(기본), off=이미 떠 있어야 함")
    p.add_argument("--twin", choices=["auto", "off"], default="auto",
                   help="MoveIt 트윈: auto=없으면 기동(기본), off=이미 떠 있어야 함")
    p.add_argument("--no-fruit-viz", action="store_true",
                   help="과일 6DoF 오버레이(FoundationPose 3D bbox 창)를 띄우지 않는다")
    p.add_argument("--no-sync-target", action="store_true",
                   help="체인 시작 전 state→target 동기화를 생략한다 (점프 위험 감수)")
    p.add_argument("--inhand-legacy", action="store_true",
                   help="seq 2 를 기존 HDF5 궤적 재생(inhand_sequence_2.py)으로 되돌린다 "
                        "(기본: VTDP 학습 정책)")
    p.add_argument("--place-logger", action="store_true", help="/place/status 로거도 함께 띄운다")
    p.add_argument("--no-stiffness-gui", action="store_true",
                   help="강성 결과 GUI 를 띄우지 않는다 (기본: 띄움 — DISPLAY 필요)")
    p.add_argument("--timeout-scale", type=float, default=1.0, help="모든 단계 타임아웃 배율")
    p.add_argument("--dry-run", action="store_true", help="preflight 만 수행 (로봇 미동작)")
    a = p.parse_args()
    a.skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    unknown = a.skip - set(ORDER)
    if unknown:
        p.error(f"--skip 에 알 수 없는 단계: {sorted(unknown)} (가능: {ORDER})")
    return a


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    args = parse_args(cfg)
    log = Log()
    pipe = None
    try:
        pipe = Pipeline(cfg, args, log)
        log.fh = open(pipe.logdir / "pipeline.log", "a", buffering=1)
        return pipe.run()
    except StageError as e:
        log(str(e), "ERROR")
        return 1
    except KeyboardInterrupt:
        log("사용자 중단 (Ctrl+C) — 하위 프로세스를 정리한다", "WARN")
        return 130
    finally:
        if pipe is not None:
            pipe.pool.stop_all()
            if pipe.watch is not None:
                pipe.watch.shutdown()


if __name__ == "__main__":
    sys.exit(main())
