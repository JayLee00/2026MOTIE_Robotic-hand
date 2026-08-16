#!/usr/bin/env python3
"""bag_to_session.py 를 **ROS2 설치 없이** 그대로 실행하기 위한 어댑터.

왜 필요한가: 이 PC(노트북/분석용)에는 /opt/ros 가 없어 rosbag2_py·rclpy 를 import 할 수
없다. 그렇다고 변환 로직을 복사하면 원본과 갈라져 phase 코드표·/runs 스키마가 어긋난다.
→ bag 읽기에 필요한 **최소 API 만 가짜 모듈로 sys.modules 에 심고**, 실제 변환은
  launch/bag_to_session.py 의 main() 을 **수정 없이** 호출한다(단일 진실 원천 유지).

바꿔치기하는 것 (그 외 로직은 전부 원본):
  rosbag2_py          → 순수 파이썬 `rosbags` 패키지(mcap/sqlite3 자동 판별) 래퍼.
  rclpy.serialization → rosbags 의 지연 역직렬화(관심 토픽만 푼다 — 원본과 동일한 흐름).
  rosidl_runtime_py   → get_message 는 타입 문자열이 이미 data 에 실려 오므로 더미.
  real_deploy_inference_final → torch/SHM 이 필요해 import 불가. 실제 쓰이는
      resultant_from_tactile 은 원본 파일에서 **AST 로 뽑아 그대로 exec** 하고(복사 아님),
      Kinesthetic_Sensor_* 는 core/shm_common.py 에서 읽는다.
      USE_MN_SIDE_CHANNEL 은 원본과 같은 환경변수 규칙(기본 0 = SHM_raw).
  collect_ros2        → rclpy/moveit 의존이라 import 불가. 자세 코드표에 쓰이는
      ARM_POSES·GRIP_POSE_CANDIDATES 만 원본 소스에서 AST 로 **리터럴 그대로** 읽는다.

전제: pip install --user rosbags mcap h5py numpy pyyaml   (시스템 python3, ROS 불필요)
      metadata.yaml 이 유실된 bag 도 mcap 을 직접 스트리밍해 변환한다(아래 _open_mcap 참고).
실행:  python3 tools/bag_to_session_noros.py <세션폴더> [--rate 100] [--out ...] [--no-raw]
       (인자는 bag_to_session.py 와 동일하게 그대로 전달된다 — 상위 폴더 배치 변환과
        h5/json 만 있는 트리로 재구성하는 --out-root/--reuse-h5/--skip-existing 도 그대로)
       python3 tools/bag_to_session_noros.py ~/Desktop/collect_logs --out-root ~/Desktop/collect_h5
"""
from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_LAUNCH = _REPO / "stiffness_deploy_ros2" / "launch"


# ── 원본 소스에서 값 가져오기 (복사하지 않는다 — 원본이 바뀌면 같이 바뀌도록) ──────────
def _top_literal(py: Path, name: str):
    """모듈 최상위의 `name = <리터럴>` 을 찾아 값으로 돌려준다(함수 안 재대입은 무시)."""
    tree = ast.parse(py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"[shim] {py.name} 에서 최상위 {name} 을 찾지 못했습니다.")


def _top_func(py: Path, name: str, glb: dict):
    """모듈 최상위 함수 하나만 떼어내 exec — 무거운 import 없이 그 함수만 쓴다."""
    tree = ast.parse(py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = dict(glb)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(py), "exec"), ns)
            return ns[name]
    raise SystemExit(f"[shim] {py.name} 에서 최상위 함수 {name} 을 찾지 못했습니다.")


# ── 가짜 rosbag2_py ─────────────────────────────────────────────────────────────
class _StorageOptions:
    def __init__(self, uri="", storage_id="", **kw):
        self.uri, self.storage_id = uri, storage_id


class _ConverterOptions:
    def __init__(self, *a, **kw):
        pass


class _TopicMeta:
    def __init__(self, name, type_):
        self.name, self.type = name, type_


class _SequentialReader:
    """rosbag2_py.SequentialReader 중 bag_to_session 이 쓰는 4개 메서드만 흉내낸다.

    read_next() 의 두 번째 값(원본에선 CDR bytes)에 (reader, bytes, 타입문자열) 을 실어
    보내서, deserialize_message 가 상태 없이 풀 수 있게 한다. 원본과 마찬가지로
    **관심 있는 토픽만** 그때 역직렬화된다(paxini/ft 같은 미사용 토픽은 건드리지 않음).
    """

    def open(self, storage_options, converter_options):
        bag = Path(storage_options.uri)
        if (bag / "metadata.yaml").exists():
            self._open_rosbags(bag)
        else:
            # ★ metadata.yaml 이 없는 bag (복구된 폴더에서 실제로 발생). mcap 은 스키마·채널을
            #   파일 안에 갖고 있으므로 metadata 없이도 완전히 읽힌다 → mcap 을 직접 훑고
            #   역직렬화만 rosbags 표준 타입저장소로 한다(토픽 타입이 전부 std_msgs/sensor_msgs).
            mcaps = sorted(bag.glob("*.mcap"))
            if not mcaps:
                raise SystemExit(f"[shim] bag 에 metadata.yaml 도 *.mcap 도 없음: {bag}")
            print(f"[shim] metadata.yaml 없음 → mcap 직접 읽기 ({len(mcaps)}개 파일)")
            self._open_mcap(mcaps)

    def _open_rosbags(self, bag: Path):
        from rosbags.highlevel import AnyReader
        self._r = AnyReader([bag])
        self._r.open()
        self._types = {c.topic: c.msgtype for c in self._r.connections}
        self._it = iter(self._r.messages())          # log_time 오름차순 (rosbag2 와 동일)
        self._buf = next(self._it, None)

    def _open_mcap(self, mcaps):
        """mcap 을 **앞에서부터 통째로 스트리밍**해 읽는다(요약/인덱스 사용 안 함).

        ★ 인덱스를 안 쓰는 이유: metadata.yaml 이 없는 bag 은 대개 **꼬리가 잘려 있다**
          (복구 과정에서 유실). 그러면 파일 끝의 footer/summary 가 없어 인덱스 기반 읽기는
          첫 줄부터 실패한다. 앞에서부터 읽으면 잘린 지점까지의 메시지는 전부 살릴 수 있다.
        타입 정의도 mcap 안에 있지만, 토픽이 전부 std_msgs/sensor_msgs 표준이라 rosbags 의
        ROS2 Humble 타입저장소로 그대로 역직렬화한다.
        """
        from mcap.records import Channel, Message, Schema
        from mcap.stream_reader import StreamReader
        from rosbags.typesys import Stores, get_typestore
        self._r = types.SimpleNamespace(
            deserialize=get_typestore(Stores.ROS2_HUMBLE).deserialize_cdr)

        self._types = {}
        msgs, schemas, channels, truncated = [], {}, {}, None
        for p in mcaps:
            with p.open("rb") as fh:
                try:
                    for rec in StreamReader(fh).records:
                        if isinstance(rec, Schema):
                            schemas[rec.id] = rec.name
                        elif isinstance(rec, Channel):
                            channels[rec.id] = rec
                            self._types.setdefault(rec.topic, schemas.get(rec.schema_id, ""))
                        elif isinstance(rec, Message):
                            ch = channels[rec.channel_id]
                            msgs.append((rec.log_time, ch.topic,
                                         schemas.get(ch.schema_id, ""), rec.data))
                except Exception as e:                # 잘린 꼬리 — 여기까지가 온전한 데이터
                    truncated = f"{p.name}: {type(e).__name__}"
        if truncated:
            print(f"[shim] ⚠ mcap 이 중간에 끊겨 있음({truncated}) — 읽힌 {len(msgs)}개 메시지까지만 변환")
        msgs.sort(key=lambda m: m[0])                 # 인덱스가 없으니 log_time 으로 직접 정렬
        # rosbags 경로와 같은 (connection, 시각, bytes) 모양으로 맞춘다.
        self._it = ((types.SimpleNamespace(topic=t, msgtype=mt), ts, raw)
                    for ts, t, mt, raw in msgs)
        self._buf = next(self._it, None)

    def get_all_topics_and_types(self):
        return [_TopicMeta(t, ty) for t, ty in self._types.items()]

    def has_next(self):
        return self._buf is not None

    def read_next(self):
        conn, ts, raw = self._buf
        self._buf = next(self._it, None)
        return conn.topic, (self._r, raw, conn.msgtype), ts


def _deserialize_message(data, _cls):
    reader, raw, msgtype = data
    return reader.deserialize(raw, msgtype)


def _install_shims():
    rosbag2_py = types.ModuleType("rosbag2_py")
    rosbag2_py.SequentialReader = _SequentialReader
    rosbag2_py.StorageOptions = _StorageOptions
    rosbag2_py.ConverterOptions = _ConverterOptions

    rclpy = types.ModuleType("rclpy")
    rclpy.__path__ = []                                   # 패키지처럼 보이게
    serialization = types.ModuleType("rclpy.serialization")
    serialization.deserialize_message = _deserialize_message
    rclpy.serialization = serialization

    rosidl = types.ModuleType("rosidl_runtime_py")
    rosidl.__path__ = []
    utilities = types.ModuleType("rosidl_runtime_py.utilities")
    utilities.get_message = lambda type_str: type("Msg", (), {"__msgtype__": type_str})
    rosidl.utilities = utilities

    std_msgs = types.ModuleType("std_msgs")
    std_msgs.__path__ = []
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    for _n in ("String", "Int8", "Int32", "Bool", "Float32MultiArray", "Float64MultiArray"):
        setattr(std_msgs_msg, _n, type(_n, (), {}))
    std_msgs.msg = std_msgs_msg

    # real_deploy_inference_final: 실제로 쓰이는 3가지만 원본에서 끌어온다.
    sys.path.insert(0, str(_REPO / "stiffness_deploy_ros2"))
    try:
        from core.shm_common import Kinesthetic_Sensor_Num, Kinesthetic_Sensor_DOF
    except Exception:                                     # ctypes 전용 모듈이라 보통 성공
        Kinesthetic_Sensor_Num, Kinesthetic_Sensor_DOF = 4, 3
    RE = types.ModuleType("real_deploy_inference_final")
    RE.Kinesthetic_Sensor_Num = Kinesthetic_Sensor_Num
    RE.Kinesthetic_Sensor_DOF = Kinesthetic_Sensor_DOF
    RE.resultant_from_tactile = _top_func(
        _LAUNCH / "real_deploy_inference_final.py", "resultant_from_tactile", {"np": np})
    # 원본과 같은 규칙(기본 0). 이 값은 session.h5 의 kin_source attr 에만 쓰인다.
    RE.USE_MN_SIDE_CHANNEL = os.environ.get(
        "USE_MN_SIDE_CHANNEL", "0").strip().lower() in ("1", "true", "yes", "on")

    # collect_ros2: 자세 숫자코드(_pose_codes)용 목록만.
    CO = types.ModuleType("collect_ros2")
    CO.ARM_POSES = _top_literal(_LAUNCH / "collect_ros2.py", "ARM_POSES")
    CO.GRIP_POSE_CANDIDATES = _top_literal(_LAUNCH / "collect_ros2.py", "GRIP_POSE_CANDIDATES")
    #   _guess_names 의 폴백 경로(outcomes.json 이 없는 폴더)에서만 쓰인다. pose txt 는
    #   launch/ 에 있다.
    CO.D = types.SimpleNamespace(_POSE_DIR=_LAUNCH)

    sys.modules.update({
        "rosbag2_py": rosbag2_py,
        "rclpy": rclpy, "rclpy.serialization": serialization,
        "rosidl_runtime_py": rosidl, "rosidl_runtime_py.utilities": utilities,
        "std_msgs": std_msgs, "std_msgs.msg": std_msgs_msg,
        "real_deploy_inference_final": RE, "collect_ros2": CO,
    })


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    _install_shims()
    sys.path.insert(0, str(_LAUNCH))
    import bag_to_session                                  # noqa: E402  (shim 설치 후에만 가능)
    bag_to_session.main()                                  # 인자는 sys.argv 그대로


if __name__ == "__main__":
    main()
