#!/usr/bin/env python3
"""범용 시퀀스 러너 — 자동 체이닝 데모/검증용.

seq_id > 1이면 latched /sequence_state에서 직전 시퀀스의 DONE을 기다린 뒤
시작한다(자동 체이닝). 실제 동작 대신 --work-sec 동안 sleep.

사용 예 (4개를 역순으로 띄워도 1→2→3→4 순서로 실행됨):
    ros2 run sequence_client run_sequence 4 &
    ros2 run sequence_client run_sequence 3 &
    ros2 run sequence_client run_sequence 2 &
    ros2 run sequence_client run_sequence 1 &

주의: 체인 전체를 다시 돌리기 전에는 arbiter를 재시작할 것
(이전 실행의 latched DONE이 남아 있으면 대기 없이 즉시 체이닝됨).

NOTE: 추후 오케스트레이터 노드가 이 자율 체이닝을 대체할 수 있음.
"""

import argparse
import sys
import time

import rclpy

from sequence_client import (ArbiterUnavailable, ControlDenied,
                             PreviousAborted, SequenceClient, SequenceError)

SEQ_NAMES = {1: "Pick", 2: "Inhand", 3: "Stiffness", 4: "Place"}


def main():
    parser = argparse.ArgumentParser(description="범용 시퀀스 러너 (자동 체이닝)")
    parser.add_argument("seq_id", type=int, choices=sorted(SEQ_NAMES),
                        help="시퀀스 번호 (1=Pick, 2=Inhand, 3=Stiffness, 4=Place)")
    parser.add_argument("--work-sec", type=float, default=3.0,
                        help="실제 동작을 대신하는 placeholder 시간 [s]")
    parser.add_argument("--wait-timeout", type=float, default=None,
                        help="직전 시퀀스 DONE 대기 제한 [s] (기본 무제한)")
    parser.add_argument("--client-id", type=int, default=0,
                        help="기본 0 = seq_id와 동일 (거부 경로 테스트용 오버라이드)")
    args = parser.parse_args()

    name = SEQ_NAMES[args.seq_id]
    rclpy.init()
    client = SequenceClient(args.seq_id, client_id=args.client_id)
    try:
        if args.seq_id > 1:
            print(f"sq{args.seq_id}({name}): sq{args.seq_id - 1} DONE 대기 중...")
            client.wait_for_previous_done(args.seq_id - 1,
                                          timeout=args.wait_timeout)
        with client:
            print(f"sq{args.seq_id}({name}) 시작(S) — {args.work_sec}s 작업")
            time.sleep(args.work_sec)
        print(f"sq{args.seq_id}({name}) 종료(E)")
    except PreviousAborted as e:
        print(f"sq{args.seq_id}({name}) 중단: 직전 시퀀스 실패 — {e}",
              file=sys.stderr)
        sys.exit(1)
    except ControlDenied as e:
        print(f"sq{args.seq_id}({name}) 중단: 제어권 거부 — {e}", file=sys.stderr)
        sys.exit(1)
    except ArbiterUnavailable as e:
        print(f"sq{args.seq_id}({name}) 중단: arbiter 없음 — {e}", file=sys.stderr)
        sys.exit(1)
    except (TimeoutError, SequenceError) as e:
        print(f"sq{args.seq_id}({name}) 중단: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        client.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
