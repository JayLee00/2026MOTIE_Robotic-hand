# 시퀀스 제어권 상태 (Arbiter가 latched 토픽 /sequence_state로 상시 게시)
int32 IDLE=0
int32 RUNNING=1
int32 DONE=2

# 동작(시퀀스) 번호 배정표 (팀 합의, 2026-07). client_id도 동일 번호를 사용한다.
# 0은 "없음/소유자 없음" 예약.
int32 SEQ_PICK=1
int32 SEQ_INHAND=2
int32 SEQ_STIFFNESS=3
int32 SEQ_PLACE=4

int32 seq_id        # 현재(또는 마지막) 동작 번호, 0=없음
int32 state         # IDLE / RUNNING / DONE
int32 owner         # 제어권 소유 클라이언트 id, 0=없음
builtin_interfaces/Time stamp
