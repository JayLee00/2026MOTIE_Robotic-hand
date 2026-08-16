#!/usr/bin/env bash
# Intel enp6s0 -> ROS2 LAN 프로파일 "ros2" 로 고정 (기존 jayfi 의 IP 192.168.0.1/24 유지)
# Realtek enp4s0 -> "프로파일 1" 고정 유지
# 실행: sudo bash /home/js/fix_ros_net.sh
set -e

# jayfi(또는 이미 ros2) 프로파일을 ros2 로 정리
SRC="jayfi"
nmcli -t -f NAME connection show | grep -qx "ros2" && SRC="ros2"

nmcli connection modify "$SRC" \
  connection.id "ros2" \
  connection.interface-name "enp6s0" \
  connection.autoconnect yes \
  connection.autoconnect-priority 100 \
  ipv4.method manual \
  ipv4.addresses "192.168.0.1/24" \
  ipv4.gateway "" \
  ipv4.never-default yes

# Realtek "프로파일 1" enp4s0 고정 보장
nmcli connection modify "프로파일 1" \
  connection.interface-name "enp4s0" \
  connection.autoconnect yes \
  connection.autoconnect-priority 100

# 적용 (enp6s0 만 잠깐 끊겼다 ros2 로 재연결, enp4s0/인터넷은 그대로)
nmcli connection up "ros2" ifname enp6s0

echo "=== 결과 ==="
nmcli -t -f DEVICE,STATE,CONNECTION device status | grep -E "enp4s0|enp6s0"
ip -br addr show enp4s0
ip -br addr show enp6s0
