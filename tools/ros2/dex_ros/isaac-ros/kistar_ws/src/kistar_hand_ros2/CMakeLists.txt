cmake_minimum_required(VERSION 3.8)
project(kistar_hand_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 기본 ament + 메시지 생성기
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# FrankaArmState.msg 에서 사용하는 메시지 타입이 있으면 여기에 추가
# 예: std_msgs/Header, sensor_msgs/JointState 등을 쓰는 경우
# find_package(std_msgs REQUIRED)
# find_package(sensor_msgs REQUIRED)

# 메시지 인터페이스 생성
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrankaArmState.msg"
  "msg/FrankaArmTarget.msg"
  "msg/HandState.msg"
  "msg/HandTarget.msg"
)

ament_export_dependencies(rosidl_default_runtime)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
