cmake_minimum_required(VERSION 3.10)
project(dual_arm_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SequenceState.msg"
  "srv/RequestControl.srv"
  "srv/ReleaseControl.srv"
  DEPENDENCIES builtin_interfaces
)

ament_package()
