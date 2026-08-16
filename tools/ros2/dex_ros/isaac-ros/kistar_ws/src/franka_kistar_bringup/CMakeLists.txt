cmake_minimum_required(VERSION 3.8)
project(franka_kistar_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY launch resource test franka_kistar_bringup rviz urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/planning_scene_static_boxes.py
    scripts/trajectory_forwarder.py
    scripts/trajectory_subscriber.py
    scripts/pose_commander.py
    scripts/joint_state_merger.py
    scripts/trajectory_bridge.py
    scripts/arm_q_streamer.py
    scripts/ee_target_sender.py
    scripts/hand_target_sender.py
    scripts/hand_gui_bridge.py
    scripts/current_ee_pose.py
    # PR-1 new ROS-runnable scripts:
    scripts/joint_states_observer.py
    scripts/measure_launch_time.py
    # PR-1 / PR-2 helper scripts (lint + URDF snapshot regen). Installed as
    # PROGRAMS so they remain executable from the package install tree even when
    # operators only have the install/ side sourced. The DIRECTORY install above
    # additionally provides them under share/${PROJECT_NAME}/scripts/.
    scripts/lint_launch_args.sh
    scripts/regenerate_urdf.sh
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
