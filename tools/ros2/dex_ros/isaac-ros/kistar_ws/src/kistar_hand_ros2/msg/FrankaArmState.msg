# Franka Arm 현재 상태 (shm -> ROS2)
# Arm_Num = 2 (Left, Right)
# Arm_DOF = 7

float64[7] joint_positions  # Arm_j_pos
float64[7] joint_torques    # Arm_j_tq
uint8 arm_id                # 0=Right, 1=Left

