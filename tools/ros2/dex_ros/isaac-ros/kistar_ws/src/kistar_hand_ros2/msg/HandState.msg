# Hand 현재 상태 (shm -> ROS2)
# Hand_Num = 2 (Left, Right)
# Hand_DOF = 16
# Kinesthetic_Sensor_DATA_NUM = 12 (4*3)
# Tactile_Sensor_DATA_NUM = 60 (4*15)

int16[16] joint_positions      # Hand_j_pos
int16[12] kinesthetic_sensors  # Hand_j_kin
int16[60] tactile_sensors      # Hand_j_tac
uint8 hand_id                   # 0=Right, 1=Left

