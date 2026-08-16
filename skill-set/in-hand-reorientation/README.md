# 실행 방법 

- shm, nd, 카메라 킬것 

# Terminal 0 - camera
```shell
ros2 launch franka_kistar_bringup realsense_front.launch.py 
```

# Terminal 1 - Activate MoveIt node 
```shell
ros2 launch franka_kistar_bringup dual_fr3_kistar_moveit.launch.py \
    joint_state_mode:=direct \
    robot_ip:=192.168.0.100 \
    use_rviz:=false
```

# Terminal 2 - Monitor stdate 2 
```shell
rs
/usr/bin/python3 scripts/inhand_sequence.py # --print-only # --once
```


# Terminal 3 - Fake Publisher (for develop) 
```shell
rs
# 체이닝 시뮬레이션: Pick DONE -> 2초 후 Inhand RUNNING
/usr/bin/python3 scripts/fake_sequence_publisher.py --sequence
```