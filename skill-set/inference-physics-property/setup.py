from setuptools import setup
from glob import glob

package_name = "stiffness_deploy_ros2"

setup(
    name=package_name,
    version="0.1.0",
    # deploy_ros2 의 sys.path 규약(‘..’=project_root, ‘.’=launch)을 유지하기 위해
    # 원본 Gen3 의 launch/ · core/ 레이아웃을 패키지 하위에 그대로 둔다.
    packages=[
        package_name,
        f"{package_name}.launch",
        f"{package_name}.core",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # 과일 파지 포즈 / 안전 포즈 txt + 과일별 임계값 yaml (FRUIT_CONFIG / fruit_thresholds 참조)
        (f"share/{package_name}/launch",
         glob(f"{package_name}/launch/*.txt") + glob(f"{package_name}/launch/*.yaml")),
    ],
    # 런타임에 Path(__file__) 기준으로 찾는 비-파이썬 자원 (colcon 설치 시 패키지 옆에 동봉).
    #   launch/*.txt = 포즈, launch/*.yaml = 과일별 임계값,
    #   models/*.pth = 강성 추론 체크포인트,
    #   labels/{,*/}*.yaml = 정규화 통계 (labels/general, labels/trial2 등 서브폴더 포함)
    package_data={
        package_name: ["launch/*.txt", "launch/*.yaml", "models/*.pth",
                       "labels/*.yaml", "labels/*/*.yaml"],
    },
    include_package_data=True,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yesol",
    maintainer_email="damilab.knu@gmail.com",
    description="deploy.py 로직을 Dual_Arm_Hand_Ctrl ROS2 토픽으로 실행하는 브리지",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # ros2 run stiffness_deploy_ros2 deploy_ros2  로도 실행 가능
            "deploy_ros2 = stiffness_deploy_ros2.launch.deploy_ros2:main",
            # 이미 파지한 상태에서 스퀴즈만 수행하는 task3 시퀀스
            "deploy_task3_ros2 = stiffness_deploy_ros2.launch.deploy_task3_ros2:main",
            # 배포 동일 경로 하이브리드 데이터 수집기 (HDF5 데모별 + 동시 rosbag)
            "collect_ros2 = stiffness_deploy_ros2.launch.collect_ros2:main",
            # rosbag → 데모별 HDF5 변환기 (Option 2a: 배포 샘플링 재생)
            "bag_to_hdf5 = stiffness_deploy_ros2.launch.bag_to_hdf5:main",
            # P5 parity 검증: Option 1 HDF5 ↔ bag→HDF5 데모·프레임 대조
            "verify_parity = stiffness_deploy_ros2.launch.verify_parity:main",
            # MoveIt 끝점 이동(Option B) plan-only 스모크 테스트: XYZ [QXYZW]
            "moveit_arm_mover = stiffness_deploy_ros2.launch.moveit_arm_mover:main",
            # 팔 관절각 캡처 → ARM_POSES 형식 출력 (teach & capture)
            "capture_pose = stiffness_deploy_ros2.launch.capture_pose:main",
            # MoveIt 팔 이동 테스트(기본 plan-only, --execute 로 실제 이동)
            "test_moveit_mover = stiffness_deploy_ros2.launch.test_moveit_mover:main",
        ],
    },
)
