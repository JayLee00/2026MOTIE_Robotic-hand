from setuptools import setup

package_name = "sequence_client"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="prime",
    maintainer_email="damilab.knu@gmail.com",
    description="시퀀스 제어권 클라이언트 (Start/End/하트비트/자동 체이닝) — GPU PC 팀용 라이브러리+예제",
    license="TODO",
    entry_points={
        "console_scripts": [
            "pick_sequence_example = sequence_client.pick_sequence_example:main",
            "run_sequence = sequence_client.run_sequence:main",
        ],
    },
)
