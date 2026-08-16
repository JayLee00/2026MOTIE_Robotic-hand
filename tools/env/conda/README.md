# conda 환경

실물 환경은 `~/miniconda3/envs/` 에 있다(디렉토리를 그대로 옮긴 뒤 prefix 를 보정했다).
여기에는 **재빌드용 스펙**과 사용법만 둔다.

| env | 파이썬 | 쓰는 곳 |
|---|---|---|
| `grasp_fruit` | 3.12 | 파지(seq 1)의 SAM3 검출 + top-down 파지점 계산 서브프로세스 |
| `molmo` | 3.11 | 내려놓기(seq 4) `molmo_service` :8810 |
| `sam3` | 3.12 | 내려놓기 `sam_service` :8811 |
| `anyplace_cu128` | 3.11 | 내려놓기 `anyplace_service` :8801, `igr_service` :8816, `grid_service` :8815 |

> **ROS2 노드는 conda python 으로 돌리지 않는다.** conda 가 번들한 libstdc++ 등이 시스템
> RMW/DDS 확장 모듈과 ABI 충돌해 종료 시 core dump 가 난다. rclpy 를 쓰는 모든 코드는
> `/usr/bin/python3`(3.10) 로 실행한다 — `tools/env/setup_env.sh` 가 conda 를 PATH 에서 걷어낸다.

## 검증

```bash
~/miniconda3/envs/anyplace_cu128/bin/python -c "import torch,torch_scatter,open3d,trimesh; print(torch.cuda.get_device_name(0))"
~/miniconda3/envs/sam3/bin/python  -c "import torch,sam3; print(sam3.__file__)"
~/miniconda3/envs/molmo/bin/python -c "import torch,transformers; print(transformers.__version__, torch.cuda.is_available())"
~/miniconda3/envs/grasp_fruit/bin/python -c "from transformers import Sam3Model; import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 재빌드 (환경이 깨졌거나 다른 PC 로 옮길 때)

`specs/` 에 env.yml + pip-freeze 가 있다. 공통 핀: **torch 2.7.1+cu128 / torchvision 0.22.1+cu128**
(sm_120 = RTX 5090 프리빌드, 재컴파일 불필요).

```bash
# 예: anyplace_cu128 (특수 index 2개 필요)
conda create -n anyplace_cu128 python=3.11.15 -y && conda activate anyplace_cu128
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install -r specs/anyplace_cu128.pip-freeze.txt
# sam3 / molmo 도 같은 패턴 (specs/*.pip-freeze.txt)
```

`grasp_fruit` 는 스펙이 skill 저장소 쪽에 있다:

```bash
cd skill-set/grasp && CONDA_BASE=$HOME/miniconda3 bash setup_pipeline_all.sh
```

## sam3 editable 재링크

`sam3` env 의 sam3 패키지는 **editable 설치**라 저장소 경로를 가리킨다. place 저장소를 옮기면:

```bash
~/miniconda3/envs/sam3/bin/pip install -e <새경로>/skill-set/place/sam3 --no-deps --no-build-isolation
```

## 환경 경로를 옮겼을 때 (prefix 보정)

raw 복사한 conda env 는 원본 prefix 문자열이 스크립트/메타데이터에 박혀 있다. 텍스트 파일만
치환하면 실행에는 문제가 없다(파이썬 바이너리는 자기 위치로 sys.prefix 를 계산한다):

```bash
grep -rIl '/OLD/PREFIX' ~/miniconda3/envs/<env> | xargs sed -i 's|/OLD/PREFIX|/NEW/PREFIX|g'
```
