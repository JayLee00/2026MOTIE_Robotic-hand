# AnyGrasp for AnyPlace

First, install PyTorch. Note that the original [graspnet-baseline](https://github.com/graspnet/graspnet-baseline) have version 1.6.0, but it was unable to adapt for CUDA 11.x. So we chose PyTorch 1.10.1:
```sh
conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Install the required pip packages:
```sh
pip install -r requirements.txt
```

Clone [graspnet-baseline](https://github.com/graspnet/graspnet-baseline), and patch the repo:
```sh
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
git apply ../graspnet-baseline-patch.patch
```

Also clone [graspnetAPI](https://github.com/graspnet/graspnetAPI) **inside** the folder graspnet-baseline, and patch the repo:
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
git apply ../../graspnetAPI-patch.patch
cd ..
```

Then, follow the instructions shown on the original repo:

Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
cd graspnetAPI
pip install .
```

Finally, download either [checkpoint-rs.tar](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing) or [checkpoint-kn.tar](https://drive.google.com/file/d/1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX/view?usp=sharing) to the `grasp_detection` folder.
