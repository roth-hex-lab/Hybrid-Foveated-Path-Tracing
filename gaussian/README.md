
# Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification


This is the official implementation of **Mini-Splatting2**, a point cloud reconstruction work in the context of Gaussian Splatting. Through aggressive Gaussian densification, our algorithm enables fast scene optimization within minutes. 
For technical details, please refer to:

**Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gaussian Densification**  <br />
Guangchi Fang and Bing Wang.<br />
**[[Paper](https://arxiv.org/pdf/2411.12788)]** <br />





<p align="center"> <img src="./assets/teaser.jpg" width="100%"> </p>



### (1) Setup
This code has been tested with Python 3.8, torch 1.12.1, CUDA 11.6.

- Clone the repository 
```
git clone git@github.com:fatPeter/mini-splatting2.git && cd mini-splatting2
```
- Setup python environment
```
conda create -n mini_splatting2 python=3.8
conda activate mini_splatting2
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

- Download datasets: [Mip-NeRF 360](https://jonbarron.info/mipnerf360/), [T&T+DB COLMAP](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).


### (2) Mini-Splatting2 (Sparse Gaussians)

Training scripts for Mini-Splatting2 are in `msv2`:
```
cd msv2
```
- Train with train/test split:
```
# mipnerf360 outdoor
python train.py -s <dataset path> -m <model path> -i images_4 --eval --imp_metric outdoor --config_path ../config/fast
# mipnerf360 indoor
python train.py -s <dataset path> -m <model path> -i images_2 --eval --imp_metric indoor --config_path ../config/fast
# t&t
python train.py -s <dataset path> -m <model path> --eval --imp_metric outdoor --config_path ../config/fast
# db
python train.py -s <dataset path> -m <model path> --eval --imp_metric indoor --config_path ../config/fast
```

- Modified full_eval script:
```
python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>
```


### (3) Mini-Splatting2-D (Dense Gaussians)

Training scripts for Mini-Splatting2-D are in `msv2_d`:
```
cd msv2_d
```
- Train with train/test split:
```
# mipnerf360 outdoor
python train.py -s <dataset path> -m <model path> -i images_4 --eval --imp_metric outdoor --config_path ../config/fast
# mipnerf360 indoor
python train.py -s <dataset path> -m <model path> -i images_2 --eval --imp_metric indoor --config_path ../config/fast
# t&t
python train.py -s <dataset path> -m <model path> --eval --imp_metric outdoor --config_path ../config/fast
# db
python train.py -s <dataset path> -m <model path> --eval --imp_metric indoor --config_path ../config/fast
```

- Modified full_eval script:
```
python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>
```


### (4) Dense Point Cloud Reconstruction

This implementation directly support dense point cloud reconstruction:

```
# similar to train.py (-i images_4/images_2, --imp_metric outdoor/indoor)
# output ply files are saved in ./teaser
python teaser.py -s <dataset path> -m <model path> -i images_4 --eval --imp_metric outdoor
```







**Acknowledgement.** This project is built upon [Mini-Splatting](https://github.com/fatPeter/mini-splatting), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs).








