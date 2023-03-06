# Broiler + Handy = BRODY

This is Broiler Weight Prediction Program which is based on 3D Computer Vision.
This program receives RGB data and depth data as input and outputs the average weight of broilers detected in the video.

## :heavy_check_mark: Tested

| Python | pytorch |  Windows   |   Mac   |   Linux  |
| :----: | :-----: | :--------: | :-----: | :------: |
| 3.8.0+ | 1.10.0+ | X | X |  Ubuntu 18.04 |


## :arrow_down: Installation

Clone repo and install [requirements.txt](https://github.com/dohyeonYoon/BRODY/blob/main/requirements.txt) in a
**Python>=3.8.0** environment, including
[**PyTorch>=1.10.0**](https://pytorch.org/get-started/locally/).

(tested environment cuda 11.3, cudnn 8.2.1 for cuda 11.x)

```bash
git clone https://github.com/dohyeonYoon/BRODY  # clone
cd BRODY
pip install -r requirements.txt  # install
```

## :rocket: Getting started

### You can inference with your own custom RGB, PGM file in /BRODY/src/input/rgb, /BRODY/src/input/depth folder.
```bash
cd src

python pyBRODY.py

```


## Demo of Weight Estimation on Broiler
![2022_05_26_09_59_32](https://user-images.githubusercontent.com/66056440/194740615-66a3d7cf-7b28-4177-8706-e849e5e4ab95.png)


### Pretrained Checkpoints

[weight](https://drive.google.com/file/d/1v4tO3tGXdqJa7pF2ERu9y8l0LN_HJrxx/view?usp=sharing)  # Mask-RCNN Weights

[Linear Model](https://drive.google.com/file/d/1wIKIskES8j602e_pYLqqJao1zJxZF_hU/view?usp=sharing)  # Huber Linear Regression Weights
