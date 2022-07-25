# Recaptured Screen Image Demoiréing in Raw Domain (RDNet)
This code is the official implementation of TMM paper "Recaptured Screen Image Demoiréing in Raw Domain".
This repository contains official implementation of TMM paper "Recaptured Screen Image Demoiréing in Raw Domain", by Huanjing Yue, Yijia Cheng, Yan Mao, Cong Cao, and Jingyu Yang.

<p align="center">
  <img width="800" src="https://github.com/tju-chengyijia/RDNet/blob/main/imgs/framework.png">
</p>

## Paper

未完待续<br>

## Demo Video

未完待续<br>

## Dataset

### Recaptured Screen Raw Image Demoiréing Dataset

<p align="center">
  <img width="600" src="https://github.com/tju-chengyijia/RDNet/blob/main/imgs/show_dataset.png">
</p>

You can download our dataset from [MEGA](https://mega.nz/file/4WMwiLiD#6HyQxZsUg-qgQ_L6eM5Nt5PiAIdrrmFLutS-tRoZ5XQ) or [Baidu Netdisk](https://pan.baidu.com/s/186tPHkRgr9eC9LpcRp59NA) (key: d6sz). We provide ground truth images and moiré images in Raw domain and sRGB domain respectively, which are placed in four folders gt_RAW_npz, gt_RGB, moire_RAW_npz and moire_RGB.

#### Copyright ####

The dataset is available for the academic purpose only. Any researcher who uses the dataset should obey the licence as below:

All of the Dataset are copyright by [Intelligent Imaging and Reconstruction Laboratory](http://tju.iirlab.org/doku.php), Tianjin University and published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License. This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.

## Code

### Dependencies and Installation

- Ubuntu 20.04.3 LTS
- Python 3.6.13
- NVIDIA GPU + CUDA 11.4 + CuDNN 11.3
- Pytorch-GPU 1.10.2

### Test

* Please download pre-trained model and test set.
* change the path in `test.py`.
* run: `python test.py`.

### Train

* Please download training set.
* change the path in `train.py`.
* run:  `python train.py`.

## Results

 <div align=center><img src="https://github.com/tju-chengyijia/RDNet/blob/main/imgs/SOTA_fig.png"></div><br>
Fig. 2. Comparison with state-of-the-art demoiréing methods. The three images are from natural, webpage, and document images. The screen and phone camera combination used in capturing the third image is excluded from the training set construction.<br>
<br>

## Citation

If you find our dataset or code helpful in your research or work, please cite our paper:

```
未完待续
```
