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

You can download our dataset from [MEGA](https://mega.nz/file/4WMwiLiD#6HyQxZsUg-qgQ_L6eM5Nt5PiAIdrrmFLutS-tRoZ5XQ) or [Baidu Netdisk](https://pan.baidu.com/s/186tPHkRgr9eC9LpcRp59NA) (key: d6sz). 

We provide ground truth images and moiré images in raw domain and sRGB domain respectively, which are placed in four folders gt_RAW_npz, gt_RGB, moire_RAW_npz and moire_RGB. The ground truth raw image is actually pseudo ground truth. The users can regenerate them by utilizing other RGB to raw inversing algorithms. Our raw domain data is stored in npz format, including black level corrected data, black level value, white level value and white balance value.

The details of our dataset are shown in the following table.

| Path | Number of file | File format | Image resolution |
|  :----:  | :----:  |  :----:  | :----:  |
| /data/trainset/gt_RAW_npz | 63180 | npz | $256\times256$ |
| /data/trainset/gt_RGB | 63180 | png | $256\times256\times3$ |
| /data/trainset/moire_RAW_npz | 63180 | npz | $256\times256$ |
| /data/trainset/moire_RGB | 63180 | png | $256\times256\times3$ |
| /data/testset/gt_RAW_npz | 408 | npz | $512\times512$ |
| /data/testset/gt_RGB | 408 | png | $512\times512\times3$ |
| /data/testset/moire_RAW_npz | 408 | npz | $512\times512$ |
| /data/testset/moire_RGB | 408 | png | $512\times512\times3$ |

#### Copyright ####

The dataset is available for the academic purpose only. Any researcher who uses the dataset should obey the licence as below:

All of the Dataset are copyright by [Intelligent Imaging and Reconstruction Laboratory](http://tju.iirlab.org/doku.php), Tianjin University and published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License. This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.

## Code

### Dependencies and Installation

- Ubuntu 20.04.3 LTS
- Python 3.6.13
- NVIDIA GPU + CUDA 11.4 + CuDNN 11.3
- Pytorch-GPU 1.10.2

### Prepare

- Download our dataset and place them in data folder according to the given path.
- Download pre-trained model. All of our pre-trained model are placed in this RDNet project. We provide ISP model, classify model and demoiré model, which are placed in ISP_model folder, classify_model folder and Demoire_model folder.

### Test

- Test pretrained model on our testset.
```
python test.py --gpu_id 0 --num_worker 0 --save_test_dir ./out/
```

### Train

- Train the demoiréing module.
```
python train.py --gpu_id 0 --max_epoch 100 --num_worker 0 --patch_size 256 --batch_size 1
```

## Results

 <div align=center><img src="https://github.com/tju-chengyijia/RDNet/blob/main/imgs/sota_fig.png"></div><br>

## Citation

If you find our dataset or code helpful in your research or work, please cite our paper:

```
@article{yue2022recaptured,
  title={Recaptured Screen Image Demoiréing in Raw Domain},
  author={Yue, Huanjing and Cheng, Yijia and Mao, Yan and Cao, Cong and Yang, Jingyu},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```
## Acknowledgement

Our work and implementations are inspired by following projects:<br/>
[RViDeNet] (https://github.com/cao-cong/RViDeNet)<br/>
[AMNet] (https://github.com/tju-maoyan/AMNet)<br/>
[MBCNN-torch] (https://github.com/JunghyunKim242/MBCNN_torch_demoire)<br/>
