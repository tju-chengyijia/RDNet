# Recaptured Screen Image Demoiréing in Raw Domain (RDNet)
This code is the official implementation of TMM paper "Recaptured Screen Image Demoiréing in Raw Domain".

Paper:<br>
--------
未完待续<br>

Environment:<br>
--------
Ubuntu 20.04.3 LTS + CUDA Version: 11.4 <br>
Python (version 3.6.13) + PyTorch (version 1.10.2+cu102) <br>

Network:<br>
-------
 <div align=center><img src="https://github.com/tju-chengyijia/RDNet/blob/main/imgs/framework.pdf"></div><br>
Fig. 1. The network structure of the proposed RDNet.<br>

Results:<br>
-------
 <div align=center><img src="https://github.com/tju-chengyijia/RDNet/blob/master/imgs/SOTA_fig.pdf"></div><br>
Fig. 2. Comparison with state-of-the-art demoiréing methods. The three images are from natural, webpage, and document images. The screen and phone camera combination used in capturing the third image is excluded from the training set construction.<br>
<br>

Download dataset:<br>
--------
`Training set and test set:` https://pan.baidu.com/s/186tPHkRgr9eC9LpcRp59NA (key:d6sz)<br>

Test:<br>
-------
* Please download pre-trained model and test set.
* change the path in `test.py`.
* run: `python test.py`.

Train:<br>
--------
* Please download training set.
* change the path in `train.py`.
* run:  `python train.py`.

Citation:<br>
-------
If you find this work useful for your research, please cite:<br>
```
未完待续
```
