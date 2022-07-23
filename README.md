# Recaptured Screen Image Demoiréing in Raw Domain (RDNet)
This code is the official implementation of TMM paper "Recaptured Screen Image Demoiréing in Raw Domain".

Paper:<br>
--------
https://ieeexplore.ieee.org/abstract/document/8972378<br>

Environment:<br>
--------
Ubuntu 20.04.3 LTS <br>
CUDA Version: 11.4 <br>
Python (version 3.6.13) + PyTorch (version 1.10.2+cu102) <br>

Network:<br>
-------
 <div align=center><img src="https://github.com/tju-chengyijia/RDNet/blob/master/imgs/framework.pdf"></div><br>
Fig. 1. The architecture of our RDNet.<br>

Results:<br>
-------
 <div align=center><img src="https://github.com/tju-chengyijia/RDNet/blob/master/imgs/SOTA_fig.pdf"></div><br>
Fig. 2. The recaptured screen images (top row), our demoiréing results (the second row), and the corresponding screenshot images (bottom row). Please zoom in the figure for better observation.<br>
<br>

Download dataset:<br>
--------
`Training set:` https://pan.baidu.com/s/1Xn5YygDb9Eg5u5zL3plrsA (key:gpxd)<br>
`Test set:` https://pan.baidu.com/s/1KCZciRYb-MP16u4W1w3X0Q (key:isn6)<br>

Test:<br>
-------
* Please download pre-trained model and test set.
* change the path in `test.py`.
* run: `python test.py`.

Train:<br>
--------
* Please download training set.
* change the path in `main.py`.
* run:  `python main.py`.

Citation:<br>
-------
If you find this work useful for your research, please cite:<br>
```
@article{Yue2020Recaptured,
	author = {Yue, Huanjing and Mao, Yan and Liang, Lipu and Xu, Hongteng and Hou, Chunping and Yang, Jingyu},
	year = {2021},
        title = {Recaptured Screen Image Demoir\'eing},
	volume={31},
	number={1},
	pages={49-60},
	journal = {IEEE Transactions on Circuits and Systems for Video Technology},
	doi = {10.1109/TCSVT.2020.2969984}
}
```

Contactor:<br>
----------
If you have any question, please concat me with maoyan@tju.edu.cn.
