# FPVT--Face Pyramid Vision Transformer
![arch](https://github.com/khawar-islam/FPVT_BMVC22/blob/main/fpvt.png)

## Usage Instructions

### 1. Preparation
Please install all dependencies  ```pip3 install -r requirement.txt```
```
pip3 install vit-pytorch
```
```
torch==1.8.1
torchvision==0.9.0+cu111
matplotlib==3.3.4
numpy==1.20.3
mxnet==1.8.0.post0
sklearn==0.0
scikit-learn==0.24.2
bcolz==1.2.1
pillow==8.2.0
ipython==7.22.0
scipy==1.6.3
opencv-python==4.5.1.48
tensorboardx==2.2
timm==0.3.2
ptflops==0.6.5
pyyaml==5.4.1
einops==0.3.0
pandas==1.3.1
```

### 2. Databases
You can download the training databases, faceScrub cleaned (version [FaceScrub](https://drive.google.com/file/d/1tr0fDodEk3CRaUhl9SlhC173IiWpGIUn/view?usp=sharing)), and put it in folder 'Data'. 

You can download the testing databases as follows and put them in folder 'eval'. 

- LFW: [Baidu Netdisk](https://pan.baidu.com/s/1WwFA1lS1_6elleu6kxMGDQ)(password: dfj0) 
- SLLFW: [Baidu Netdisk](https://pan.baidu.com/s/19lb0f9ZkAunKDpTzhJQUag)(password: l1z6)
- CALFW: [Baidu Netdisk](https://pan.baidu.com/s/1QyjRZNE0chm9BmobE2iOHQ)(password: vvqe)
- CPLFW: [Baidu Netdisk](https://pan.baidu.com/s/1ZmnIBu1IwBq6pPBGByxeyw)(password: jyp9)
- TALFW: [Baidu Netdisk](https://pan.baidu.com/s/1p-qhd2IdV9Gx6F6WaPhe5Q)(password: izrg) 
- CFP_FP: [Baidu Netdisk](https://pan.baidu.com/s/1lID0Oe9zE6RvlAdhtBlP1w)(password: 4fem)--refer to [Insightface](https://github.com/deepinsight/insightface/)
- AGEDB: [Baidu Netdisk](https://pan.baidu.com/s/1vf08K1C5CSF4w0YpF5KEww)(password: rlqf)--refer to [Insightface](https://github.com/deepinsight/insightface/)

## Citation
If you find this code useful for your research, please cite our work

```bash
@InProceedings{Khawar_BMVC22_FPVT,
      author = {Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood},
      title = {Face Pyramid Vision Transformer},
      booktitle = {Proceedings of the British Machine Vision Conference},
      year = {2022}
}
@inproceedings{islam2021face,
      title={Face Recognition Using Shallow Age-Invariant Data},
      author={Islam, Khawar and Lee, Sujin and Han, Dongil and Moon, Hyeonjoon},
      booktitle={2021 36th International Conference on Image and Vision Computing New Zealand (IVCNZ)},
      pages={1--6},
      year={2021},
      organization={IEEE}
}
```
## Contact
If you find any problem in code and want to ask any question, please send us email
```khawarr dot islam at gmail dot com```

## Acknowledgment
The code is mainly adopted from [Face Transformer](https://github.com/zhongyy/Face-Transformer), [Vision Transformer](https://github.com/lucidrains/vit-pytorch), and [DeiT](https://github.com/facebookresearch/deit).
