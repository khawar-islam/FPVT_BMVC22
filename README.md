```
### 2. Databases
You can download the training databases, MS-Celeb-1M (version [ms1m-retinaface](https://github.com/deepinsight/insightface/tree/master/challenges/iccv19-lfr)), and put it in folder 'Data'. 

You can download the testing databases as follows and put them in folder 'eval'. 

- LFW: [Baidu Netdisk](https://pan.baidu.com/s/1WwFA1lS1_6elleu6kxMGDQ)(password: dfj0) 
- SLLFW: [Baidu Netdisk](https://pan.baidu.com/s/19lb0f9ZkAunKDpTzhJQUag)(password: l1z6)
- CALFW: [Baidu Netdisk](https://pan.baidu.com/s/1QyjRZNE0chm9BmobE2iOHQ)(password: vvqe)
- CPLFW: [Baidu Netdisk](https://pan.baidu.com/s/1ZmnIBu1IwBq6pPBGByxeyw)(password: jyp9)
- TALFW: [Baidu Netdisk](https://pan.baidu.com/s/1p-qhd2IdV9Gx6F6WaPhe5Q)(password: izrg) 
- CFP_FP: [Baidu Netdisk](https://pan.baidu.com/s/1lID0Oe9zE6RvlAdhtBlP1w)(password: 4fem)--refer to [Insightface](https://github.com/deepinsight/insightface/)
- AGEDB: [Baidu Netdisk](https://pan.baidu.com/s/1vf08K1C5CSF4w0YpF5KEww)(password: rlqf)--refer to [Insightface](https://github.com/deepinsight/insightface/)



### 3. Train Models

- ViT-P8S8
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VIT -head CosFace --outdir ./results/ViT-P8S8_ms1m_cosface_s1 --warmup-epochs 1 --lr 3e-4 

CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VIT -head CosFace --outdir ./results/ViT-P8S8_ms1m_cosface_s2 --warmup-epochs 0 --lr 1e-4 -r path_to_model 

CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VIT -head CosFace --outdir ./results/ViT-P8S8_ms1m_cosface_s3 --warmup-epochs 0 --lr 5e-5 -r path_to_model 
```

- ViT-P12S8
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VITs -head CosFace --outdir ./results/ViT-P12S8_ms1m_cosface_s1 --warmup-epochs 1 --lr 3e-4 

CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VITs -head CosFace --outdir ./results/ViT-P12S8_ms1m_cosface_s2 --warmup-epochs 0 --lr 1e-4 -r path_to_model 

CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -u train.py -b 480 -w 0,1,2,3 -d retina -n VITs -head CosFace --outdir ./results/ViT-P12S8_ms1m_cosface_s3 --warmup-epochs 0 --lr 5e-5 -r path_to_model 
```

### 4. Pretrained Models and Test Models (on LFW, SLLFW, CALFW, CPLFW, TALFW, CFP_FP, AGEDB)
You can download the following models
- ViT-P8S8: [Baidu Netdisk](https://pan.baidu.com/s/1ppgQe1GG3oa2-uz2zzL6EQ)(password: spkf)
- ViT-P12S8: [Baidu Netdisk](https://pan.baidu.com/s/1VrDfvz4SvYVnPcTlHVKAkg)(password: 7caa)

You can test Models
```
python test.py --model ./results/ViT-P12S8_ms1m_cosface/Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth --network VIT 

python test.py --model ./results/ViT-P12S8_ms1m_cosface/Backbone_VITs_Epoch_2_Batch_12000_Time_2021-03-17-04-05_checkpoint.pth --network VITs 
```


