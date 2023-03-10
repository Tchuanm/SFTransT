

# SFTransT: Learning Spatial-Frequency Transformer for Visual Object Tracking
The implement of [_Learning Spatial-Frequency Transformer for Visual Object Tracking_](https://arxiv.org/abs/2208.08829)


<p align="center">
  <img width="85%" src="https://github.com/Tchuanm/SFTransT/blob/main/arch.png" alt="Framework"/>
</p>

## TL;DR
SFTransT follows the Siamese matching framework which takes the template and search frame as input. The Swin-Tiny network is adopted as the backbone, and the cross-scale features are fused as embedded features. Then, a Multi-Head Cross-Attention (MHCA) module is used to boost the interactions between the dual features. The output will be fed into our core component Spatial-Frequency Transformer, which models the Gaussian spatial prior and low-/high-frequency feature information simultaneously. More in detail, the GGN is adopted to predict the Gaussian spatial attention which will be added to the self-attention matrix. Then, the GPHA is designed to decompose them into low- and high-pass branches to achieve all-pass information propagation. Finally, the enhanced features will be fed into the classification and regression head for target object tracking. C , R , S represent the Concatenate, Rearrange, and Shift-and-Sum operators, respectively.


| GOT-10K (AO) |   Tracker   | LaSOT (AUC) | TrackingNet (AUC) | UAV123(AUC) | LaSOT-ext(AUC) | TNL2k(AUC) | WebUAV-3M |
|:------------:|:-----------:|:-----------:|:-----------------:|:-----------:|:--------------:|:----------:|-----------|
|     72.7     |  SFTransT   |    69.0     |       82.9        |    71.3     |      46.4      |    54.6    |   58.2    | 


## Installation
1. Create and activate a conda environment
```bash
conda create -n sftranst python=3.7
conda activate sftranst
```

2. Install the necessary packages. Please install them line by line to ensure the success.
```bash
conda install -c pytorch pytorch=1.5 torchvision=0.6.1 cudatoolkit=10.2
conda install matplotlib pandas tqdm
pip install opencv-python tb-nightly visdom scikit-image tikzplotlib gdown
conda install cython scipy
sudo apt-get install libturbojpeg
pip install pycocotools jpeg4py
pip install wget yacs
pip install shapely==1.6.4.post2 
pip install timm
pip install einops
```

3. Add the softlink of datasets into the path './dataset/'
```
     |--dataset
        |--got10k
        |--lasot
        |--trackingnet
        |--.......
```
   
4. Setup Environment.

```bash
# Environment settings for ltr. Saved at ltr/admin/local.py
cd SFTransT
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```


## Run training 

```bash
cd SFTransT/ltr
conda activate sftranst
export CUDA_VISIBLE_DEVICES=4,5,6,7
python run_training.py --train_module sftranst  --train_name sftranst_cfa_gpha_mlp  
```


## Test and Eval
1. For UAV, OTB， GOT10k

```bash
cd SFTransT/pysot_toolkit
conda activate sftranst
python eval_got10k_global.py --cuda 0  --begin 99 --end 100 --interval 1 --folds sftranst_cfa_gpha_mlp --subset test
```

2. For other datasets. 
python test_global.py --dataset LaSOT --cuda 5 --epoch 300  --win 0.50



## Acknowledgement

This is a combination version of the python tracking framework [PyTracking](https://github.com/visionml/pytracking) 
and [PySOT-Toolkit](https://github.com/StrangerZhang/pysot-toolkit).  
Thanks for the [TransT](https://github.com/chenxin-dlut/TransT/tree/main/pysot_toolkit) which firstly introduce the Transformer into visual tracking.


## Citation

```
@article{tang2022learning,
  title={Learning Spatial-Frequency Transformer for Visual Object Tracking},
  author={Tang, Chuanming and Wang, Xiao and Bai, Yuanchao and Wu, Zhe and Zhang, Jianlin and Huang, Yongmei},
  journal={arXiv preprint arXiv:2208.08829},
  year={2022}
}
```

or
```
@ARTICLE{tang2022learning,
  author={Tang, Chuanming and Wang, Xiao and Bai, Yuanchao and Wu, Zhe and Zhang, Jianlin and Huang, Yongmei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Learning Spatial-Frequency Transformer for Visual Object Tracking}, 
  year={2023},
  doi={10.1109/TCSVT.2023.3249468}}
```

