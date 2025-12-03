# MS2MNet
MS2MNet:Multiscale State-Space Mixer Network For MultiModal Semantic Segmentation

<p align="center">
<img width="3976" height="2452" alt="Framework" src="https://github.com/DrWuHonglin/MS2MNet/blob/main/images/Framework.png" />
</p>

The main contributions of this work are summarized as follows:
1) We propose a novel multimodal semantic segmentation network, MS2MNet, featuring a dual-stream architecture and a multi-level feature fusion strategy. We introduce an efficient VFE module designed to efficiently capture and calibrate complementary information within multimodal fused features through multiscale feature
extraction, state-space modeling, and a channel attention mechanism.
2) We propose an Adaptive Spatial Attention Fusion (ASAF) module to fuse enhanced low-level semantic features from the encoder with high-level detail features
from the decoder. Additionally, we introduce a DSM Feature Enhancement (DFE)
module that enriches elevation data representations through gradient information augmentation.
3) We conducted extensive experiments on the ISPRS Vaihingen and Potsdam
datasets and the results show that our proposed network achieves better semantic
segmentation with lower complexity.

## Results

1. MS2MNet achieves competitive results on the following datasets:
- Vaihingen: 84.14% mIoU
- Potsdam: 86.21% mIoU
2. We provide visualizations of our results on the Vaihingen and Potsdam datasets:
<p align="center">
  <img src="https://github.com/DrWuHonglin/MS2MNet/blob/main/images/vaihingen_compare.png" width="800" height="450">
</p>
<p align="center">
  <img src="https://github.com/DrWuHonglin/MS2MNet/blob/main/images/potsdam_compare.png" width="800" height="450">
</p>

## Installation
1. Requirements
   
- Python 3.10.15	
- CUDA 12.1
- torch==1.13.0+cu117
- torchvision==0.14.0+cu117
- tqdm==4.66.4
- numpy==1.23.5
- pandas==2.0.1
- ipython==8.12.3

## Demo
To quickly test the MS2MNet with randomly generated tensors, you can run the demo.py file. This allows you to verify the model functionality without requiring a dataset.
1. Ensure that the required dependencies are installed:
```
pip install -r requirements.txt
```
2. Run the demo script:
```
python demo.py
```

## Datasets
All datasets including ISPRS Potsdam, ISPRS Vaihingen can be downloaded [here](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets).
