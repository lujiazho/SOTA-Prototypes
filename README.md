# SOTA-Prototypes

Simple prototypes of SOTAs with dummy data for better vision of model architecture itself.

## Environment
Other versions might work as well.
- Python 3.7.10
- Pytorch 1.11.0

## Prototypes - (Example Purpose)

#### 1. Classification Models

- 1-1. Vision Transformer (ViT)
  - Paper - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020)](https://arxiv.org/pdf/2010.11929)
  - Ipynb - [ViT.ipynb](1-1.ViT/ViT.ipynb)

- 1-2. Swin Transformer (Swin-T)
  - Paper - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (2021)](https://arxiv.org/pdf/2103.14030)
  - Ipynb - [Swin-T.ipynb](1-2.Swin-T/Swin-T.ipynb)

- 1-3. Resnet
  - Paper - [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/pdf/1512.03385)
  - Ipynb - [Resnet.ipynb](1-3.Resnet/Resnet.ipynb)

- 1-4. VGG
  - Paper - [Very Deep Convolutional Networks for Large-Scale Image Recognition (2014)](https://arxiv.org/pdf/1409.1556)
  - Ipynb - [VGG.ipynb](1-4.VGG/VGG.ipynb)


#### 2. Detection Models

- 2-1. Detection Transformer (DETR)
  - Paper - [End-to-End Object Detection with Transformers (2020)](https://arxiv.org/pdf/2005.12872)
  - Ipynb - [DETR.ipynb](2-1.DETR/DETR.ipynb)

- 2-2. Single Shot Detector (SSD)
  - Paper - [SSD: Single Shot MultiBox Detector (2015)](https://arxiv.org/pdf/1512.02325)
  - Ipynb - [SSD300.ipynb](2-2.SSD/SSD300.ipynb)


#### 3. Segmentation Models

- 3-1. U-Net
  - Paper - [U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/pdf/1505.04597)
  - Ipynb - [U-Net.ipynb](3-1.U-Net/U-Net.ipynb)


#### 4. Self-Supervised Models

- 4-1. Masked Autoencoder (MAE)
  - Paper - [Masked Autoencoders Are Scalable Vision Learners (2021)](https://arxiv.org/pdf/2111.06377)
  - Ipynb - [MAE.ipynb](4-1.MAE/MAE.ipynb)


#### 5. Unsupervised Models

- 5-1. Generative Adversarial Networks (GAN)
  - Paper - [Generative Adversarial Networks (2014)](https://arxiv.org/pdf/1406.2661)
  - Ipynb - [GAN.ipynb](5-1.GAN/GAN.ipynb)


#### 6. NLP Models

- 6-1. Transformer
  - Paper - [Attention Is All You Need (2017)](https://arxiv.org/pdf/1706.03762)
  - Ipynb - [Transformer.ipynb](6-1.Transformer/Transformer.ipynb)
  
  
## DistributedDataParallel - (Example Purpose)

- 0-1. Distributed Data Parallel (DDP)
  - Ipynb - [DDP1.0.ipynb](0-1.DDP/DDP1.0.ipynb) (Single Node Single GPU)
  - Ipynb - [DDP2.0.ipynb](0-1.DDP/DDP2.0.ipynb) (Multi Nodes Multi GPUs)
  
