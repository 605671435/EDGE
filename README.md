# EDGE: Edge Distillation and Gap Elimination for Heterogeneous Networks in 3D Medical Image Segmentation

A pytorch implement for 'EDGE: Edge Distillation and Gap Elimination for Heterogeneous Networks in 3D Medical Image Segmentation'

# How to use

## Requirement
- Pytorch >= 1.12.0
- MMCV
- MMEngine
- MMSeg
- MMRazor
- MONAI

This repo is required the MMCV, it can be simply install by:

```pycon
pip install -U openmim
mim install mmcv
```

We have a lot of required package listed in [here](requirements.txt).

You can simply run the following script to install the required package:

```pycon
pip install -r requirements.txt
```

## Datasets

### BTCV

You counld get BTCV raw data from [here](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752 "https://www.synapse.org/#!Synapse:syn3193805/wiki/217752").

### WORD
You counld get WORD raw data from this [repo](https://github.com/HiLab-git/LCOVNet-and-KD?tab=readme-ov-file#dataset).

Put datasets file into PROJECT/data.

## Training and testing

### Get Checkpoints

After the paper is accepted, we will publish the weights of our pre-trained teachers.

### Commands
-   Training on one GPU:

```pycon
python train.py {config}
```

-   Testing on one GPU:

```pycon
python test.py {config}
```
{config} means the config path. The config path can be found in [configs](configs/distill/msftd "configs").

# Acknowledgement

Specially thanks to the following: 
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation "MMSegmentation")
- [MMEngine](https://github.com/open-mmlab/mmengine "MMEngine")
- [Monai](https://github.com/Project-MONAI)

# Citation

```bash
@misc{mmseg2020,
  title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
  author={MMSegmentation Contributors},
  howpublished = {\url{[https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)}},
  year={2020}
}
```

```bash
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year={2022}
}
```

# License

This project is released under the [Apache 2.0 license](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE "Apache 2.0 license").
