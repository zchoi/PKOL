## Video Question Answering with Prior Knowledge and Object-sensitive Learning (PKOL)

[![](https://img.shields.io/badge/python-3.7.11-orange.svg)](https://www.python.org/)  [![](https://img.shields.io/apm/l/vim-mode.svg)](https://github.com/zchoi/S2-Transformer/blob/main/LICENSE)  [![](https://img.shields.io/badge/Pytorch-1.7.1-red.svg)](https://pytorch.org/)

This is the official code implementation for the paper：

[Video Question Answering with Prior Knowledge and Object-sensitive Learning]()

<p align="center">
  <img src="framework.jpg" alt="Relationship-Sensitive Transformer" width="850"/>
</p>


## Table of Contents

- [Setups](#Setups)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reference and Citation](#reference-and-citation)
- [Acknowledgements](#acknowledgements)

## Setups

- **Ubuntu** 20.04
- **CUDA** 11.5
- **Python** 3.7
- **PyTorch** 1.7.0 + cu110

1. Clone this repository：

```
conda create -n hcrn_videoqa python=3.6
conda activate hcrn_videoqa
conda install -c conda-forge ffmpeg
conda install -c conda-forge scikit-video
pip install -r requirements.txt
```

2. Install dependencies：

```
conda create -n vqa python=3.7
conda activate vqa
pip install -r requirements.txt
```
## Data Preparation

- ### Text Features

  Download pre-extracted text features from [here](), and place it into `data/{dataset}-qa/` for MSVD-QA, MSRVTT-QA and `data/tgif-qa/{question_type}/` for TGIF-QA, respectively.

- ### Visual Features

  Download pre-extracted visual features (i.e., appearance, motion, object) from [here](), and place it into `data/{dataset}-qa/` for MSVD-QA, MSRVTT-QA and `data/tgif-qa/{question_type}/` for TGIF-QA, respectively.

> **Note:** The object features are huge, (especially ~700GB for TGIF-QA), please be cautious of disk space when downloading.

## Experiments

- ###  For MSVD-QA and MSRVTT-QA：

<u>Training</u>：

```
python train_iterative.py --cfg configs/msvd_qa.yml
```
<u>Evaluation</u>：

```
python validate_iterative.py --cfg configs/msvd_qa.yml
```

- ###  For TGIF-QA：

  Choose a suitable config file in `configs/{task}.yml` for one of 4 tasks: `action, transition, count, frameqa` to train/val the model. For example, to train with action task, run the following command:

<u>Training</u>：

```
python train_iterative.py --cfg configs/tgif_qa_action.yml
```

<u>Evaluation</u>：

```
python validate_iterative.py --cfg configs/tgif_qa_action.yml
```
## Results

Performance on MSVD-QA and MSRVTT-QA datasets:

| Model   | MSVD-QA | MSRVTT-QA |
|:----------  |:-------:  |:-:  |
| PKOL |    41.1    | 36.9 |

Performance on TGIF-QA dataset:

| Model | Count ↓ | FrameQA ↑ | Trans. ↑ | Action ↑ |
| :---- | :-----: | :-------: | :------: | :------: |
| PKOL  |  3.67   |   61.8    |   82.8   |   74.6   |

## Reference
[1] Le, Thao Minh, et al. "Hierarchical conditional relation networks for video question answering." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

## Citation
```
@inproceedings{PKOL,
  author    = {Pengpeng Zeng and
               Haonan Zhang and
               Lianli Gao and
               Jingkuan Song and 
               Heng Tao Shen
               },
  title     = {Video Question Answering with Prior Knowledge and Object-sensitive Learning},
  booktitle = {TIP},
  % pages     = {????--????}
  year      = {2022}
}
```
## Acknowledgements
Our code implementation is based on this [repo](https://github.com/thaolmk54/hcrn-videoqa).
