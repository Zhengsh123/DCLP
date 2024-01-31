# DCLP: Neural Architecture Predictor with Curriculum Contrastive Learning (AAAI24)

The implementation for DCLP: Neural Architecture Predictor with Curriculum Contrastive Learning (AAAI24)

# Requirements

```
python ==3.6
tensorflow==2.6.2
torch ==1.9.1
torch-cluster ==1.5.9
torch-geometric==2.0.3
torch-scatter ==2.0.7
torch-sparse ==0.6.10
torch-spline-conv ==1.2.1
torchvision ==0.10.0
```

# Data Preparation

We use NAS-Bench-101 and NAS-Bench-201 datasets, both of which are available in open-source projects. The corresponding links can be found at the end of this section, and the relevant configurations refer to the open-source project configuration. 

We get the **nasbench_only108.tfrecord** file of NAS-Bench-101 and **NAS-Bench-201-v1_1-096897.pth** file of NAS-Bnech-201 in. /dataset.

For the DARTS search space, there are two ways to obtain the labeled dataset, the first one is to start training from scratch, refer to . /train/generate_data.py; the second one is to extract from the training results of NAS-Bench-301. Place the **./darts** folder in nasbench301_full_data in . /dataset . **Note that we did not use the predicted results of NAS-Bench-301, but only its training results of some architectures on CIFAR-10.**

**NAS-Bench-101:** 

project links:https://github.com/google-research/nasbench

dataset links:https://storage.googleapis.com/nasbench/nasbench_full.tfrecord

**NAS-Bench-201:**

project links:https://github.com/D-X-Y/NAS-Bench-201

dataset links:https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view

**DARTS Search space:**

project links:https://github.com/automl/nasbench301

dataset links:https://figshare.com/articles/dataset/nasbench301_full_data/13286105

# Search

The three .sh files under the folder correspond to searching on three search spaces . The configuration of which can be modified accordingly. 

## Reference

```
@article{zheng2023dclp,
  title={DCLP: Neural Architecture Predictor with Curriculum Contrastive Learning},
  author={Zheng, Shenghe and Wang, Hongzhi and Mu, Tianyu},
  journal={arXiv preprint arXiv:2302.13020},
  year={2023}
}
```

