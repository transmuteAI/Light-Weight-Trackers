## Designing Light-Weight Object Trackers
This is the official repository for the **ICASSP 2023** paper "On Designing Light-Weight Object Trackers Through Network Pruning: Use CNNs or Transformers?" by Saksham Aggarwal, Taneesh Gupta, Pawan K. Sahu, Arnav Chavan, Rishabh Tiwari, Dilip K. Prasad and Deepak K. Gupta

## Abstract
Object trackers deployed on low-power devices need to be light-weight, however, most of the current state-of-the-art (SOTA) methods rely on using compute-heavy backbones built using CNNs or Transformers. Large sizes of such models do not allow their deployment in low-power conditions and designing compressed variants of large tracking models is of great importance. This paper demonstrates how highly compressed light-weight object trackers can be designed using neural architectural pruning of large CNN and Transformer based trackers. Further, a comparative study on architectural choices best suited to design light-weight trackers is provided. A comparison between SOTA trackers using CNNs, Transformers as well as the combination of the two is presented to study their stability at various compression ratios. Finally results for extreme pruning scenarios going as low as 1% in some cases are shown to study the limits of network pruning in object tracking. This work provides deeper insights into designing highly efficient trackers from existing SOTA methods.

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
**Note:** We only train on GOT10k train dataset and evaluate on test data of GOT10k, LaSOT, OTB and TrackingNet

## Trackers
You can find tracker-specific codebase and its details below:
- [OSTrack](./OSTrack/README.md)
- [STARK](./Stark_sparse/README.md)
- [SuperDiMP](./Super_Dimp/README.md)

## Acknowledgments
* Thanks to the authors of [OSTrack](https://github.com/botaoye/OSTrack), [STARK](https://github.com/researchmm/Stark), [PyTracking](https://github.com/visionml/pytracking) and [VIT-Slim](https://github.com/Arnav0400/ViT-Slim), which helped us to quickly implement our ideas.
* We use the implementation of the ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repository.

## Citation
If our work is useful for your research, please consider cite:
```
Citation
```


