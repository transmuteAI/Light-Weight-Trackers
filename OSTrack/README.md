# OSTrack

## Install the environment
```
conda create -n ostrack python=3.8
conda activate ostrack
bash install.sh
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Pre-Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)  (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details). Path to the pretrained weights is to be specified in config file `MODEL.PRETRAIN_FILE`.

```
python tracking/train.py --name  experiment_name --script ostrack --config vitb_256_mae_ce_32x4_got10k_ep100 --save_dir /path/to/save/directory  --mode multiple --nproc_per_node 1 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/ostrack`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`. To use wandb, add auth key in config TRAIN.wandb_key which can be found [here](https://wandb.ai/authorize).

Pretrained OSTrack weights can be downloaded from [here](https://drive.google.com/drive/folders/1Vsd6tTUPl9G-mDUNzRas4CG_wEqrNam-?usp=share_link).

## Sparsity Training

Specify path to OSTrack pretrained weights in config file `MODEL.PRETRAIN_FILE`. Make sure to set `sparsity_train` in config file as True for sparsity training. W1 and W2 in config are tunable sparsity hyperparameters.
```
python tracking/train.py --name  experiment_name --script ostrack --config vitb_256_mae_ce_32x4_got10k_ep100_sparse --save_dir /path/to/save/directory  --mode multiple --nproc_per_node 1 --use_wandb 1 
```
Pretrained Sparsity trained weughts can be downloaded from [here](https://drive.google.com/drive/folders/1YjzD_oU5A6DA36leRnhl857XHD_FKrEW?usp=share_link).

## Child Finetune
Add path to sparsity trained weights in config file `MODEL.PRETRAIN_FILE`.
Specify Pruning type (naive or layerwise) and MLP and Attention budgets in config.
(Layerwise pruning is generally done in cases of extreme pruning i.e. 1% budget in our case.)

```
python tracking/train.py --name  experiment_name --script ostrack --child_train 1 --config vitb_256_mae_ce_32x4_got10k_ep100_sparse --save_dir /path/to/save/directory  --mode multiple --nproc_per_node 1 --use_wandb 1 
```

## Evaluation
Download the Child finetuned weights from [Google Drive](https://drive.google.com/drive/folders/1eG9a5dBef7FCMLwsT942fB6aTeDyDNHS?usp=share_link) 

Specify the weight path in config `TEST.PRETRAIN_FILE`.

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:

- OTB
```
python tracking/test.py  --name result_folder_name   ostrack --tracker_param vitb_256_mae_ce_32x4_got10k_ep100_sparse --dataset otb --threads 6 --num_gpus 1
python tracking/analysis_results.py --results_dir path/to/result/folder --dataset_name otb --config vitb_256_mae_ce_32x4_ep300_sparse
```

- LaSOT
```
python tracking/test.py  --name result_folder_name   ostrack --tracker_param vitb_256_mae_ce_32x4_got10k_ep100_sparse --dataset lasot --threads 6 --num_gpus 1
python tracking/analysis_results.py --results_dir path/to/result/folder --dataset_name lasot --config vitb_256_mae_ce_32x4_ep300_sparse
```
- GOT10K-test
```
python tracking/test.py  --name result_folder_name   ostrack --tracker_param vitb_256_mae_ce_32x4_got10k_ep100_sparse --dataset got10k_test --threads 6 --num_gpus 1
python lib/test/utils/transform_got10k.py --name result_folder_name --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep100_sparse
```
- TrackingNet
```
python tracking/test.py  --name result_folder_name   ostrack --tracker_param vitb_256_mae_ce_32x4_got10k_ep100_sparse --dataset trackingnet --threads 6 --num_gpus 1
python lib/test/utils/transform_trackingnet.py --name result_folder_name --tracker_name ostrack --cfg_name vitb_256_mae_ce_32x4_got10k_ep100_sparse
```




