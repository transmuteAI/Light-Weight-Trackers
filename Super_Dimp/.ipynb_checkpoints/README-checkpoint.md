# SuperDiMP

## Install the environment
```
conda create -n pytracking python=3.8
conda activate pytracking
bash install.sh 
```

## Installation

#### Clone the GIT repository.  
```bash
git clone link_of_repo
```

#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path pytracking
```  
This script will also download the default networks and set-up the environment.  


## Set project paths
you can modify paths by editing these two files
```
ltr/admin/local.py  # paths about training
pytracking/evaluation/local.py  # paths about testing
```
Make sure you have activated ```pytracking``` environment before proceeding. 

## Pre-Training

## Sparsity Training

## Child Finetune

## Evaluation
