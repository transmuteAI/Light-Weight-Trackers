## Mini Stark Training
```
python tracking/train.py --script stark_st1 --config baseline_got10k_new_config --save_dir --mode multiple --nproc_per_node 8 --master_port 45555
```

## Sparsity Training
```
python tracking/train.py --script stark_sparse --config baseline_got10k_only_sparse --save_dir --mode multiple --nproc_per_node 8 --master_port 45555
```

## Child Training

 - **Budget 10**
   ```
   python tracking/train.py --script stark_child_no_clf --config baseline_got10k_only_child_budget_10 --save_dir --mode multiple --nproc_per_node 8 --master_port 45555
   ```

- **Budget 25**
   ```
   python tracking/train.py --script stark_child_no_clf --config baseline_got10k_only_child_budget_25 --save_dir --mode multiple --nproc_per_node 8 --master_port 45555
   ```

- **Budget 50**
   ```
   python tracking/train.py --script stark_child_no_clf --config baseline_got10k_only_child_budget_50 --save_dir --mode multiple --nproc_per_node 8 --master_port 45555
   ```

- **Budget 75**
   ```
   python tracking/train.py --script stark_child_no_clf --config baseline_got10k_only_child_budget_75 --save_dir --mode multiple --nproc_per_node 8 --master_port 45555
   ```

## Evaluation
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1ai63mKR47Z8s-yI6rKeDfB4-HdpvX-x5?usp=sharing) 
Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/stark`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- **LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)**

 - **Mini-Stark**
   ```
   python tracking/test.py stark_st baseline_got10k_new_config --dataset lasot --threads 16
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```

 - **Compressed Mini-Stark (Budget10)**
   ```
   python tracking/test.py stark_st baseline_got10k_budget_10 --dataset lasot --threads 16
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```

 - **Compressed Mini-Stark (Budget25)**
   ```
   python tracking/test.py stark_st baseline_got10k_budget_25 --dataset lasot --threads 16
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```

 - **Compressed Mini-Stark (Budget50)**
   ```
   python tracking/test.py stark_st baseline_got10k_budget_50 --dataset lasot --threads 16
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```

 - **Compressed Mini-Stark (Budget75)**
   ```
   python tracking/test.py stark_st baseline_got10k_budget_75 --dataset lasot --threads 16
   python tracking/analysis_results.py # need to modify tracker configs and names
   ```

- **GOT10K-test**

  - **Mini-Stark**
    ```
    python tracking/test.py stark_st baseline_got10k_new_config --dataset got10k_test --thread 16
    python lib/test/utils/transform_got10k.py --tracker_name stark_st --cfg_name baseline_got10k_new_config
    ```

  - **Compressed Mini-Stark (Budget10)**
    ```
    python tracking/test.py stark_st baseline_got10k_budget_10 --dataset got10k_test --thread 32
    python lib/test/utils/transform_got10k.py --tracker_name stark_st --cfg_name baseline_got10k_budget_10
    ```

  - **Compressed Mini-Stark (Budget25)**
    ```
    python tracking/test.py stark_st baseline_got10k_budget_25 --dataset got10k_test --thread 32
    python lib/test/utils/transform_got10k.py --tracker_name stark_st --cfg_name baseline_got10k_budget_25
    ```

  - **Compressed Mini-Stark (Budget50)**
    ```
    python tracking/test.py stark_st baseline_got10k_budget_50 --dataset got10k_test --thread 32
    python lib/test/utils/transform_got10k.py --tracker_name stark_st --cfg_name baseline_got10k_budget_50
    ```

  - **Compressed Mini-Stark (Budget75)**
    ```
    python tracking/test.py stark_st baseline_got10k_budget_75 --dataset got10k_test --thread 32
    python lib/test/utils/transform_got10k.py --tracker_name stark_st --cfg_name baseline_got10k_budget_75
    ```

- **TrackingNet**
   
   - **Mini-Stark**
     ```
     python tracking/test.py stark_st baseline_got10k_new_config --dataset got10k_test --thread 16
     python lib/test/utils/transform_got10k.py --tracker_name stark_st --cfg_name baseline_got10k_new_config
     ```
   - **Compressed Mini-Stark (Budget10)**
     ```
     python tracking/test.py stark_st baseline_got10k_budget_10 --dataset got10k_test --thread 32
     python lib/test/utils/transform_trackingnet.py --tracker_name stark_st --cfg_name baseline_got10k_budget_10
     ```
   - **Compressed Mini-Stark (Budget25)**
     ```
     python tracking/test.py stark_st baseline_got10k_budget_25 --dataset got10k_test --thread 32
     python lib/test/utils/transform_trackingnet.py --tracker_name stark_st --cfg_name baseline_got10k_budget_25
     ```
   - **Compressed Mini-Stark (Budget50)**
     ```
     python tracking/test.py stark_st baseline_got10k_budget_50 --dataset trackingnet --thread 32
     python lib/test/utils/transform_trackingnet.py --tracker_name stark_st --cfg_name baseline_got10k_budget_50
     ```
   - **Compressed Mini-Stark (Budget75)**
     ```
     python tracking/test.py stark_st baseline_got10k_budget_75 --dataset trackingnet --thread 32
     python lib/test/utils/transform_trackingnet.py --tracker_name stark_st --cfg_name baseline_got10k_budget_75
     ```
    
