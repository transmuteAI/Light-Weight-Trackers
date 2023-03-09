from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.stark_st2.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    prj_dir = '/workspace/Stark_sparse'
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/stark_st2/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
#     cfg.ckpt_dir = f'/workspace/tracking_datasets/stark/cnn_child/budget_{cfg.MODEL.BACKBONE.BUDGET}'
    params.cfg = cfg
#     print('*'*25,cfg.ckpt_dir,'*'*25)
#     print("test config: ", cfg)
    
    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
#     params.checkpoint = os.path.join(save_dir, "checkpoints/train/stark_st2/%s/STARKST_ep%04d.pth.tar" %
#                                      (yaml_name, cfg.TEST.EPOCH))
#     params.checkpoint = '/workspace/tracking_datasets/stark/child_exp1/checkpoints/train/stark_child/baseline_got10k_only_child_exp1/STARKST_ep0083.pth.tar'

#     params.checkpoint = '/workspace/tracking_datasets/stark/exp4_child_no_clf/checkpoints/train/stark_child_no_clf/baseline_got10k_only_child_exp4/STARKST_ep0273.pth.tar'
    
#     params.checkpoint = '/workspace/tracking_datasets/stark/exp1_child_clf/checkpoints/train/stark_child_clf/baseline_got10k_only_child_clf_exp1/STARKST_ep0017.pth.tar'
#     params.checkpoint = '/workspace/tracking_datasets/stark/exp4_child_comb_clf_and_reg/checkpoints/train/stark_child/baseline_got10k_only_child_exp4/STARKST_ep0328.pth.tar'

    
#     params.checkpoint = '/workspace/tracking_datasets/stark/exp_wacv_child_fin1_bud_75_no_clf/checkpoints/train/stark_child_no_clf/baseline_got10k_only_child_exp1_no_clf_budget_75/STARKST_ep0287.pth.tar'
    
    params.checkpoint = '/workspace/tracking_datasets/stark/sparse_exp_cnn_budget_10_child_new/checkpoints/train/stark_child_no_clf/baseline_got10k_only_child_exp1_no_clf_budget_10/STARKST_ep0260.pth.tar'
    
    
#     params.checkpoint = '/workspace/tracking_datasets/stark/exp_wacv_child_fin1_bud_25_no_clf/checkpoints/train/stark_child_no_clf/baseline_got10k_only_child_exp1_no_clf_budget_25/STARKST_ep0283.pth.tar'

#     params.checkpoint = '/workspace/tracking_datasets/stark/wacv_fin_sparse_exp1_no_clf/checkpoints/train/stark_sparse_no_clf/baseline_got10k_only_sparse_exp1_no_clf/STARKST_ep0087.pth.tar'

#     params.checkpoint = '/workspace/tracking_datasets/stark/wacv_fin_exp1_child_no_clf/checkpoints/train/stark_child_no_clf/baseline_got10k_only_child_exp1_no_clf/STARKST_ep0287.pth.tar'
    
#     params.checkpoint = '/workspace/tracking_datasets/stark/exp_wacv_child_fin2/checkpoints/train/stark_sparse_no_clf/baseline_got10k_only_sparse_exp2_no_clf/STARKST_ep0082.pth.tar'
    
#     params.checkpoint = '/workspace/tracking_datasets/stark/child_new_exp_cnn_budget_50/checkpoints/train/stark_child_no_clf/baseline_got10k_only_sparse_no_clf_cnn_budget_50/STARKST_ep0161.pth.tar'
    
#     print(params.checkpoint)
    # whether to save boxes from all queries
    
#     params.checkpoint = 
    params.save_all_boxes = False

    return params
