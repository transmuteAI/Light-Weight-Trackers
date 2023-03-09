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
    yaml_file = os.path.join(prj_dir, 'experiments/stark_child_no_clf/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    cfg.TEST.EPOCH = 83
#     params.checkpoint = os.path.join(save_dir, "checkpoints/train/stark_child/%s/STARKST_ep%04d.pth.tar" %
#                                      (yaml_name, cfg.TEST.EPOCH))
#     params.checkpoint = '/workspace/tracking_datasets/stark/child_exp1/checkpoints/train/stark_child/baseline_got10k_only_child_exp1/STARKST_ep0083.pth.tar'
    
    params.checkpoint = '/workspace/tracking_datasets/stark/exp4_child_no_clf/checkpoints/train/stark_child_no_clf/baseline_got10k_only_child_exp4/STARKST_ep0273.pth.tar'

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
