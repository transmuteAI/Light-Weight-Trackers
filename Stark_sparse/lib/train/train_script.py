import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.stark import build_starks, build_starkst,build_starkst_sparse,build_starkst_sparse_encoder
from lib.models.stark import build_stark_lightning_x_trt
# forward propagation related
from lib.train.actors import STARKSActor, STARKSTActor,STARKSPARSEActor
from lib.train.actors import STARKLightningXtrtActor
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')
    
    # update settings based on cfg
    update_settings(settings, cfg)

    
    settings.max_epochs = cfg.TRAIN.EPOCH
    settings.s = cfg.TRAIN.SPARSITY
    settings.trans_prune = cfg.TRAIN.TRANS_PRUNE
    settings.conv_prune = cfg.TRAIN.CONV_PRUNE
    settings.w1_enc = cfg.TRAIN.W1_ENC
    settings.w2_enc = cfg.TRAIN.W2_ENC
    
    settings.w1_dec = cfg.TRAIN.W1_DEC
    settings.w2_dec = cfg.TRAIN.W2_DEC
    
    settings.w3 = cfg.TRAIN.W3
    
    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

#     if "resnet_50_child" in cfg.MODEL.BACKBONE.TYPE:
#         cfg.ckpt_dir = f'/workspace/tracking_datasets/stark/cnn_child/budget_{cfg.MODEL.BACKBONE.BUDGET}'
#         print('*'*25,cfg.ckpt_dir,'*'*25)
    
    # Create network
    if settings.script_name == "stark_s":
        net = build_starks(cfg)
        
    elif settings.script_name == "stark_st1" or settings.script_name == "stark_st2":
        net = build_starkst(cfg)
        ckpt = torch.load('/workspace/tracking_datasets/STARKST_ep0050.pth.tar', map_location='cuda')['net']
        net.load_state_dict(ckpt, strict=True)
        print('loaded_pretrained')
    
    elif settings.script_name == "stark_sparse" or settings.script_name == "stark_sparse_no_clf":
        net = build_starkst_sparse(cfg)
#         ckpt = torch.load('/workspace/tracking_datasets/stark_sparse_pretrained_cnn_budget_10.pth')['net']
#         net.load_state_dict(ckpt, strict=False)
        print('loaded_pretrained')
          
    elif settings.script_name == "stark_child" or "stark_child_no_clf":
        net = build_starkst_sparse(cfg)
        settings.b_att = cfg.MODEL.BUDGET_ATTN
        settings.b_mlp = cfg.MODEL.BUDGET_MLP
        
        ckpt = torch.load('/workspace/tracking_datasets/stark/exp4/checkpoints/train/stark_sparse/baseline_got10k_only_sparse_exp4/STARKST_ep0026.pth.tar', map_location='cuda')['net']
        thresh_enc_attn, thresh_enc_mlp, thresh_dec_attn,thresh_dec_mlp = net.transformer.compress_sep_l(settings.b_att,settings.b_mlp,0.5)

    elif settings.script_name == "stark_lightning_X_trt":
        net = build_stark_lightning_x_trt(cfg, phase="train")
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "stark_s" or settings.script_name == "stark_st1":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
        
    elif settings.script_name == "stark_st2" or settings.script_name == "stark_child_clf":
        objective = {'cls': BCEWithLogitsLoss()}
        loss_weight = {'cls': 1.0}
        actor = STARKSTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
        
    elif settings.script_name == "stark_sparse":
        objective = {'giou': giou_loss, 'l1': l1_loss,'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT,'cls': cfg.TRAIN.CLF_WEIGHT}
        actor = STARKSPARSEActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    
    elif settings.script_name == "stark_child_no_clf":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
        
    elif settings.script_name == "stark_lightning_X_trt":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKLightningXtrtActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

#     trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True)
    # train process
    if settings.script_name in ["stark_st2", "stark_st2_plus_sp"]:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, load_previous_ckpt=True)
    else:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
      
