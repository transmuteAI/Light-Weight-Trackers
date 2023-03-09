import os
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
import torch.nn as nn
from torch.cuda.amp import GradScaler
import wandb
from tqdm import tqdm

class searchloss(torch.nn.Module):
    def __init__(self, settings, device):
        super().__init__()
#         self.base_criterion = base_criterion
        self.device = device
        self.settings = settings

    def forward(self, inputs, actor):
        base_loss, stats = actor(inputs)
        
        sparsity_loss_attn_enc, sparsity_loss_mlp_enc,sparsity_loss_patch_enc,sparsity_loss_attn_dec, sparsity_loss_mlp_dec,sparsity_loss_patch_dec = actor.net.module.transformer.get_sparsity_loss(self.device)
        
        total_loss = base_loss + self.settings.w1_enc*sparsity_loss_attn_enc + self.settings.w2_enc*sparsity_loss_mlp_enc + self.settings.w1_dec*sparsity_loss_attn_dec + self.settings.w2_dec*sparsity_loss_mlp_dec
        
        return total_loss, base_loss, sparsity_loss_attn_enc, sparsity_loss_mlp_enc,sparsity_loss_attn_dec, sparsity_loss_mlp_dec,stats
    
class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()
        
        self.sparsity_loss_attn_enc = 0
        self.sparsity_loss_mlp_enc = 0
        
        self.sparsity_loss_attn_dec = 0
        self.sparsity_loss_mlp_dec = 0
        
        if settings.local_rank in [-1, 0]:
            if self.settings.script_name=='stark_child' or self.settings.script_name=='stark_child_no_clf' or self.settings.script_name=='stark_child_clf':
                wandb.login(key='be7f0d41e450e88a50bffe21de84b92e10fbb826')
                run = wandb.init(project="Pruning_in_Stark",
                             name=self.settings.script_name + '_b_att' + str(self.settings.b_att) + '_b_mlp_' + str(self.settings.b_mlp),
                             group='stark_child',
                             entity="pruning_in_tracking",
                             resume="allow"
                           )
            elif self.settings.script_name=='stark_sparse_encoder':
                wandb.login(key='be7f0d41e450e88a50bffe21de84b92e10fbb826')
                run = wandb.init(project="Pruning_in_Stark",
                             name=self.settings.script_name,
                             group='stark_sparse_encoder',
                             entity="pruning_in_tracking",
                             resume="allow"
                           )
            else:
                wandb.login(key='be7f0d41e450e88a50bffe21de84b92e10fbb826')
                run = wandb.init(project="Pruning_in_Stark",
                             name=self.settings.script_name,
                             group='stark_sparse',
                             entity="pruning_in_tracking",
                             resume="allow"
                             )
        
        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})
        self.searchloss = searchloss(settings, self.device)
        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)
    
    def updateBN(self):
        for m in self.actor.net.module.backbone[0].modules():
            if isinstance(m, nn.BatchNorm2d):
                try:
                    m.weight.grad.data.add_(self.settings.s*torch.sign(m.weight.data))  # L1
                    print('updated')
                except:
                    continue
                    
    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()
        tk1 = tqdm(enumerate(loader),total = len(loader),leave=True)
        for i, data in tk1:
            # get inputs
#             print(data.keys())
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.settings.trans_prune:
                if not self.use_amp:
                    loss, stats = self.actor(data)
                else:
                    with autocast():
                        loss, stats = self.actor(data)
            if self.settings.trans_prune:
#                 sparsity_loss_attn, sparsity_loss_mlp, sparsity_loss_patch = self.actor.net.module.transformer.get_sparsity_loss(self.device)
#                 loss = loss + self.settings.w1*sparsity_loss_attn + self.settings.w2*sparsity_loss_mlp
                  loss,loss1,self.sparsity_loss_attn_enc,self.sparsity_loss_mlp_enc,self.sparsity_loss_attn_dec,self.sparsity_loss_mlp_dec,stats= self.searchloss(data, self.actor)
            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    torch.autograd.set_detect_anomaly(True)
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    if self.settings.conv_prune:
                        self.updateBN()
                    self.optimizer.step()
                else:
                    torch.autograd.set_detect_anomaly(True)
                    self.scaler.scale(loss).backward()
                    if self.settings.conv_prune:
                        self.updateBN()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

#             print('sparse_loss_atn:',sparsity_loss_attn,'   sparsity_loss_mlp',sparsity_loss_mlp)
            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)
#             print(self.stats[loader.name].keys())
#             tk1.set_postfix({'epoch': self.epoch,'s_att':self.sparsity_loss_attn.item(),
#                              's_mlp':self.sparsity_loss_mlp.item(),
#                              'loss':loss1.item(),'l1_loss':stats["Loss/l1"],
#                              'clf_loss':stats["cls_loss"]})
            if self.settings.trans_prune:
                tk1.set_postfix({'epoch':self.epoch,'s_att_enc':self.settings.w1_enc*self.sparsity_loss_attn_enc.item(),
                                 's_mlp_enc':self.settings.w2_enc*self.sparsity_loss_mlp_enc.item(),
                                 's_att_dec':self.settings.w1_dec*self.sparsity_loss_attn_dec.item(),
                                 's_mlp_dec':self.settings.w2_dec*self.sparsity_loss_mlp_dec.item(),
                                 'naive_loss':loss1.item(),
                                 'total_loss':loss.item()})
            else:
                tk1.set_postfix({'epoch': self.epoch,'naive_loss':loss.item()
                                 })
            # print statistics
#             self._print_stats(i, loader, batch_size)

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)
        
#         train_loss_total = self.stats['train']['Loss/total'].avg
#         train_loss_l1 = self.stats['train']['Loss/l1'].avg
#         train_loss_giou = self.stats['train']['Loss/giou'].avg
#         train_loss_clf = self.stats['train']["cls_loss"].avg
#         train_iou = self.stats['train']['IoU'].avg
        
#         print(self.stats['val'].keys())
#         valid_loss_total = self.stats['val']['Loss/total'].avg
#         valid_loss_l1 = self.stats['val']['Loss/l1'].avg
#         valid_loss_giou = self.stats['val']['Loss/giou'].avg
#         valid_loss_clf = self.stats['val']["cls_loss"].avg
#         valid_iou = self.stats['val']['IoU'].avg
        
#         wandb.log({
#                    'train_loss_total':train_loss_total,'train_loss_l1':train_loss_l1,
#                    'train_loss_giou':train_loss_giou,'train_loss_clf':train_loss_clf,
#                    'train_iou':train_iou,'valid_loss_total':valid_loss_total,
#                    'valid_loss_l1':valid_loss_l1,'valid_loss_giou':valid_loss_giou,
#                    'valid_loss_clf':valid_loss_clf,'valid_iou':valid_iou
#                   })
        

#         self._stats_new_epoch()

        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()
            
            if self.settings.script_name=='stark_sparse':
                train_loss_naive_total = self.stats['train']['Loss/total'].avg
                train_loss_l1 = self.stats['train']['Loss/l1'].avg
                train_loss_giou = self.stats['train']['Loss/giou'].avg
                train_loss_location = self.stats['train']["cls_loss"].avg
                train_iou = self.stats['train']['IoU'].avg

                valid_loss_naive_total = self.stats['val']['Loss/total'].avg
                valid_loss_l1 = self.stats['val']['Loss/l1'].avg
                valid_loss_giou = self.stats['val']['Loss/giou'].avg
                valid_loss_location = self.stats['val']["cls_loss"].avg
                valid_iou = self.stats['val']['IoU'].avg
                

                wandb.log({
                        'w1_enc' : self.settings.w1_enc, 'w2_enc' : self.settings.w2_enc,
                        'w1_dec' : self.settings.w1_dec, 'w2_dec' : self.settings.w2_dec,
                        'attn_loss_enc' : self.sparsity_loss_attn_enc, 'mlp_loss_enc' : self.sparsity_loss_mlp_enc,
                        'attn_loss_dec' : self.sparsity_loss_attn_dec, 'mlp_loss_dec' : self.sparsity_loss_mlp_dec,
                        'train_loss_total':train_loss_naive_total,'train_loss_l1':train_loss_l1,
                        'train_loss_giou':train_loss_giou,'train_loss_clf':train_loss_location,
                        'train_iou':train_iou,
                        'valid_loss_total':valid_loss_naive_total,
                        'valid_loss_l1':valid_loss_l1,'valid_loss_giou':valid_loss_giou,
                        'valid_loss_clf':valid_loss_location,'valid_iou':valid_iou
                      })
            elif self.settings.script_name=='stark_sparse_encoder':
                train_loss_naive_total = self.stats['train']['Loss/total'].avg
                train_loss_l1 = self.stats['train']['Loss/l1'].avg
                train_loss_giou = self.stats['train']['Loss/giou'].avg
                train_loss_location = self.stats['train']["cls_loss"].avg
                train_iou = self.stats['train']['IoU'].avg

                valid_loss_naive_total = self.stats['val']['Loss/total'].avg
                valid_loss_l1 = self.stats['val']['Loss/l1'].avg
                valid_loss_giou = self.stats['val']['Loss/giou'].avg
                valid_loss_location = self.stats['val']["cls_loss"].avg
                valid_iou = self.stats['val']['IoU'].avg

                wandb.log({
                        'w1_enc' : self.settings.w1_enc, 'w2_enc' : self.settings.w2_enc,
                        'attn_loss_enc' : self.sparsity_loss_attn_enc, 'mlp_loss_enc' : self.sparsity_loss_mlp_enc,
                        'train_loss_total':train_loss_naive_total,'train_loss_l1':train_loss_l1,
                        'train_loss_giou':train_loss_giou,'train_loss_clf':train_loss_location,
                        'train_iou':train_iou,
                        'valid_loss_total':valid_loss_naive_total,
                        'valid_loss_l1':valid_loss_l1,'valid_loss_giou':valid_loss_giou,
                        'valid_loss_clf':valid_loss_location,'valid_iou':valid_iou
                      })
            
            elif self.settings.script_name=='stark_sparse_encoder_no_clf':
                train_loss_naive_total = self.stats['train']['Loss/total'].avg
                train_loss_l1 = self.stats['train']['Loss/l1'].avg
                train_loss_giou = self.stats['train']['Loss/giou'].avg
                train_iou = self.stats['train']['IoU'].avg

                valid_loss_naive_total = self.stats['val']['Loss/total'].avg
                valid_loss_l1 = self.stats['val']['Loss/l1'].avg
                valid_loss_giou = self.stats['val']['Loss/giou'].avg
                valid_iou = self.stats['val']['IoU'].avg

                wandb.log({
                        'w1_enc' : self.settings.w1_enc, 'w2_enc' : self.settings.w2_enc,
                        'attn_loss_enc' : self.sparsity_loss_attn_enc, 'mlp_loss_enc' : self.sparsity_loss_mlp_enc,
                        'train_loss_total':train_loss_naive_total,'train_loss_l1':train_loss_l1,
                        'train_loss_giou':train_loss_giou,
                        'train_iou':train_iou,
                        'valid_loss_total':valid_loss_naive_total,
                        'valid_loss_l1':valid_loss_l1,'valid_loss_giou':valid_loss_giou,
                        'valid_iou':valid_iou
                      })
                
            elif self.settings.script_name=='stark_child':
                train_loss_naive_total = self.stats['train']['Loss/total'].avg
                train_loss_l1 = self.stats['train']['Loss/l1'].avg
                train_loss_giou = self.stats['train']['Loss/giou'].avg
                train_loss_location = self.stats['train']["cls_loss"].avg
                train_iou = self.stats['train']['IoU'].avg

                valid_loss_naive_total = self.stats['val']['Loss/total'].avg
                valid_loss_l1 = self.stats['val']['Loss/l1'].avg
                valid_loss_giou = self.stats['val']['Loss/giou'].avg
                valid_loss_location = self.stats['val']["cls_loss"].avg
                valid_iou = self.stats['val']['IoU'].avg

                wandb.log({
                        'train_loss_total':train_loss_naive_total,'train_loss_l1':train_loss_l1,
                        'train_loss_giou':train_loss_giou,'train_loss_clf':train_loss_location,
                        'train_iou':train_iou,
                        'valid_loss_total':valid_loss_naive_total,
                        'valid_loss_l1':valid_loss_l1,'valid_loss_giou':valid_loss_giou,
                        'valid_loss_clf':valid_loss_location,'valid_iou':valid_iou
                      })
            elif self.settings.script_name=='stark_child_clf':
                train_loss_location = self.stats['train']["cls_loss"].avg
                valid_loss_location = self.stats['val']["cls_loss"].avg

                wandb.log({
                        'train_loss_clf':train_loss_location,
                        'valid_loss_clf':valid_loss_location
                      })
            
            elif self.settings.script_name=='stark_child_no_clf':
                train_loss_naive_total = self.stats['train']['Loss/total'].avg
                train_loss_l1 = self.stats['train']['Loss/l1'].avg
                train_loss_giou = self.stats['train']['Loss/giou'].avg
                train_iou = self.stats['train']['IoU'].avg

#                 valid_loss_naive_total = self.stats['val']['Loss/total'].avg
#                 valid_loss_l1 = self.stats['val']['Loss/l1'].avg
#                 valid_loss_giou = self.stats['val']['Loss/giou'].avg
#                 valid_iou = self.stats['val']['IoU'].avg
                
                
                if self.epoch % loader.epoch_interval==0:
                    valid_loss_naive_total = self.stats['val']['Loss/total'].avg
                    valid_loss_l1 = self.stats['val']['Loss/l1'].avg
                    valid_loss_giou = self.stats['val']['Loss/giou'].avg
                    valid_iou = self.stats['val']['IoU'].avg

                else:
                    valid_loss_naive_total = 0
                    valid_loss_l1 = 0
                    valid_loss_giou = 0
                    valid_iou = 0
                
                wandb.log({
                        'train_loss_total':train_loss_naive_total,'train_loss_l1':train_loss_l1,
                        'train_loss_giou':train_loss_giou,
                        'train_iou':train_iou,
                        'valid_loss_total':valid_loss_naive_total,
                        'valid_loss_l1':valid_loss_l1,'valid_loss_giou':valid_loss_giou,
                        'valid_iou':valid_iou
                         })
                
             
            else:
                train_loss_naive_total = self.stats['train']['Loss/total'].avg
                train_loss_l1 = self.stats['train']['Loss/l1'].avg
                train_loss_giou = self.stats['train']['Loss/giou'].avg
                train_iou = self.stats['train']['IoU'].avg

                valid_loss_naive_total = self.stats['val']['Loss/total'].avg
                valid_loss_l1 = self.stats['val']['Loss/l1'].avg
                valid_loss_giou = self.stats['val']['Loss/giou'].avg
                valid_iou = self.stats['val']['IoU'].avg

                wandb.log({
                        'w1_enc' : self.settings.w1_enc, 'w2_enc' : self.settings.w2_enc,
                        'w1_dec' : self.settings.w1_dec, 'w2_dec' : self.settings.w2_dec,
                        'attn_loss_enc' : self.sparsity_loss_attn_enc, 'mlp_loss_enc' : self.sparsity_loss_mlp_enc,
                        'attn_loss_dec' : self.sparsity_loss_attn_dec, 'mlp_loss_dec' : self.sparsity_loss_mlp_dec,
                        'train_loss_total':train_loss_naive_total,'train_loss_l1':train_loss_l1,
                        'train_loss_giou':train_loss_giou,
                        'train_iou':train_iou,
                        'valid_loss_total':valid_loss_naive_total,
                        'valid_loss_l1':valid_loss_l1,'valid_loss_giou':valid_loss_giou,
                        'valid_iou':valid_iou
                      })
            self._stats_new_epoch()
            if self.epoch==self.settings.max_epochs:
                wandb.finish()
            
            
            
        

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
