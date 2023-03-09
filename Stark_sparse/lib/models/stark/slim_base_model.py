import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.searchable_modules = []

    def calculate_search_threshold(self, budget_attn, budget_mlp, budget_patch):
        zetas_attn, zetas_mlp, zetas_patch = self.give_zetas()
        zetas_attn = sorted(zetas_attn)
        zetas_mlp = sorted(zetas_mlp)
        zetas_patch = sorted(zetas_patch)
        threshold_attn = zetas_attn[int((1.-budget_attn)*len(zetas_attn))]
        threshold_mlp = zetas_mlp[int((1.-budget_mlp)*len(zetas_mlp))]
        threshold_patch = zetas_patch[int((1.-budget_patch)*len(zetas_patch))]
        return threshold_attn, threshold_mlp, threshold_patch
    
    def calculate_search_threshold_sep(self, budget_attn, budget_mlp, budget_patch):
        zetas_attn_enc, zetas_mlp_enc, zetas_patch_enc, zetas_attn_dec, zetas_mlp_dec, zetas_patch_dec = self.give_zetas_sep()
        
        zetas_attn_enc = sorted(zetas_attn_enc)
        zetas_mlp_enc = sorted(zetas_mlp_enc)
        zetas_patch_enc = sorted(zetas_patch_enc)
        
        zetas_attn_dec = sorted(zetas_attn_dec)
        zetas_mlp_dec = sorted(zetas_mlp_dec)
        zetas_patch_dec = sorted(zetas_patch_dec)
        
        threshold_attn_enc = zetas_attn_enc[int((1.-budget_attn)*len(zetas_attn_enc))]
        threshold_mlp_enc = zetas_mlp_enc[int((1.-budget_mlp)*len(zetas_mlp_enc))]
        threshold_patch_enc = zetas_patch_enc[int((1.-budget_patch)*len(zetas_patch_enc))]
        
        threshold_attn_dec = zetas_attn_dec[int((1.-budget_attn)*len(zetas_attn_dec))]
        threshold_mlp_dec = zetas_mlp_dec[int((1.-budget_mlp)*len(zetas_mlp_dec))]
        threshold_patch_dec = zetas_patch_dec[int((1.-budget_patch)*len(zetas_patch_dec))]
        
        return threshold_attn_enc, threshold_mlp_enc, threshold_patch_enc,threshold_attn_dec, threshold_mlp_dec, threshold_patch_dec
    
    def calculate_search_threshold_sep_l(self, budget_attn, budget_mlp, budget_patch):
        zetas_attn_enc, zetas_mlp_enc, zetas_patch_enc,zetas_attn_dec, zetas_mlp_dec, zetas_patch_dec = self.give_zetas_sep_l()
        
        zetas_attn_enc = sorted(zetas_attn_enc)
        zetas_mlp_enc = sorted(zetas_mlp_enc)
        zetas_patch_enc = sorted(zetas_patch_enc)
        
        threshold_attn_enc = zetas_attn_enc[int((1.-budget_attn)*len(zetas_attn_enc))]
        threshold_mlp_enc = zetas_mlp_enc[int((1.-budget_mlp)*len(zetas_mlp_enc))]
        threshold_patch_enc = zetas_patch_enc[int((1.-budget_patch)*len(zetas_patch_enc))]

        threshold_attn_dec = []
        threshold_mlp_dec = []
        threshold_patch_dec = []
        
        for i in range(len(zetas_attn_dec)):
            zeta_attn_dec = sorted(zetas_attn_dec[i])
            threshold_attn_dec.append(zeta_attn_dec[int((1.-budget_attn)*len(zeta_attn_dec))])
            
        for i in range(len(zetas_mlp_dec)):
            zeta_mlp_dec = sorted(zetas_mlp_dec[i])
            zeta_patch_dec = sorted(zetas_patch_dec[i])
            
            threshold_mlp_dec.append(zeta_mlp_dec[int((1.-budget_mlp)*len(zeta_mlp_dec))])
            threshold_patch_dec.append(zeta_patch_dec[int((1.-budget_patch)*len(zeta_patch_dec))])
        
        return threshold_attn_enc, threshold_mlp_enc, threshold_patch_enc,threshold_attn_dec, threshold_mlp_dec, threshold_patch_dec
    
    def calculate_search_threshold_layerwise(self, budget_attn, budget_mlp, budget_patch):
        zetas_attn, zetas_mlp, zetas_patch = self.give_zetas_layerwise()
        
        threshold_attn = []
        threshold_mlp = []
        threshold_patch = []
        
        for i in range(len(zetas_attn)):
            zeta_attn = sorted(zetas_attn[i])
            
            threshold_attn.append(zeta_attn[int((1.-budget_attn)*len(zeta_attn))])
            
        for i in range(len(zetas_mlp)):
            zeta_mlp = sorted(zetas_mlp[i])
            zeta_patch = sorted(zetas_patch[i])
            
            threshold_mlp.append(zeta_mlp[int((1.-budget_mlp)*len(zeta_mlp))])
            threshold_patch.append(zeta_patch[int((1.-budget_patch)*len(zeta_patch))])
                           
        return threshold_attn, threshold_mlp, threshold_patch
    
    def n_remaining(self, m):
        if hasattr(m, 'num_heads'):
            return  (m.searched_zeta if m.is_searched else m.zeta).sum(), (m.searched_patch_zeta if m.is_searched else torch.tanh(m.patch_zeta)).sum()
        return (m.searched_zeta if m.is_searched else m.get_zeta()).sum()
    
    def get_remaining(self):
        """return the fraction of active zeta""" 
        n_rem_attn = 0
        n_total_attn = 0
        n_rem_mlp = 0
        n_total_mlp = 0
        n_rem_patch = 0
        n_total_patch = 0
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                attn, patch = self.n_remaining(l_block)
                n_rem_attn += attn
                n_total_attn += l_block.num_gates*l_block.num_heads
                n_rem_patch += patch
                n_total_patch += self.num_patches
            else:
                n_rem_mlp += self.n_remaining(l_block)
                n_total_mlp += l_block.num_gates
        return n_rem_attn/n_total_attn, n_rem_mlp/n_total_mlp, n_rem_patch/n_total_patch

    def get_sparsity_loss(self, device):
        loss_attn_enc = torch.FloatTensor([]).to(device)
        loss_mlp_enc = torch.FloatTensor([]).to(device)
        loss_patch_enc = torch.FloatTensor([]).to(device)
        
        loss_attn_dec = torch.FloatTensor([]).to(device)
        loss_mlp_dec = torch.FloatTensor([]).to(device)
        loss_patch_dec = torch.FloatTensor([]).to(device)
        
        count_att = 0
        count_mlp = 0
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                count_att+=1
#                 print(l_block,count_att)
                zeta_attn, zeta_patch = l_block.get_zeta()
                if count_att<7:
                    loss_attn_enc = torch.cat([loss_attn_enc, torch.abs(zeta_attn.view(-1))])
                    loss_patch_enc = torch.cat([loss_patch_enc, torch.abs(zeta_patch.view(-1))])
                else:
                    loss_attn_dec = torch.cat([loss_attn_dec, torch.abs(zeta_attn.view(-1))])
                    loss_patch_dec = torch.cat([loss_patch_dec, torch.abs(zeta_patch.view(-1))])
                    
            else:
                count_mlp+=1
#                 print(l_block,count_mlp)
                if count_mlp<7:
                    loss_mlp_enc = torch.cat([loss_mlp_enc, torch.abs(l_block.get_zeta().view(-1))])
                else:
                    loss_mlp_dec = torch.cat([loss_mlp_dec, torch.abs(l_block.get_zeta().view(-1))])
        return torch.sum(loss_attn_enc).to(device),torch.sum(loss_mlp_enc).to(device), torch.sum(loss_patch_enc).to(device), torch.sum(loss_attn_dec).to(device),torch.sum(loss_mlp_dec).to(device),torch.sum(loss_patch_dec).to(device)

    def give_zetas(self):
        zetas_attn = []
        zetas_mlp = []
        zetas_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                zetas_attn.append(zeta_attn.abs().cpu().detach().reshape(-1).numpy().tolist())
                zetas_patch.append(zeta_patch.abs().cpu().detach().reshape(-1).numpy().tolist())
            else:
                zetas_mlp.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
        zetas_attn = [z for k in zetas_attn for z in k ]
        zetas_mlp = [z for k in zetas_mlp for z in k ]
        zetas_patch = [z for k in zetas_patch for z in k ]
        return zetas_attn, zetas_mlp, zetas_patch
    
    def give_zetas_layerwise(self):
        zetas_attn = []
        zetas_mlp = []
        zetas_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                zetas_attn.append(zeta_attn.abs().cpu().detach().reshape(-1).numpy().tolist())
                zetas_patch.append(zeta_patch.abs().cpu().detach().reshape(-1).numpy().tolist())
            else:
                zetas_mlp.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
        return zetas_attn, zetas_mlp, zetas_patch
    
    def give_zetas_layerwise_dec(self):
        zetas_attn = []
        zetas_mlp = []
        zetas_patch = []
        
        count_att = 0
        count_mlp = 0
        
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                if count_att<6:
                    continue
                else:
                    zetas_attn.append(zeta_attn.abs().cpu().detach().reshape(-1).numpy().tolist())
                    zetas_patch.append(zeta_patch.abs().cpu().detach().reshape(-1).numpy().tolist())
                count_att+=1
            else:
                if count_mlp<6:
                    continue
                else:
                    zetas_mlp.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
                count_mlp+=1
                
        return zetas_attn, zetas_mlp, zetas_patch
    
    def give_zetas_sep(self):
        zetas_attn_enc = []
        zetas_mlp_enc = []
        zetas_patch_enc = []
        
        zetas_attn_dec = []
        zetas_mlp_dec = []
        zetas_patch_dec = []
        
        count_att = 0
        count_mlp = 0
        
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                if count_att<4:
                    zetas_attn_enc.append(zeta_attn.abs().cpu().detach().reshape(-1).numpy().tolist())
                    zetas_patch_enc.append(zeta_patch.abs().cpu().detach().reshape(-1).numpy().tolist())
                else:
                    zetas_attn_dec.append(zeta_attn.abs().cpu().detach().reshape(-1).numpy().tolist())
                    zetas_patch_dec.append(zeta_patch.abs().cpu().detach().reshape(-1).numpy().tolist())
                    
                count_att+=1
            else:
                if count_mlp<4:
                    zetas_mlp_enc.append(l_block.get_zeta().abs().cpu().detach().reshape(-1).numpy().tolist())
                else:
                    zetas_mlp_dec.append(l_block.get_zeta().abs().cpu().detach().reshape(-1).numpy().tolist())
                    
                count_mlp+=1 
        
        print('number of layers enc : ',len(zetas_attn_enc))
        print('number of layers dec : ',len(zetas_attn_dec))
        
        zetas_attn_enc = [z for k in zetas_attn_enc for z in k ]
        zetas_mlp_enc = [z for k in zetas_mlp_enc for z in k ]
        zetas_patch_enc = [z for k in zetas_patch_enc for z in k ]
        
        zetas_attn_dec = [z for k in zetas_attn_dec for z in k ]
        zetas_mlp_dec = [z for k in zetas_mlp_dec for z in k ]
        zetas_patch_dec = [z for k in zetas_patch_dec for z in k ]
        
        return zetas_attn_enc, zetas_mlp_enc, zetas_patch_enc , zetas_attn_dec , zetas_mlp_dec , zetas_patch_dec
    
    def give_zetas_sep_l(self):
        zetas_attn_enc = []
        zetas_mlp_enc = []
        zetas_patch_enc = []
        
        zetas_attn_dec = []
        zetas_mlp_dec = []
        zetas_patch_dec = []
        
        count_att = 0
        count_mlp = 0
        
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                zeta_attn, zeta_patch = l_block.get_zeta()
                if count_att<6:
                    zetas_attn_enc.append(zeta_attn.cpu().detach().reshape(-1).numpy().tolist())
                    zetas_patch_enc.append(zeta_patch.cpu().detach().reshape(-1).numpy().tolist())
                else:
                    zetas_attn_dec.append(zeta_attn.cpu().detach().reshape(-1).numpy().tolist())
                    zetas_patch_dec.append(zeta_patch.cpu().detach().reshape(-1).numpy().tolist())
                    
                count_att+=1
            else:
                if count_mlp<6:
                    zetas_mlp_enc.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
                else:
                    zetas_mlp_dec.append(l_block.get_zeta().cpu().detach().reshape(-1).numpy().tolist())
                    
                count_mlp+=1 
        
        zetas_attn_enc = [z for k in zetas_attn_enc for z in k ]
        zetas_mlp_enc = [z for k in zetas_mlp_enc for z in k ]
        zetas_patch_enc = [z for k in zetas_patch_enc for z in k ]
        
        return zetas_attn_enc, zetas_mlp_enc, zetas_patch_enc , zetas_attn_dec , zetas_mlp_dec , zetas_patch_dec
    
    def plot_zt(self):
        """plots the distribution of zeta_t and returns the same"""
        zetas_attn, zetas_mlp, zetas_patch = self.give_zetas()
        zetas = zetas_attn + zetas_mlp + zetas_patch
        exactly_zeros = np.sum(np.array(zetas)==0.0)
        exactly_ones = np.sum(np.array(zetas)==1.0)
        plt.hist(zetas)
        plt.show()
        return exactly_zeros, exactly_ones

    def compress(self, budget_attn, budget_mlp, budget_patch):
        """compress the network to make zeta exactly 1 and 0"""
        if self.searchable_modules == []:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
        thresh_attn, thresh_mlp, thresh_patch = self.calculate_search_threshold(budget_attn, budget_mlp, budget_patch)
                
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                l_block.compress(thresh_attn)
            else:
                l_block.compress(thresh_mlp)
        self.compress_patch(thresh_patch)
        return thresh_attn, thresh_mlp, 0
    
    def compress_layerwise(self, budget_attn, budget_mlp, budget_patch):
        """compress the network to make zeta exactly 1 and 0"""
        if self.searchable_modules == []:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
        thresh_attn, thresh_mlp, thresh_patch = self.calculate_search_threshold_layerwise(budget_attn, budget_mlp, budget_patch)
         
        count_attn = 0
        count_mlp = 0
        
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                l_block.compress(thresh_attn[count_attn])
                count_attn+=1
            else:
                l_block.compress(thresh_mlp[count_mlp])
                count_mlp+=1
#         self.compress_patch(thresh_patch)
        return thresh_attn, thresh_mlp, 0

    #Seperate threshold for encoder and decoder
    def compress_sep(self, budget_attn, budget_mlp, budget_patch):
        """compress the network to make zeta exactly 1 and 0"""
        if self.searchable_modules == []:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
        thresh_attn_enc, thresh_mlp_enc, thresh_patch_enc, thresh_attn_dec, thresh_mlp_dec, thresh_patch_dec = self.calculate_search_threshold_sep(budget_attn, budget_mlp, budget_patch)
        
        count_attn = 0
        count_mlp = 0
        
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                if count_attn<4:
                    l_block.compress(thresh_attn_enc)
                else:
                    l_block.compress(thresh_attn_dec)
                count_attn += 1
            else:
                if count_mlp<4:
                    l_block.compress(thresh_mlp_enc)
                else:
                    l_block.compress(thresh_mlp_dec)
                count_mlp += 1
#         self.compress_patch(thresh_patch)
        return thresh_attn_enc , thresh_mlp_enc , thresh_attn_dec , thresh_mlp_dec
    
    def compress_sep_l(self, budget_attn, budget_mlp, budget_patch):
        """compress the network to make zeta exactly 1 and 0"""
        if self.searchable_modules == []:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'zeta')]
        thresh_attn_enc, thresh_mlp_enc, thresh_patch_enc, thresh_attn_dec, thresh_mlp_dec, thresh_patch_dec = self.calculate_search_threshold_sep_l(budget_attn, budget_mlp, budget_patch)
        
#         print(len(thresh_attn_dec))
        count_attn = 0
        count_mlp = 0
        
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                if count_attn<6:
                    l_block.compress(thresh_attn_enc)
                else:
                    l_block.compress(thresh_attn_dec[count_attn-6])
                count_attn += 1
            else:
                if count_mlp<6:
                    l_block.compress(thresh_mlp_enc)
                else:
                    l_block.compress(thresh_mlp_dec[count_mlp-6])
                count_mlp += 1
#         self.compress_patch(thresh_patch)
        return thresh_attn_enc , thresh_mlp_enc , thresh_attn_dec , thresh_mlp_dec

    def compress_patch(self, threshold):
        zetas = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                _, zeta_patch = l_block.get_zeta()
                zeta_patch = zeta_patch.cpu().detach().numpy()
                zetas.append(zeta_patch)
        mask = np.zeros_like(zeta_patch)
        for i in range(len(zetas)-1, -1, -1):
            temp_mask = zetas[i]>=threshold
            mask = np.logical_or(mask, temp_mask).astype(np.float32)
            zetas[i] = mask
        i = 0
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                l_block.compress_patch(threshold, zetas[i])
                i+=1

    def correct_require_grad(self, w1, w2, w3):
        if w1==0:
            for l_block in self.searchable_modules:
                if hasattr(l_block, 'num_heads'):
                    l_block.zeta.requires_grad = False
        if w2==0:
            for l_block in self.searchable_modules:
                if not hasattr(l_block, 'num_heads'):
                    l_block.zeta.requires_grad = False
        if w3==0:
            for l_block in self.searchable_modules:
                if hasattr(l_block, 'num_heads'):
                    l_block.patch_zeta.requires_grad = False

    def decompress(self):
        for l_block in self.searchable_modules:
            l_block.decompress()
    
    def get_channels(self):
        active_channels_attn = []
        active_channels_mlp = []
        active_channels_patch = []
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'num_heads'):
                active_channels_attn.append(l_block.searched_zeta.numpy())
                active_channels_patch.append(l_block.searched_patch_zeta.numpy())
            else:
                active_channels_mlp.append(l_block.searched_zeta.sum().item())
        return np.squeeze(np.array(active_channels_attn)), np.array(active_channels_mlp), np.squeeze(np.array(active_channels_patch))

    def get_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        searched_params = total_params
        for l_block in self.searchable_modules:
            searched_params-=l_block.get_params_count()[0]
            searched_params+=l_block.get_params_count()[1]
        return total_params, searched_params.item()
