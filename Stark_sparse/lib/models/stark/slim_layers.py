import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=197):
        super().__init__()
        self.num_heads = num_heads
        self.num_patches = num_patches
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=197):
        super().__init__()
        self.num_heads = num_heads
        self.num_patches = num_patches
        head_dim = dim // num_heads
        self.dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q,k,v,attn_mask=None):
        
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        
        
        B,N_q,N_k,N_v,C = q.shape[0],q.shape[1],k.shape[1],v.shape[1],q.shape[2]
        
        
        q = self.q(q).reshape(B, N_q, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        k = self.k(k).reshape(B, N_k, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        v = self.v(v).reshape(B, N_v, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]

        
#         print(q.shape,k.shape,v.shape)
        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)

            attn_mask = attn_mask.view(B, 1, 1, N_k).expand(-1, self.num_heads, -1, -1).reshape(B*self.num_heads, 1, N_k)

            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        
        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q.reshape(B*self.num_heads,N_q,C // self.num_heads), k.reshape(B*self.num_heads,N_k,C // self.num_heads).transpose(-2, -1)) * self.scale
            attn = attn.reshape(B,self.num_heads,N_q,N_k)
            
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
        attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(1,0,2)
        return x   

#  dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=197 
class MultiHeadSparseAttention(MultiHeadAttention):
    def __init__(self, attn_module, head_search=False, uniform_search=False):
        super().__init__(attn_module.q.in_features, attn_module.num_heads, True, attn_module.scale, attn_module.attn_drop.p, attn_module.proj_drop.p)
        self.is_searched = False
        self.num_gates = attn_module.q.in_features // self.num_heads
        if head_search and not uniform_search:
            self.zeta = nn.Parameter(torch.ones(1, 1, self.num_heads, 1, 1))
        elif uniform_search and not head_search:
            self.zeta = nn.Parameter(torch.ones(1, 1, 1, 1, self.num_gates))
        elif head_search and uniform_search:
            self.zeta = nn.Parameter(torch.ones(1, 1, self.num_heads, 1, self.num_gates))
        else:
            self.zeta = nn.Parameter(torch.ones(1, 1, self.num_heads, 1, self.num_gates))
            
        self.searched_zeta = torch.ones_like(self.zeta)
        
        self.patch_zeta = nn.Parameter(torch.ones(1, self.num_patches, 1)*3)
        self.searched_patch_zeta = torch.ones_like(self.patch_zeta)
        self.patch_activation = nn.Tanh()
    
    def forward(self,q,k,v,attn_mask=None):
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        
        z = self.searched_zeta if self.is_searched else self.zeta
        
        B,N_q,N_k,N_v,C = q.shape[0],q.shape[1],k.shape[1],v.shape[1],q.shape[2]
        
        q = self.q(q).reshape(B, N_q, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(k).reshape(B, N_k, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(v).reshape(B, N_v, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q = (q*z)[0]
        k = (k*z)[0]
        v = (v*z)[0]
        
        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)

            attn_mask = attn_mask.view(B, 1, 1, N_k).expand(-1, self.num_heads, -1, -1).reshape(B*self.num_heads, 1, N_k)

            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        
        
        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q.reshape(B*self.num_heads,N_q,C // self.num_heads), k.reshape(B*self.num_heads,N_k,C // self.num_heads).transpose(-2, -1)) * self.scale
            attn = attn.reshape(B,self.num_heads,N_q,N_k)
            
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            
        attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(1,0,2)
        return x
    
    def get_zeta(self):
        return self.zeta.cuda(), self.patch_activation(self.patch_zeta).cuda()
    
    def compress(self, threshold_attn):
        self.is_searched = True
        self.searched_zeta = (self.zeta>=threshold_attn).float().cuda()
        self.zeta.requires_grad = False
        
    def compress_patch(self, threshold_patch=None, zetas=None):
        self.is_searched = True
        zetas = torch.from_numpy(zetas).reshape_as(self.patch_zeta)
        self.searched_patch_zeta = (zetas).float().to(self.zeta.device)
        self.patch_zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True
        self.patch_zeta.requires_grad = True

    def get_params_count(self):
        dim = self.q.in_features
        active = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1:
            active*=self.num_gates
        elif self.zeta.shape[2] == 1:
            active*=self.num_heads
        total_params = dim*dim*3 + dim*3
        total_params += dim*dim + dim
        active_params = dim*active*3 + active*3
        active_params += active*dim +dim
        return total_params, active_params
    
    def get_flops_s(self,num_patches_query,num_patches_key):
        H = self.num_heads
        N_q = num_patches_query
        N_k = num_patches_key
        d = self.num_gates
        sd = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1: # Head Elimination
            sd*=self.num_gates
        elif self.zeta.shape[2] == 1: # Uniform Search
            sd*=self.num_heads
        total_flops = N_q * (H*d * (H*d)) + N_q*H*d #linear q
        total_flops += 2*(N_k * (H*d * (H*d)) + N_k*H*d) #linear k and v
#         total_flops = N * (H*d * (3*H*d)) + 3*N*H*d #linear: qkv
        total_flops += H*N_q*d*N_k + H*N_q*N_k #q@k
        total_flops += 5*H*N_q*N_k #softmax
        total_flops += H*N_q*N_k*d #attn@v
        total_flops += N_q * (H*d * (H*d)) + N_q*H*d #linear: proj
        
        return total_flops
    
    def get_flops(self, num_patches, active_patches):
        H = self.num_heads
        N = num_patches
        n = active_patches
        d = self.num_gates
        sd = self.searched_zeta.sum().data
        if self.zeta.shape[-1] == 1: # Head Elimination
            sd*=self.num_gates
        elif self.zeta.shape[2] == 1: # Uniform Search
            sd*=self.num_heads
        total_flops = N * (H*d * (3*H*d)) + 3*N*H*d #linear: qkv
        total_flops += H*N*d*N + H*N*N #q@k
        total_flops += 5*H*N*N #softmax
        total_flops += H*N*N*d #attn@v
        total_flops += N * (H*d * (H*d)) + N*H*d #linear: proj
        
        active_flops = n * (H*d * (3*sd)) + 3*n*sd #linear: qkv
        active_flops += n*n*sd + H*n*n #q@k
        active_flops += 5*H*n*n #softmax
        active_flops += n*n*sd #attn@v
        active_flops += n * (sd * (H*d)) + n*H*d #linear: proj
        return total_flops, active_flops

    @staticmethod
    def from_attn(attn_module, head_search=False, uniform_search=False):
        attn_module = MultiHeadSparseAttention(attn_module, head_search, uniform_search)
        return attn_module

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='gelu', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act =  _get_activation_fn(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
#         x = self.drop(x)
        return x

class SparseMlp(Mlp):
    def __init__(self, mlp_module):
        super().__init__(mlp_module.fc1.in_features, mlp_module.fc1.out_features, mlp_module.fc2.out_features, act_layer='gelu', drop=mlp_module.drop.p)
        self.is_searched = False
        self.num_gates = mlp_module.fc1.out_features
        self.zeta = nn.Parameter(torch.ones(1, 1, self.num_gates))
        self.searched_zeta = torch.ones_like(self.zeta)  
    
    def forward(self, x, patch_zeta=None):
        if patch_zeta is not None:
            x*=patch_zeta
        z = self.searched_zeta if self.is_searched else self.get_zeta()
        x = self.fc1(x)
        x = self.act(x)
        x *= z # both fc1 and fc2 dimensions eliminated here
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def get_zeta(self):
        return self.zeta.cuda()
    
    def compress(self, threshold):
        self.is_searched = True
        self.searched_zeta = (self.get_zeta()>=threshold).float().cuda()
        self.zeta.requires_grad = False

    def decompress(self):
        self.is_searched = False
        self.zeta.requires_grad = True

    def get_params_count(self):
        dim1 = self.fc1.in_features
        dim2 = self.fc1.out_features
        active_dim2 = self.searched_zeta.sum().data
        total_params = 2*(dim1*dim2) + dim1 + dim2
        active_params = 2*(dim1*active_dim2) + dim1 + active_dim2
        return total_params, active_params
    
    def get_flops(self, num_patches, active_patches):
        total_params, active_params = self.get_params_count()
        return total_params*num_patches, active_params*active_patches

    @staticmethod
    def from_mlp(mlp_module):
        mlp_module = SparseMlp(mlp_module)
        return mlp_module
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    

class ModuleInjection:
    method = 'search'
    searchable_modules = []

    @staticmethod
    def make_searchable_attn(attn_module, head_search=False, uniform_search=False):
        if ModuleInjection.method == 'full':
            return attn_module
        attn_module = MultiHeadSparseAttention.from_attn(attn_module, head_search, uniform_search)
        ModuleInjection.searchable_modules.append(attn_module)
#         print(attn_module)
        return attn_module

    @staticmethod
    def make_searchable_mlp(mlp_module):
        if ModuleInjection.method == 'full':
            return mlp_module
        mlp_module = SparseMlp.from_mlp(mlp_module)
        ModuleInjection.searchable_modules.append(mlp_module)
#         print(mlp_module)
#         print(ModuleInjection.searchable_modules)
        return mlp_module