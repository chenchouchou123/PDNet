"""
- refer to "https://github.com/chinhsuanwu/mobilevit-pytorch/blob/master/mobilevit.py"
"""
import math
import torch
import torch.nn as nn
import pywt
import numpy as np
from einops import rearrange
from scipy.fft import dct
import torch.nn.functional as F

from xformers.components.attention.core import scaled_dot_product_attention




def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, mode='self'):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, C, N, D_3 = x.shape
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b c n (n_qkv h d) -> n_qkv b h n (c d)', n_qkv = 3, h = self.heads)
        qkv = qkv.flatten(1, 2)
        q, k, v = qkv.unbind()

        mask = (torch.rand((k.shape[1], k.shape[1])) <= 1).to(k.device)
        with torch.no_grad():

            attn_matrix = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn_matrix = attn_matrix.masked_fill(~mask, -torch.finfo(q.dtype).max)

            attn_weights = attn_matrix.softmax(dim=-1)
            

            self.last_attn_weights = attn_weights.detach().cpu()
        
        out = scaled_dot_product_attention(q, k, v, att_mask=mask)

        out = rearrange(out, '(b h) n (c d) -> b c n (h d)', b = B, c = C)
        return self.to_out(out)







class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self, x):
        B, C, N_2, D = x.shape
        q = self.to_q(x[:,:,0:N_2//2,:])
        q = rearrange(q, 'b c n (h d) -> (b h c) n d', h = self.heads)
        k = self.to_k(x[:,:,N_2//2:,:])
        k = rearrange(k, 'b c n (h d) -> (b h c) n d', h = self.heads)
        v = self.to_v(x[:,:,N_2//2:,:])
        v = rearrange(v, 'b c n (h d) -> (b h c) n d', h = self.heads)
        

        mask = (torch.rand((k.shape[1], k.shape[1])) <= 1).to(k.device)
        out = scaled_dot_product_attention(q, k, v, att_mask=mask)

        out = rearrange(out, '(b h c) n d -> b c n (h d)', b = B, c = C)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x




class SqueezeExcite(nn.Module):
    def __init__(self, c, se_ratio=0.25):
        super().__init__()
        m = max(1, int(c * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, m, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(m, c, 1, bias=True)
        self.gate = nn.Hardsigmoid(inplace=True)
    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s); s = self.act(s); s = self.fc2(s)
        return x * self.gate(s)



class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)




class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,du=64):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    
    def forward(self, x, u = None):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        
        # new
        if u is not None:
            # x = self.token_film(x, u)
            x = self.ada_zero(x,u)
        
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class MobileViTBlock_Cross(nn.Module):
    """
    V0: Indap: cross_attention + followed Cross_FF
    """
    def __init__(self, dim, depth, cross_attn_depth, channel, kernel_size, patch_size_single, patch_size_cross, mlp_dim, dropout=0.,du=64):
        super().__init__()
        self.ph_single, self.pw_single = patch_size_single
        self.ph, self.pw = patch_size_cross
        self.num_prompts = 4

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, 4, 8, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
            ]))

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
        

        self.prompt_injection = PhysicsPromptInjection(du=du, dim=dim, num_prompts=self.num_prompts)
    
    def forward(self, x1, x2, u=None):
        y1 = x1.clone()
        y2 = x2.clone()                                     

        # Local representations
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        
        # Global representations
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph_single, pw=self.pw_single)
        x2 = rearrange(x2, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph_single, pw=self.pw_single)


        if u is not None:
            x1 = self.prompt_injection(x1, u)
            x2 = self.prompt_injection(x2, u)
        
        x1 = self.transformer(x1)   # (TODO): test bet. weight share / non-share
        x2 = self.transformer(x2)
        
        if u is not None:
            x1 = x1[:, :, self.num_prompts:, :]
            x2 = x2[:, :, self.num_prompts:, :]

        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph_single, w=w//self.pw_single, ph=self.ph_single, pw=self.pw_single)
        x2 = rearrange(x2, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph_single, w=w//self.pw_single, ph=self.ph_single, pw=self.pw_single)
        
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x2 = rearrange(x2, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)

        for cross_attn_1, f_1, f_2 in self.cross_attn_layers:
            cal_qkv = torch.cat((x1,x2), dim=2)
            cal_out = x1 + cross_attn_1(cal_qkv)
            x1_out = f_1(cal_out)  # (TODO): test cal_out = f_1(cal_out)+cal_out or cal_out = f_1(norm(cal_out))+cal_out
            
            cal_qkv = torch.cat((x2,x1), dim=2) 
            cal_out =  x2 + cross_attn_1(cal_qkv)
            x2_out = f_2(cal_out)



        x1 = rearrange(x1_out, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x2 = rearrange(x2_out, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x1= self.conv3(x1)
        x1 = torch.cat((x1, y1), 1)
        x1 = self.conv4(x1)
        x2 = self.conv3(x2)
        x2 = torch.cat((x2, y2), 1)
        x2 = self.conv4(x2)
        return (x1, x2)

class MobileViT(nn.Module):
    def __init__(self, args, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2), fusion_path_embed='time_centric'):
        super().__init__()
        self.fusion_level = args.fusion.fusion_level
        self.fusion_mode = args.fusion.fusion_mode
        self.project = args.project
        ih, iw = image_size
        if fusion_path_embed=='time_centric':
            patch_single = [(ih//8,1),(ih//16,1),(ih//32,1)]
            patch_cross = [(ih//8,1),(ih//16,1),(ih//32,1)]

        L = [3, 3, 3]
        L_C = [3, 3, 3]

        self.conv1 = conv_nxn_bn(1, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        

        
        if self.fusion_mode=='cross':
            self.mvit_cross = nn.ModuleList([])
            self.mvit_cross.append(MobileViTBlock_Cross(dims[0], L[0], L_C[0], channels[5], kernel_size, patch_single[0], patch_cross[0], int(dims[0]*2)))
            self.mvit_cross.append(MobileViTBlock_Cross(dims[1], L[1], L_C[1], channels[7], kernel_size, patch_single[1], patch_cross[1], int(dims[1]*4)))
            self.mvit_cross.append(MobileViTBlock_Cross(dims[2], L[2], L_C[2], channels[9], kernel_size, patch_single[2], patch_cross[2], int(dims[2]*4)))
        else:
            self.mvit = nn.ModuleList([])
            self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, (16, 1), int(dims[0]*2)))
            self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, (8, 1), int(dims[1]*4)))
            self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, (4, 1), int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AdaptiveAvgPool2d((1,None))
        

    def encoding_single(self, x, u=None):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x, u)

        x = self.mv2[5](x)
        x = self.mvit[1](x, u)

        x = self.mv2[6](x)
        x = self.mvit[2](x, u)
        x = self.conv2(x)

        return x

    def encoding_cross(self, x1, x2,u=None):
        # x1 processing before transformer
        x1 = self.conv1(x1)
        x1 = self.mv2[0](x1)

        x1 = self.mv2[1](x1)
        x1 = self.mv2[2](x1)
        x1 = self.mv2[3](x1)      # Repeat
        # x2 processing before transformer
        x2 = self.conv1(x2)
        x2 = self.mv2[0](x2)

        x2 = self.mv2[1](x2)
        x2 = self.mv2[2](x2)
        x2 = self.mv2[3](x2)      # Repeat

        # cross-attention transformer for multi-view fusion
        x1 = self.mv2[4](x1)
        x2 = self.mv2[4](x2)
        (x1, x2) = self.mvit_cross[0](x1, x2, u)


        x1 = self.mv2[5](x1)
        x2 = self.mv2[5](x2)
        (x1, x2) = self.mvit_cross[1](x1, x2, u)


        x1 = self.mv2[6](x1)
        x2 = self.mv2[6](x2)
        (x1, x2) = self.mvit_cross[2](x1, x2, u)

                
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)

        return (x1, x2)

    def forward(self, x, u=None):
        if 'single' in self.fusion_level:
            if '1' in self.fusion_level:
                x = x[:,0,:,:].unsqueeze(dim=1)
            elif '2' in self.fusion_level:
                x = x[:,1,:,:].unsqueeze(dim=1)
            x = self.encoding_single(x,u=u)
            x = self.pool(x).squeeze(dim=(2))
        else:
            x1 = x[:,0,:,:].unsqueeze(dim=1)
            x2 = x[:,1,:,:].unsqueeze(dim=1)
            if 'cross' in self.fusion_mode:
                x1, x2 = self.encoding_cross(x1,x2,u=u)
                x1 = self.pool(x1).squeeze(dim=(2))
                x2 = self.pool(x2).squeeze(dim=(2))
                x = (x1+x2)/2
            elif 'late' in self.fusion_level:
                if 'average' in self.fusion_mode:
                    x1 = self.encoding_single(x1)
                    x2 = self.encoding_single(x2)
                    x1 = self.pool(x1).squeeze(dim=(2))
                    x2 = self.pool(x2).squeeze(dim=(2))
                    x = (x1+x2)/2
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channels // reduction)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size = 1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class SE1D(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        r = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Conv1d(channels, r, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(r, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        B,V,C,T = x.shape
        y = x.view(B*V, C, T)
        w = self.fc(self.pool(y))        # (B*V,C,1)
        y = y * w                        # broadcast on T
        return y.view(B, V, C, T)




def bvt_to_bct(x):
    B,V,C,T = x.shape
    return x.view(B*V, C, T), B, V
def bct_to_bvt(y, B, V):
    C, T = y.shape[1], y.shape[2]
    return y.view(B, V, C, T)










class PhysicsConditionEncoder(nn.Module):

    def __init__(self, du=64, 
                 use_score_mean=True,
                 use_phase_mean=True,
                 use_phase_speed=True,
                 use_score_var=True
                 ):
        super().__init__()
        self.du = du
        

        self.use_score_mean = use_score_mean
        self.use_phase_mean = use_phase_mean
        self.use_phase_speed = use_phase_speed
        self.use_score_var = use_score_var


        in_features = 0
        if self.use_score_mean:  in_features += 1
        if self.use_phase_mean:  in_features += 2
        if self.use_phase_speed: in_features += 1
        if self.use_score_var:   in_features += 1
        

        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Linear(128, du),
            nn.SiLU()
        )

    def forward(self, score, phase):
        B, V, _, T = score.shape
        
        feature_list = []


        if self.use_score_mean:
            feature_list.append(score.mean(dim=-1)) # (B, V, 1)


        if self.use_phase_mean:
            feature_list.append(phase[:, :, 0:1, :].mean(dim=-1)) # sin_mean (B, V, 1)
            feature_list.append(phase[:, :, 1:2, :].mean(dim=-1)) # cos_mean (B, V, 1)


        if self.use_phase_speed:
            dphase = phase[..., 1:] - phase[..., :-1]            
            phase_speed = torch.sqrt((dphase ** 2).sum(dim=2, keepdim=True) + 1e-8)  
            phase_speed = torch.nn.functional.pad(phase_speed, (1,0))              
            feature_list.append(phase_speed.mean(dim=-1)) # (B, V, 1)


        if self.use_score_var:
            feature_list.append(score.var(dim=-1, unbiased=False)) # (B, V, 1)


        feat = torch.cat(feature_list, dim=2)  # (B, V, in_features)
        feat = feat.view(B*V, -1)              # (B*V, in_features)
        
        u_v = self.mlp(feat).view(B, V, self.du) # (B, V, du)
        u = u_v.mean(dim=1)
        
        return u, u_v




    
    



class PhysicsPromptInjection(nn.Module):

    def __init__(self, du, dim, num_prompts=4):
        super().__init__()
        self.num_prompts = num_prompts
        self.dim = dim
        

        self.prompt_proj = nn.Sequential(
            nn.Linear(du, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, num_prompts * dim)
        )

    def forward(self, x, u):

        B, P, N, D = x.shape
        

        prompts = self.prompt_proj(u)                  # [B, K * D]
        prompts = prompts.view(B, self.num_prompts, D) # [B, K, D]
        

        prompts = prompts.unsqueeze(1).expand(-1, P, -1, -1)
        

        x_fused = torch.cat([prompts, x], dim=2)
        
        return x_fused






class CrossModalFusion(nn.Module):

    def __init__(self, c_m: int, c_r: int, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.m2d = nn.Conv1d(c_m, d_model, kernel_size=1)
        self.r2d = nn.Conv1d(c_r, d_model, kernel_size=1)

        self.attn_m_q_r = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_r_q_m = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)


        self.ffn_m = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_r = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )


        self.d2m = nn.Conv1d(d_model, c_m, kernel_size=1)
        self.d2r = nn.Conv1d(d_model, c_r, kernel_size=1)

        self.gate_m = nn.Sequential(nn.Conv1d(c_m, 1, kernel_size=1), nn.Sigmoid())
        self.gate_r = nn.Sequential(nn.Conv1d(c_r, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x_m, x_r):
        # x_m: [B, C_m, T_m], x_r: [B, C_r, T_r]
        B, C_m, T_m = x_m.shape
        B2, C_r, T_r = x_r.shape
        assert B == B2, "Batch size must match"

        # -> [B, T, d]
        m = self.m2d(x_m).transpose(1, 2)
        r = self.r2d(x_r).transpose(1, 2)


        m_ctx, _ = self.attn_m_q_r(query=m, key=r, value=r, need_weights=False)

        r_ctx, _ = self.attn_r_q_m(query=r, key=m, value=m, need_weights=False)


        m_f = m + m_ctx
        r_f = r + r_ctx
        m_f = m_f + self.ffn_m(m_f)
        r_f = r_f + self.ffn_r(r_f)


        m_f = m_f.transpose(1, 2)
        r_f = r_f.transpose(1, 2)
        m_f = self.d2m(m_f)   # [B, C_m, T_m]
        r_f = self.d2r(r_f)   # [B, C_r, T_r]

        gm = self.gate_m(m_f)           # [B, 1, T_m]
        gr = self.gate_r(r_f)           # [B, 1, T_r]
        x_m_out = gm * m_f + (1 - gm) * x_m
        x_r_out = gr * r_f + (1 - gr) * x_r
        

        return x_m_out, x_r_out




class MultiScaleRotaryModulation(nn.Module):

    def __init__(self, C, hidden=64):
        super().__init__()
        assert C % 2 == 0, "Channels must be even for Complex Rotary Modulation"
        

        self.trend_extractor = nn.Sequential(
            nn.AvgPool1d(kernel_size=15, stride=1, padding=7), 
            nn.Conv1d(C, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, C // 2, 1)
        )

        self.detail_extractor = nn.Sequential(
            nn.Conv1d(C, hidden, kernel_size=3, padding=1),
            nn.InstanceNorm1d(hidden), 
            nn.GELU(),
            nn.Conv1d(hidden, C // 2, 1)
        )

    def forward(self, x):
        # x:[B, V, C, T]
        B, V, C, T = x.shape
        y = x.view(B*V, C, T)

 
        theta = self.trend_extractor(y)        # [B*V, C/2, T]
        
        amplitude = 1.0 + torch.tanh(self.detail_extractor(y)) # [B*V, C/2, T]，值域[0, 2]


        cos_theta = torch.cos(theta)    #[B*V, C/2, T]
        sin_theta = torch.sin(theta)    #[B*V, C/2, T]

        y_real = y[:, 0::2, :]
        y_imag = y[:, 1::2, :]
        

        y_real_rotated = amplitude * (y_real * cos_theta - y_imag * sin_theta)
        y_imag_rotated = amplitude * (y_real * sin_theta + y_imag * cos_theta)
        

        y_out = torch.empty_like(y)
        y_out[:, 0::2, :] = y_real_rotated
        y_out[:, 1::2, :] = y_imag_rotated
        

        phase_proxy = torch.cat([cos_theta[:, 0:1, :], sin_theta[:, 0:1, :]], dim=1)
        aux = {
            'phase': phase_proxy.view(B, V, 2, T),             # [B, V, 2, T]
            'amplitude': amplitude.view(B, V, C // 2, T)       # [B, V, C/2, T]
        }
        
        return y_out.view(B, V, C, T), aux

class KinematicBranch(nn.Module):

    def __init__(self, C, k=3, dils=(1, 2), is_torso=False, direction='center'):
        super().__init__()
        self.direction = direction # 'center', 'forward', 'backward'
        layers =[]
        
        if is_torso:

            layers +=[nn.Conv1d(C, C, kernel_size=7, padding=3, groups=C), nn.GELU()]
        else:
            for d in dils:
                # 计算总的 padding 数量
                total_padding = (k - 1) * d
                
                if direction == 'forward':

                    self.pad = nn.ConstantPad1d((total_padding, 0), 0)
                    layers +=[self.pad, nn.Conv1d(C, C, k, padding=0, dilation=d, groups=C), nn.GELU()]
                
                elif direction == 'backward':

                    self.pad = nn.ConstantPad1d((0, total_padding), 0)
                    layers +=[self.pad, nn.Conv1d(C, C, k, padding=0, dilation=d, groups=C), nn.GELU()]
                
                else:

                    layers +=[nn.Conv1d(C, C, k, padding=total_padding//2, dilation=d, groups=C), nn.GELU()]
        
        layers +=[nn.Conv1d(C, C, 1), nn.BatchNorm1d(C)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PartAwareKinematicBlock(nn.Module):

    def __init__(self, C, G=4, reduction=16):
        super().__init__()
        self.G = G
        assert C % G == 0, "Channels must be divisible by Groups (G)"
        

        self.multi_head_router = nn.Sequential(
            nn.AvgPool1d(kernel_size=5, stride=1, padding=2),
            nn.Conv1d(C, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, G * 3, kernel_size=1) 
        )
        self.temperature = nn.Parameter(torch.tensor(0.5))


        self.branch_static = KinematicBranch(C, is_torso=True, direction='center')       
        self.branch_pos    = KinematicBranch(C, k=3, dils=(1, 2, 4), direction='forward') 
        self.branch_neg    = KinematicBranch(C, k=3, dils=(1, 2, 4), direction='backward') 

        self.fuse_pw = nn.Conv1d(3 * C, C, 1)
        self.se = SE1D(C, reduction=reduction)

    def forward(self, x):  
        residual = x
        y, B, V = bvt_to_bct(x)   # [B*V, C, T]
        C, T = y.shape[1], y.shape[2]
        

        logits = self.multi_head_router(y) # [B*V, G*3, T]

        logits = logits.view(B*V, self.G, 3, T)
        

        temp = torch.clamp(self.temperature, min=0.01)
        weights = F.softmax(logits / temp, dim=2) # [B*V, G, 3, T]
        

        w_static = weights[:, :, 0, :]
        w_pos    = weights[:, :, 1, :]
        w_neg    = weights[:, :, 2, :]
        

        channels_per_group = C // self.G
        
        def expand_weights(w):

            return w.unsqueeze(2).expand(-1, -1, channels_per_group, -1).reshape(B*V, C, T)

        w_static_ex = expand_weights(w_static)
        w_pos_ex    = expand_weights(w_pos)
        w_neg_ex    = expand_weights(w_neg)
        

        global_score = (w_pos - w_neg).mean(dim=1).view(B, V, 1, T)
        

        y_static = self.branch_static(y * w_static_ex)
        y_pos    = self.branch_pos(y * w_pos_ex)
        y_neg    = self.branch_neg(y * w_neg_ex)
        

        y_fused = torch.cat([y_static, y_pos, y_neg], dim=1) 
        y_out = self.fuse_pw(y_fused)
        y_out = bct_to_bvt(y_out, B, V)
        y_out = self.se(y_out)
        

        aux_dict = {
            'score': global_score,
            'multi_head_weights': (w_static, w_pos, w_neg) # [B*V, 4, T]
        }
        
        return y_out + residual, aux_dict





class main_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_type = args.train.model
        self.decoder_input = args.model.decoder_input
        if 'mobileVit' in model_type:
            fusion_patch_embedding = 'time_centric'
            if 'xxs' in model_type:
                dims = [64, 80, 96]
                channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
                expansion = 2
            elif 'xs' in model_type:
                dims = [96, 120, 144]
                channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
                expansion = 4
            elif 's' in model_type:
                dims = [144, 192, 240]
                channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
                expansion = 4
            self.radar_mD_encoder = MobileViT(args, image_size=(args.transforms.Dop_size, args.transforms.win_size), 
                                                    dims=dims, 
                                                    channels=channels, 
                                                    expansion=expansion, 
                                                    fusion_path_embed=fusion_patch_embedding)
            self.radar_Rng_encoder = MobileViT(args, image_size=(args.transforms.R_size_rng, args.transforms.win_size_rng), 
                                                    dims=dims, 
                                                    channels=channels, 
                                                    expansion=expansion, 
                                                    fusion_path_embed=fusion_patch_embedding)
        self.regress_mD = nn.Sequential(
                        nn.Conv1d(channels[-1], 3*17, kernel_size=1),
                        nn.BatchNorm1d(3*17),
                        nn.Tanh()
                        )
        self.regress_Rng = nn.Sequential(
                        nn.Conv1d(channels[-1], 3*17, kernel_size=1),
                        nn.BatchNorm1d(3*17),
                        nn.Tanh()
                        )
        if self.decoder_input=='all':
            decoder_dim = 20
        elif self.decoder_input=='vel':
            decoder_dim = 16
        elif self.decoder_input=='rng':
            decoder_dim = 4
        self.fc = nn.Sequential(
                        nn.Linear(decoder_dim*3*17, 16*3*17),
                        nn.Tanh(),
                        nn.Dropout(p=0.5),
                        nn.Linear(16*3*17, 16*3*17)
                        )

        self._initialize_weights()
        self.tb_se_mD = None
        self.tb_se_R = None
        self.cb_se_mD = None
        self.cb_se_R = None

        self.phase_aware_mD = MultiScaleRotaryModulation(C = 128)
        
        self.sign_aware_mD = PartAwareKinematicBlock(C = 128, G = 4)

        c_m_out = 640
        c_r_out = 640
    
        
        self.fusion = CrossModalFusion(c_m=c_m_out, c_r=c_r_out, d_model=128, nhead=4, dropout=0.1)

        self.cond_enc_mD = PhysicsConditionEncoder(du=64, use_score_mean=True, use_phase_mean=False, use_phase_speed=False, use_score_var=False)

        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def apply_dwt_time(self, x, wavelet='db1'):

        x_np = x.cpu().numpy()
        coeffs = pywt.wavedec(x_np, wavelet, axis=-1, level=1)
        cA, cD = coeffs

        x_dwt = np.concatenate([cA, cD], axis=-1) if cD.size > 0 else cA
        return torch.from_numpy(x_dwt).to(x.device)
    
    def apply_dwt_channel(self, x, wavelet='db1'):

        x_np = x.cpu().numpy()
        cA, cD = pywt.wavedec(x_np, wavelet, axis=2, level=1)
        x_dwt = np.concatenate([cA, cD], axis=2)
        out = torch.from_numpy(x_dwt).to(x.device).type_as(x)
        return out
    
    



    
    
    
    def forward(self, x_mD, x_R):

        

        
        #phase_aware
        x_mD, aux_phase_mD = self.phase_aware_mD(x_mD)

        # sign_aware
        x_mD, aux_mD = self.sign_aware_mD(x_mD)



        u_mD, _ = self.cond_enc_mD(aux_mD['score'], aux_phase_mD['phase'])  # (B,64)

        u_R = None
        
        x_mD = self.radar_mD_encoder(x_mD,u = u_mD)
        x_R = self.radar_Rng_encoder(x_R,u = u_R)
        # print(f"x_mD shape after radar_mD_encoder: {x_mD.shape}")
        # print(f"x_R shape after radar_mD_encoder: {x_R.shape}")
        
        x_mD, x_R = self.fusion(x_mD, x_R)
        # print(f"x_mD shape after fusion: {x_mD.shape}")
        # print(f"x_R shape after fusion: {x_R.shape}")

        # Decoder
        _,_,T_mD = x_mD.size()
        _,_,T_R = x_R.size()

        x_mD = self.regress_mD(x_mD).view(-1,T_mD*17*3)     # 17x3xT_mD
        x_R = self.regress_Rng(x_R).view(-1,T_R*17*3)       # 17x3xT_R
        # print(f"x_mD shape after regress_mD and view: {x_mD.shape}"),
        # print(f"x_R shape after regress_mD and view: {x_R.shape}")


        if self.decoder_input=='all':
            x = self.fc(torch.cat((x_mD,x_R),dim=-1))
        elif self.decoder_input=='vel':
            x = self.fc(x_mD)
        elif self.decoder_input=='rng':
            x = self.fc(x_R)
        x = rearrange(x, 'b (t j c) -> b t j c', j=17, t=16, c=3).contiguous()
        
        return x
        



def mobilevit_xxs(args):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels, expansion=2)


def mobilevit_xs(args):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels)


def mobilevit_s(args):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(5, 1, 256, 256)
    
    vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_s()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))