import math
import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from natten.functional import NATTEN2DQKRPBFunction, NATTEN2DAVFunction

from timm.models.layers import DropPath, to_2tuple



class DAttentionBaseline(nn.Module):
    
    # an example parameter setting:
    #   - fmap_size = (56,56)
    #   - heads = 4
    #   - hc = 128 // 4 = 32 (head_dim)
    #   - n_groups = 2 (groups of offset mentioned in the paper)
    #   - attn_drop = 0.0; proj_drop = 0.0
    #   - stride = 8; offset_range_factor = -1
    #   - use_pe = True; dwc_pe = False
    #  - no_off = False; fixed_pe = False; ksize = 9; log_cpb = False
    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride, 
        offset_range_factor, use_pe, dwc_pe,
        no_off, fixed_pe, ksize, log_cpb
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        # essentially, a channel-wise interaction layer: each out channel is a linear combination of all in channels
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(
                    self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            elif self.log_cpb:
                # Borrowed from Swin-V2
                self.rpe_table = nn.Sequential(
                    nn.Linear(2, 32, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, self.n_group_heads, bias=False)
                )
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def forward(self, x):
        """
        NOTE: the code comment following takes x.shape = (2, 128, 56, 56) as example
        """
        
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # proj_q: essentially, a channel-wise interaction layer: each out channel is a linear combination of all in channels
        q = self.proj_q(x) # q.shape = (2, 128, 56, 56)
        
        # self.n_group_channels = self.nc // self.n_groups = 128 // 2 = 64 (analogy to attn head dim)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels) # q_off.shape = (2, 64, 56, 56)
        
        """
        In [14]: self.conv_offset
        ----------------------------------------------------------------------
        Out[14]: 
        Sequential(
            (0): Conv2d(64, 64, kernel_size=(9, 9), stride=(8, 8), padding=(4, 4), groups=64) # out: (4, 64, 7, 7)
            (1): LayerNormProxy(
                (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
            )
            (2): GELU(approximate='none')
            
            # linear combination of the 64 channels to get 2 channels, that is, x and y offset
            (3): Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
        -----------------------------------------------------------------------
        """
        # In the paper:
        #     Considering that each reference point covers a local r×r region,
        #   the offset generation network should also have at least r×r perception of the local features to learn reasonable offsets.
        #     The input features are first passed through a k×k depth-wise convolution with r stride to capture local features,
        #   where k is slightly larger than r to ensure the reasonable receptive field,
        #   followed by a LayerNorm and a GELU activation.
        #   Then, a 1×1 convolution is adopted to predict the 2-D offsets,
        #   bias is dropped to alleviate the inappropriate compulsive shift for all locations.
        
        # NOTE: therefore, the essence of deformable attention is to offset the kv according to local features map
        # Although the local feature map may already be spatially mixed by the previous layers.
        # One weird thing is that the offset can make the reference point go outside the local feature box
        offset = self.conv_offset(q_off).contiguous()  # (B*g, 2, Hg, Wg). here offset.shape = (4,2,7,7)
        Hk, Wk = offset.size(2), offset.size(3) # 7, 7
        n_sample = Hk * Wk # 7 * 7 = 49

        # in this paper, the author doesn't scale the offset, so offset_range_factor = -1
        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p') # offset.shape = (4, 7, 7, 2)
        
        # initialize reference points as 2D grid, normalized to [-1, 1]
        reference = self._get_ref_points(Hk, Wk, B, dtype, device) # reference.shape = (4, 7, 7, 2)

        if self.no_off: # False
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else: # in the paper "we simply clamp the offset so that the offset points are within the image"
            pos = (offset + reference).clamp(-1., +1.) # pos.shape = (4, 7, 7, 2)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            # get the feature value of the offset points, this will be used to calculate k and v
            # in the paper "we use bilinear interpolation to sample the feature values at the offset points"
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
                grid=pos[..., (1, 0)], # y, x -> x, y
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg; here x_sampled.shape = (4, 64, 7, 7)
                

        x_sampled = x_sampled.reshape(B, C, 1, n_sample) # x_sampled.shape = (2, 128, 1, 49)

        # attention calculation
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W) # q.shape = (8, 32, 3136)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale) # attn.shape = (8, 3136, 49) here is the attn weights, paid by 3136 queries to 49 kv's

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(4.0) # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement) # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                # NOTE: here is the pe used in the paper
                
                # self.rpe_table is nn.Parameter(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
                # here q_h = 56, q_w = 56, so rpe_table.shape = (4, 111, 111)
                # why self.q_h * 2 - 1 rather than self.q_h?
                # because the offset is in the range of [-1, 1], so the max value is 1, and the min value is -1
                # For instance, the q point is (1,1) and the offset point is (0,0), then the displacement is (1,1) - (0,0) = (1,1)
                #   and if the q point is (0,0) and the offset point is (1,1), then the displacement is (0,0) - (1,1) = (-1,-1)
                # so we need to double the size of the q_h to cover all the possible displacement values
                # NOTE: here the author may be wrong, the self.rpe_table should be (self.n_heads, self.q_h * 2 + 1, self.q_w * 2 + 1)
                rpe_table = self.rpe_table # rpe_table.shape = (4, 111, 111)
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1) # rpe_bias.shape = (2, 4, 111, 111)
                
                # q_grid is a normalized 2D grid in [-1, 1]
                q_grid = self._get_q_grid(H, W, B, dtype, device) # q_grid.shape = (4, 56, 56, 2)
                
                # displacement.shape = [4, 3136, 49, 2]
                # displacement = (4,3136,1,2) - (4,1,49,2) -> (4,3136,49,2)
                # displacement[..., i, j, :] means the displacement vector between i-th query point and the j-th offset point
                # a displacement can be from [-1,-1] to [1,1]
                # each range from 0 to 1 or -1 to 0 is divided into 56 parts
                # according to the displacement, the attn_bias looks up the rpe_table and interpolates the value
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups), # (4,2,111,111)
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns; here [4, 2, 3136, 49]

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample) # 
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v) # out.shape = (8, 32, 3136)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W) # out.shape = (2, 128, 56, 56)

        y = self.proj_drop(self.proj_out(out)) # y.shape = (2, 128, 56, 56)

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)







class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """
    def __init__(self, dim, kernel_size, num_heads, attn_drop=0., proj_drop=0.,
                 dilation=None):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size
        if type(dilation) is str:
            self.dilation = None
            self.window_size = None
        else:
            assert dilation is None or dilation >= 1, \
                f"Dilation must be greater than or equal to 1, got {dilation}."
            self.dilation = dilation or 1
            self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        x = x.permute(0, 2, 3, 1)
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        dilation = self.dilation
        window_size = self.window_size
        if window_size is None:
            dilation = max(min(H, W) // self.kernel_size, 1)
            window_size = dilation * self.kernel_size
        if H < window_size or W < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - W)
            pad_b = max(0, window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        
        attn = NATTEN2DQKRPBFunction.apply(q, k, self.rpb, self.kernel_size, dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = NATTEN2DAVFunction.apply(attn, v, self.kernel_size, dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]

        return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2), None, None



class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))
    
    def forward(self, x):

        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

    
class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 1, 1, 0),
            # nn.GELU(),
            # nn.BatchNorm2d(self.dim2, eps=1e-5)
        )
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        # self.bn = nn.BatchNorm2d(self.dim2, eps=1e-5)
        self.linear2 = nn.Sequential(
            nn.Conv2d(self.dim2, self.dim1, 1, 1, 0),
            # nn.BatchNorm2d(self.dim1, eps=1e-5)
        )
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        # x = self.bn(x)
        x = self.linear2(x)
        x = self.drop2(x)
        
        return x

class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio,
                 heads, heads_q, stride,
                 offset_range_factor,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, 
                 use_dwc_mlp, ksize, nat_ksize,
                 k_qna, nq_qna, qna_activation, 
                 layer_scale_value, use_lpu, log_cpb):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        
        # in the current config, dim_in = dim_embed
        # it is for channel-wise interaction: each out channel is a linear combination of all in channels
        self.proj = nn.Conv2d(
            in_channels=dim_in,
            out_channels=dim_embed,
            kernel_size=1,
            stride=1,
            padding=0
        ) if dim_in != dim_embed else nn.Identity()
        
        self.stage_spec = stage_spec
        self.use_lpu = use_lpu

        self.ln_cnvnxt = nn.ModuleDict(
            {str(d): LayerNormProxy(dim_embed) for d in range(depths) if stage_spec[d] == 'X'}
        )
        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) if stage_spec[d // 2] != 'X' else nn.Identity() for d in range(2 * depths)]
        )

        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP

        self.mlps = nn.ModuleList(
            [ 
                mlp_fn(dim_embed, expansion, drop) for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        self.layer_scales = nn.ModuleList(
            [
                LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity() 
                for _ in range(2 * depths)
            ]
        )
        
        # lpu module.
        # In paper:
        #   - "a depth-wise convolution wrapped by a residual connection"
        #   - "Similar to CPE, lpu is usually placed at the top of every transformer block to enhance positional information implicitly"
        # It is called 'depth-wise' because the 'groups' parameter is set to dim_embed.
        # In this way, an output channel only depends on the corresponding input channel (rather than all input channels).
        # Depth-wise convolution is used to get spatial correlation.
        self.local_perception_units = nn.ModuleList(
            [
                nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed) if use_lpu else nn.Identity()
                for _ in range(depths)
            ]
        )

        for i in range(depths):
            if stage_spec[i] == 'L':
                pass # not used
                # self.attns.append(
                #     LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                # )
            elif stage_spec[i] == 'D':
                # an example parameter setting:
                #   - fmap_size = (56,56)
                #   - heads = 4
                #   - hc = 128 // 4 = 32 (head_dim)
                #   - n_groups = 2 (groups of offset mentioned in the paper)
                #   - attn_drop = 0.0; proj_drop = 0.0
                #   - stride = 8; offset_range_factor = -1
                #   - use_pe = True; dwc_pe = False
                #  - no_off = False; fixed_pe = False; ksize = 9; log_cpb = False
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, ksize, log_cpb)
                )
            elif stage_spec[i] == 'S':
                pass # not used
                # shift_size = math.ceil(window_size / 2)
                # self.attns.append(
                #     ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                # )
            elif stage_spec[i] == 'N':
                self.attns.append(
                    NeighborhoodAttention2D(dim_embed, nat_ksize, heads, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'P':
                pass # not used
                # self.attns.append(
                #     PyramidAttention(dim_embed, heads, attn_drop, proj_drop, sr_ratio)
                # )
            elif stage_spec[i] == 'Q':
                pass # not used
                # self.attns.append(
                #     FusedKQnA(nq_qna, dim_embed, heads_q, k_qna, 1, 0, qna_activation)
                # )
            elif self.stage_spec[i] == 'X':
                self.attns.append(
                    nn.Conv2d(dim_embed, dim_embed, kernel_size=window_size, padding=window_size // 2, groups=dim_embed)
                )
            elif self.stage_spec[i] == 'E':
                pass # not used
                # self.attns.append(
                #     SlideAttention(dim_embed, heads, 3)
                # )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')

            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())

    def forward(self, x):

        x = self.proj(x) # do channel-wise interaction, see definition of self.proj

        for d in range(self.depths):
            
            # see explanation in __init__
            if self.use_lpu:
                x0 = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + x0
                # out: same shape as x

            if self.stage_spec[d] == 'X':
                x0 = x
                x = self.attns[d](x)
                x = self.mlps[d](self.ln_cnvnxt[str(d)](x))
                x = self.drop_path[d](x) + x0
            else:
                x0 = x
                x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
                x = self.layer_scales[2 * d](x)
                x = self.drop_path[d](x) + x0
                x0 = x
                x = self.mlps[d](self.layer_norms[2 * d + 1](x))
                x = self.layer_scales[2 * d + 1](x)
                x = self.drop_path[d](x) + x0

        return x


class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], 
                 heads=[3, 6, 12, 24], heads_q=[6, 12, 24, 48],
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1],
                 offset_range_factor=[1, 2, 3, 4],
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 ksizes=[9, 7, 5, 3],
                 ksize_qnas=[3, 3, 3, 3],
                 nqs=[2, 2, 2, 2],
                 qna_activation='exp',
                 nat_ksizes=[3,3,3,3],
                 layer_scale_values=[-1,-1,-1,-1],
                 use_lpus=[False, False, False, False],
                 log_cpb=[False, False, False, False],
                 **kwargs):
        super().__init__()

        # patch_size = 4, dim_stem = 128, use_conv_patches = True
        # note the kernel size is bigger than the stride, so there are overlaps
        # in paper "Overlapped patch embedding layers provide remarkable gains".
        self.patch_proj = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim_stem // 2, kernel_size=3, stride=patch_size // 2, padding=1),
            LayerNormProxy(dim_stem // 2),
            nn.GELU(),
            nn.Conv2d(dim_stem // 2, dim_stem, 3, patch_size // 2, 1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        )

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(
                    img_size, window_sizes[i], ns_per_pts[i],
                    dim1, dim2, depths[i],
                    stage_spec[i], groups[i], use_pes[i],
                    sr_ratios[i], heads[i], heads_q[i], strides[i],
                    offset_range_factor[i],
                    dwc_pes[i], no_offs[i], fixed_pes[i],
                    attn_drop_rate, drop_rate, expansion, drop_rate,
                    dpr[sum(depths[:i]):sum(depths[:i + 1])], use_dwc_mlps[i],
                    ksizes[i], nat_ksizes[i], ksize_qnas[i], nqs[i],qna_activation,
                    layer_scale_values[i], use_lpus[i], log_cpb[i]
                )
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            # can think each down_proj as 'further patchify the latent image output by the previous stage'
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=dims[i], out_channels=dims[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )

        self.cls_norm = LayerNormProxy(dims[-1]) 
        self.cls_head = nn.Linear(dims[-1], num_classes)

        self.lower_lr_kvs = lower_lr_kvs

        self.reset_parameters()

    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def load_pretrained(self, state_dict, lookup_22k):

        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l_side = int(math.sqrt(n))
                    assert n == l_side ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l_side, l_side, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
                if 'cls_head' in keys:
                    new_state_dict[state_key] = state_value[lookup_22k]

        msg = self.load_state_dict(new_state_dict, strict=False)
        return msg

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x):
        x = self.patch_proj(x) # [2, 3, 224, 224] -> [2, 128, 56, 56]
        
        # the x shape of each stage:
        #   - start of iter: [2, 128, 56, 56], [2, 256, 28, 28], [2, 512, 14, 14], [2, 1024, 7, 7]
        #   - after stage forward: [2, 128, 56, 56], [2, 256, 28, 28], [2, 512, 14, 14], [2, 1024, 7, 7]
        #   - after down projection: [2, 256, 28, 28], [2, 512, 14, 14], [2, 1024, 7, 7]
        # output shape: [2, 1024, 7, 7]
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)

        x = self.cls_norm(x) # [2, 1024, 7, 7] -> [2, 1024, 7, 7]
        x = F.adaptive_avg_pool2d(x, 1) # [2, 1024, 7, 7] -> [2, 1024, 1, 1]
        x = torch.flatten(x, 1) # [2, 1024, 1, 1] -> [2, 1024]
        x = self.cls_head(x) # [2, 1024] -> [2, 1000]
        return x, None, None


if __name__ == '__main__':
    
    # NOTE: here uses dat_base.yaml
    
    config = {
        'img_size': 224,
        'patch_size': 4,
        'num_classes': 1000,
        'expansion': 4,
        'dim_stem': 128,
        'dims': [128, 256, 512, 1024],
        'depths': [2, 4, 18, 2],
        'stage_spec': [['N', 'D'], ['N', 'D', 'N', 'D'], ['N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D', 'N', 'D'], ['D', 'D']],
        'heads': [4, 8, 16, 32],
        'window_sizes': [7, 7, 7, 7],
        'groups': [2, 4, 8, 16],
        'use_pes': [True, True, True, True],
        'dwc_pes': [False, False, False, False],
        'strides': [8, 4, 2, 1], # This will be passed to the 'strides' parameter of __init__
        'offset_range_factor': [-1, -1, -1, -1],
        'no_offs': [False, False, False, False],
        'fixed_pes': [False, False, False, False],
        'use_dwc_mlps': [True, True, True, True],
        'use_lpus': [True, True, True, True],
        'use_conv_patches': True,
        'ksizes': [9, 7, 5, 3],
        'nat_ksizes': [7, 7, 7, 7],
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.7
    }

    # Initialize the DAT model with the provided configuration
    # Parameters not in the config will use their default values from the __init__ method
    dat_model = DAT(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        expansion=config['expansion'],
        dim_stem=config['dim_stem'],
        dims=config['dims'],
        depths=config['depths'],
        heads=config['heads'],
        # heads_q will use its default: [6, 12, 24, 48]
        window_sizes=config['window_sizes'],
        drop_rate=config['drop_rate'],
        attn_drop_rate=config['attn_drop_rate'],
        drop_path_rate=config['drop_path_rate'],
        strides=config['strides'], # from config
        offset_range_factor=config['offset_range_factor'],
        stage_spec=config['stage_spec'],
        groups=config['groups'],
        use_pes=config['use_pes'],
        dwc_pes=config['dwc_pes'],
        # sr_ratios will use its default: [8, 4, 2, 1]
        # lower_lr_kvs will use its default: {}
        fixed_pes=config['fixed_pes'],
        no_offs=config['no_offs'],
        # ns_per_pts will use its default: [4, 4, 4, 4]
        use_dwc_mlps=config['use_dwc_mlps'],
        use_conv_patches=config['use_conv_patches'],
        ksizes=config['ksizes'],
        # ksize_qnas will use its default: [3, 3, 3, 3]
        # nqs will use its default: [2, 2, 2, 2]
        # qna_activation will use its default: 'exp'
        nat_ksizes=config['nat_ksizes'],
        # layer_scale_values will use its default: [-1,-1,-1,-1]
        use_lpus=config['use_lpus']
        # log_cpb will use its default: [False, False, False, False]
    )

    dummy_input = torch.randn(2,3,224,224)  # Batch size of 2, 3 channels, 224x224 image size

    out, _, _ = dat_model(dummy_input) # the output is a logit of 1000 classes
