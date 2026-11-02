import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.block import C2f
from einops import rearrange
from ..modules.conv import Conv

__all__ = ['LSRB', 'TASFI', 'Rep_AMFE']


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ChannelAttention_HSFPN(nn.Module):
    def __init__(self, in_planes, ratio=4, flag=True):
        super(ChannelAttention_HSFPN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x if self.flag else self.sigmoid(out)


class LSRU(nn.Module):
    def __init__(self, ch_in, depth=1, kernel=3):
        super(LSRU, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(kernel, kernel), groups=ch_in,
                              padding=(kernel // 2, kernel // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                    ChannelAttention_HSFPN(ch_in)
                )),
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
            ) for i in range(depth)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class LSRB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(LSRU(self.c) for _ in range(n))


#################################################################################################################################

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm1(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm1, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(SFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                            groups=hidden_features, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=3,
                                groups=hidden_features * 2, bias=bias, dilation=3)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm1(dim, 'WithBias'),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm1(dim, 'WithBias'),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2)

        self.fft_channel_weight = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))
        self.fft_channel_bias = nn.Parameter(torch.randn((1, hidden_features, 1, 1)))

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        hw = x.shape[-1]
        return x[..., t_pad[0]:hw - t_pad[1]]

    def forward(self, x, spatial):
        x_dtype = x.dtype
        x = self.project_in(x)

        #### Spatial branch
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)
        ####

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)

        x2, pad_w = self.pad(x2, 2)
        x2 = torch.fft.rfft2(x2.float())
        x2 = self.fft_channel_weight * x2 + self.fft_channel_bias
        x2 = torch.fft.irfft2(x2)
        x2 = self.unpad(x2, pad_w)

        x = F.gelu(x1) * x2
        x = self.project_out(x.to(x_dtype))
        return x

class TASRA(nn.Module):
    """
    Target-Aware Sparse Relational Attention (TASRA)
    Enhanced for UAV Small Object Detection
    """

    def __init__(
            self,
            dim,
            num_heads=8,
            num_rel_queries=32,
            topk_tokens=128,
            attn_drop=0.0,
            proj_drop=0.0,
            temperature=0.7,
            use_entmax=False,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.num_rel_queries = num_rel_queries
        self.topk_tokens = topk_tokens
        self.temperature = temperature
        self.use_entmax = use_entmax

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ---------------------------------------------------------
        # 1.  Objectness Predictor
        # ---------------------------------------------------------
        self.objectness = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, 1, kernel_size=1)
        )

        # ---------------------------------------------------------
        # 2. Learnable Relation Queries
        # ---------------------------------------------------------
        self.rel_queries = nn.Parameter(
            torch.randn(num_rel_queries, dim)
        )

        # ---------------------------------------------------------
        # 3. Linear projections
        # ---------------------------------------------------------
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        #  LayerNorm
        self.norm = nn.LayerNorm(dim)

    # -------------------------------------------------------------

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # ---------------------------------------------------------
        # Step 1: Target-aware token selection
        # ---------------------------------------------------------
        obj_map = torch.sigmoid(self.objectness(x))  # [B,1,H,W]
        obj_score = obj_map.flatten(2)  # [B,1,N]

        feat = x.flatten(2).transpose(1, 2)  # [B,N,C]

        score = obj_score / self.temperature
        prob = F.softmax(score, dim=-1)

        k = min(self.topk_tokens, N)
        _, idx = torch.topk(prob, k=k, dim=-1)
        idx = idx.squeeze(1)  # [B,K]

        kv_tokens = torch.gather(
            feat,
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, C)
        )  # [B,K,C]

        kv_tokens = self.norm(kv_tokens)

        # ---------------------------------------------------------
        # Step 2: Feature-guided Relation Queries
        # ---------------------------------------------------------
        feat_context = feat.mean(dim=1)  # [B,C]
        rel_q = self.rel_queries.unsqueeze(0) + feat_context.unsqueeze(1)

        q = self.q_proj(rel_q)
        k = self.k_proj(kv_tokens)
        v = self.v_proj(kv_tokens)

        # Multi-head reshape
        q = q.view(B, self.num_rel_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # ---------------------------------------------------------
        # Step 3: Sparse Relational Attention
        # ---------------------------------------------------------
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B,h,Q,K]

        if self.use_entmax:
            from entmax import entmax15
            attn = entmax15(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        out = attn @ v  # [B,h,Q,d]
        out = out.transpose(1, 2).reshape(B, self.num_rel_queries, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        # ---------------------------------------------------------
        # Step 4: Spatially Modulated Broadcast
        # ---------------------------------------------------------
        rel_response = out.mean(dim=1)  # [B,C]
        rel_response = rel_response.unsqueeze(-1).unsqueeze(-1)

        spatial_gate = torch.sigmoid(obj_map)  # [B,1,H,W]
        response = rel_response * spatial_gate + x * (1 - spatial_gate)

        return response

class TASFI(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.attn = TASRA(c1, num_heads=num_heads)
        # Implementation of Feedforward model
        self.ffn = SFFN(c1, 2.0, False)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        BS, C, H, W = src.size()
        src_ = src
        src2 = self.attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.ffn(src, src_)
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

###############################################################################################################################

class AMFE(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()

        # -------------------------------
        # Stage 1: Local detail extraction
        # -------------------------------
        self.conv1 = Conv(c1, c1, k=3)

        # -------------------------------
        # Stage 2: Medium-scale context (DWConv)
        # -------------------------------
        self.conv2 = Conv(
            c1 // 2,
            c1 // 2,
            k=5,
            g=c1 // 2
        )

        # -------------------------------
        # Stage 3: Large receptive field
        # (use dilated DWConv instead of 7×7)
        # -------------------------------
        self.conv3 = Conv(
            c1 // 4,
            c1 // 4,
            k=3,
            d=3,
            g=c1 // 4
        )

        # -------------------------------
        # High-frequency enhancement branch
        # -------------------------------
        self.avg_pool = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )

        # -------------------------------
        # Spatial gate (target-aware)
        # -------------------------------
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # -------------------------------
        # Output projection
        # -------------------------------
        self.conv4 = Conv(c1, c2, 1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """

        # -------------------------------
        # Progressive multi-scale extraction
        # -------------------------------
        conv1_out = self.conv1(x)
        conv1_out_1, conv1_out_2 = conv1_out.chunk(2, dim=1)

        conv2_out = self.conv2(conv1_out_1)
        conv2_out_1, conv2_out_2 = conv2_out.chunk(2, dim=1)

        conv3_out = self.conv3(conv2_out_1)

        # -------------------------------
        # Multi-scale feature aggregation
        # -------------------------------
        out = torch.cat(
            [conv3_out, conv2_out_2, conv1_out_2],
            dim=1
        )

        # -------------------------------
        # High-frequency enhancement
        # -------------------------------
        hf = x - self.avg_pool(x)

        out = self.conv4(out)

        # -------------------------------
        # Spatially adaptive modulation
        # -------------------------------
        gate = self.spatial_gate(x)
        out = out * gate + x * (1.0 - gate)

        # -------------------------------
        # Residual connection + HF
        # -------------------------------
        out = out + hf

        return out

class Rep_AMFE(nn.Module):
    """Rep-style C3 with PMSFA1-v2"""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)

        self.m = nn.Sequential(
            *[AMFE(c_, c_) for _ in range(n)]
        )

        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))