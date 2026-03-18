# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from collections import OrderedDict
import numbers
from einops import rearrange

from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block
from collections import OrderedDict
from diffusers.utils import is_torch_version

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class Encoder(nn.Module):
    """Encoder layer of a variational autoencoder that encodes input into a latent representation."""

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 4,
            down_block_types: Tuple[str, ...] = (
            "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
            layers_per_block: int = 2,
            norm_num_groups: int = 32,
            act_fn: str = "silu",
            double_z: bool = True,
            mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.use_rgb = False
        self.down_block_type = down_block_types
        self.block_out_channels = block_out_channels

        self.tile_sample_min_size = 1024
        self.tile_latent_min_size = int(self.tile_sample_min_size / 8)
        self.tile_overlap_factor = 0.25
        self.use_tiling = False

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        self.gradient_checkpointing = False

    def to_rgb_init(self):
        """Initialize layers to convert features to RGB."""
        self.to_rgbs = nn.ModuleList([])
        self.use_rgb = True
        for i, down_block_type in enumerate(self.down_block_type):
            output_channel = self.block_out_channels[i]
            self.to_rgbs.append(nn.Conv2d(output_channel, 3, kernel_size=3, padding=1))

    def enable_tiling(self):
        """Enable tiling for large inputs."""
        self.use_tiling = True

    def encode(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """Encode the input tensor into a latent representation."""
        sample = self.conv_in(sample)
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)
            return sample
        else:
            for down_block in self.down_blocks:
                sample = down_block(sample)
            sample = self.mid_block(sample)
            return sample

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend two tensors vertically with a smooth transition."""
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend two tensors horizontally with a smooth transition."""
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Encode the input tensor using tiling for large inputs."""
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[:, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                tile = self.encode(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        moments = torch.cat(result_rows, dim=2)
        return moments

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        """Forward pass of the encoder, using tiling if enabled for large inputs."""
        if self.use_tiling and (
                sample.shape[-1] > self.tile_latent_min_size or sample.shape[-2] > self.tile_latent_min_size):
            return self.tiled_encode(sample)
        return self.encode(sample)


class ControlNetConditioningEmbedding(nn.Module):
    """A small network to preprocess conditioning inputs, inspired by ControlNet."""

    def __init__(
            self,
            conditioning_embedding_channels: int,
            conditioning_channels: int = 4
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, conditioning_channels, kernel_size=3, padding=1)
        self.norm_in = nn.GroupNorm(num_channels=conditioning_channels, num_groups=32, eps=1e-6)
        self.conv_out = zero_module(
            nn.Conv2d(conditioning_channels, conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        """Process the conditioning input through the network."""
        conditioning = self.norm_in(conditioning)
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


class QuickGELU(nn.Module):
    """A fast approximation of the GELU activation function."""

    def forward(self, x: torch.Tensor):
        """Apply the QuickGELU activation to the input tensor."""
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        """Apply LayerNorm and preserve the input dtype."""
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    """A transformer-style block with self-attention and an MLP."""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 2)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 2, d_model))])
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        """Apply self-attention to the input tensor."""
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """Forward pass through the residual attention block."""
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """The output of UnifiedUNet2DConditionModel."""
    sample: torch.FloatTensor = None

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

class LayerNorm_res(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_res, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm_res(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm_res(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class FeedForward_DRS(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_DRS, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

class Attention_DRS(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_DRS, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        attn3 = attn3.softmax(dim=-1)

        out3 = (attn3 @ v)

        out = out3 * self.attn3

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock_DRS(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_DRS, self).__init__()

        self.norm1 = LayerNorm_res(dim, LayerNorm_type)
        self.attn = Attention_DRS(dim, num_heads, bias)
        self.norm2 = LayerNorm_res(dim, LayerNorm_type)
        self.ffn = FeedForward_DRS(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class CustomMoE2D(nn.Module):
    def __init__(self, in_channels, out_channels, experts: list):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = nn.Conv2d(in_channels, len(experts), kernel_size=1)

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=1)
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        gate_scores = gate_scores.unsqueeze(2)

        output = (expert_outputs * gate_scores).sum(dim=1)
        return output

class ExpertA(nn.Module):
    def __init__(self, in_channels, out_channels, num_trans_head, num_trans_layer):
        super().__init__()
        self.ExpertA_transformer_layes = nn.Sequential(*[ResidualAttentionBlock(in_channels, num_trans_head) for _ in range(num_trans_layer)])

    def forward(self, concat_feat):
        batch_size, channel, height, width = concat_feat.shape
        return self.ExpertA_transformer_layes(concat_feat.view(batch_size, channel, height * width).transpose(1, 2)).transpose(1, 2).view(batch_size, channel, height, width)

class ExpertB(nn.Module):
    def __init__(self, in_channels, out_channels, num_trans_head, num_trans_layer):
        super().__init__()
        self.ExpertB_transformer_layes = nn.Sequential(*[TransformerBlock(dim=in_channels, num_heads=num_trans_head, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(num_trans_layer)])

    def forward(self, x):
        return self.ExpertB_transformer_layes(x)

class ExpertC(nn.Module):
    def __init__(self, in_channels, out_channels, num_trans_head, num_trans_layer):
        super().__init__()
        self.ExpertC_transformer_layes = nn.Sequential(*[TransformerBlock_DRS(dim=in_channels, num_heads=num_trans_head, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias') for i in range(num_trans_layer)])

    def forward(self, x):
        return self.ExpertC_transformer_layes(x)


class UNet2DConditionModel(OriginalUNet2DConditionModel, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    """A unified 2D UNet model extending OriginalUNet2DConditionModel with custom functionality."""
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: int = 4,
            center_input_sample: bool = False,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
            ),
            mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
            up_block_types: Tuple[str] = (
            "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: Union[int, Tuple[int]] = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            dropout: float = 0.0,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: Union[int, Tuple[int]] = 1280,
            transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
            reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
            encoder_hid_dim: Optional[int] = None,
            encoder_hid_dim_type: Optional[str] = None,
            attention_head_dim: Union[int, Tuple[int]] = 8,
            num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
            dual_cross_attention: bool = False,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            addition_embed_type: Optional[str] = None,
            addition_time_embed_dim: Optional[int] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",
            resnet_skip_time_act: bool = False,
            resnet_out_scale_factor: float = 1.0,
            time_embedding_type: str = "positional",
            time_embedding_dim: Optional[int] = None,
            time_embedding_act_fn: Optional[str] = None,
            timestep_post_act: Optional[str] = None,
            time_cond_proj_dim: Optional[int] = None,
            conv_in_kernel: int = 3,
            conv_out_kernel: int = 3,
            projection_class_embeddings_input_dim: Optional[int] = None,
            attention_type: str = "default",
            class_embeddings_concat: bool = False,
            mid_block_only_cross_attention: Optional[bool] = None,
            cross_attention_norm: Optional[str] = None,
            addition_embed_type_num_heads: int = 64,
    ):
        """Initialize the UnifiedUNet2DConditionModel."""
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            dropout=dropout,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_skip_time_act=resnet_skip_time_act,
            resnet_out_scale_factor=resnet_out_scale_factor,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            time_embedding_act_fn=time_embedding_act_fn,
            timestep_post_act=timestep_post_act,
            time_cond_proj_dim=time_cond_proj_dim,
            conv_in_kernel=conv_in_kernel,
            conv_out_kernel=conv_out_kernel,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            attention_type=attention_type,
            class_embeddings_concat=class_embeddings_concat,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            cross_attention_norm=cross_attention_norm,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )

        # Additional attributes
        self.denoise_encoder = None
        self.information_transformer_layes = None
        self.condition_embedding = None
        self.agg_net = None
        self.spatial_ch_projs = None
        self.MoE = None

    def init_vae_encoder(self, dtype):
        self.denoise_encoder = Encoder()
        if dtype is not None:
            self.denoise_encoder.dtype = dtype

    def init_information_transformer_layes(self):
        num_trans_channel = 640
        num_trans_head = 8
        num_trans_layer = 2
        num_proj_channel = 320
        experts = [ExpertA(num_trans_channel, num_trans_channel, num_trans_head, num_trans_layer), ExpertB(num_trans_channel, num_trans_channel, num_trans_head, num_trans_layer),
                   ExpertC(num_trans_channel, num_trans_channel, num_trans_head, num_trans_layer)]
        self.information_transformer_layes = nn.Sequential(
            *[ResidualAttentionBlock(num_trans_channel, num_trans_head) for _ in range(num_trans_layer)])
        self.spatial_ch_projs = zero_module(nn.Linear(num_trans_channel, num_proj_channel))

        self.MoE = CustomMoE2D(num_trans_channel, num_trans_channel, experts)

    def init_ControlNetConditioningEmbedding(self, channel=512):
        self.condition_embedding = ControlNetConditioningEmbedding(320, channel)

    def init_extra_weights(self):
        self.agg_net = nn.ModuleList()

    def load_additional_layers(self, dtype: Optional[torch.dtype] = torch.float16, channel: int = 512,
                               weight_path: Optional[str] = None):
        """Load additional layers and weights from a file.

        Args:
            weight_path (str): Path to the weight file.
            dtype (torch.dtype, optional): Data type for the loaded weights. Defaults to torch.float16.
            channel (int): Conditioning embedding channel out size. Defaults 512.
        """
        if self.denoise_encoder is None:
            self.init_vae_encoder(dtype)

        if self.information_transformer_layes is None:
            self.init_information_transformer_layes()

        if self.condition_embedding is None:
            self.init_ControlNetConditioningEmbedding(channel)

        if self.agg_net is None:
            self.init_extra_weights()

        # Load weights if provided
        if weight_path is not None:
            state_dict = torch.load(weight_path, weights_only=False)
            self.load_state_dict(state_dict, strict=True)

        # Move all modules to the same device and dtype as the model
        device = next(self.parameters()).device
        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype or next(self.parameters()).dtype)

    def to(self, *args, **kwargs):
        """Override to() to move all additional modules to the same device and dtype."""
        super().to(*args, **kwargs)
        for module in [self.denoise_encoder, self.information_transformer_layes,
                       self.condition_embedding, self.agg_net, self.spatial_ch_projs]:
            if module is not None:
                module.to(*args, **kwargs)
        return self

    def load_state_dict(self, state_dict, strict=True):
        """Load state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
            strict (bool, optional): Whether to strictly enforce that all keys match. Defaults to True.
        """
        core_dict = {}
        additional_dicts = {
            'denoise_encoder': {},
            'information_transformer_layes': {},
            'MoE': {},
            'condition_embedding': {},
            'agg_net': {},
            'spatial_ch_projs': {}
        }

        for key, value in state_dict.items():
            if key.startswith('MoE.'):
                additional_dicts['MoE'][key[len('MoE.'):]] = value
            if key.startswith('denoise_encoder.'):
                additional_dicts['denoise_encoder'][key[len('denoise_encoder.'):]] = value
            elif key.startswith('information_transformer_layes.'):
                additional_dicts['information_transformer_layes'][key[len('information_transformer_layes.'):]] = value
            elif key.startswith('condition_embedding.'):
                additional_dicts['condition_embedding'][key[len('condition_embedding.'):]] = value
            elif key.startswith('agg_net.'):
                additional_dicts['agg_net'][key[len('agg_net.'):]] = value
            elif key.startswith('spatial_ch_projs.'):
                additional_dicts['spatial_ch_projs'][key[len('spatial_ch_projs.'):]] = value
            else:
                core_dict[key] = value

        super().load_state_dict(core_dict, strict=False)
        for module_name, module_dict in additional_dicts.items():
            module = getattr(self, module_name, None)
            if module is not None and module_dict:
                module.load_state_dict(module_dict, strict=strict)

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            input_embedding: Optional[torch.Tensor] = None,
            add_sample: bool = True,
            return_dict: bool = True,
            use_condition_embedding: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """Forward pass prioritizing the original modified implementation.

        Args:
            sample (torch.FloatTensor): The noisy input tensor with shape `(batch, channel, height, width)`.
            timestep (Union[torch.Tensor, float, int]): The number of timesteps to denoise an input.
            encoder_hidden_states (torch.Tensor): The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (torch.Tensor, optional): Optional class labels for conditioning.
            timestep_cond (torch.Tensor, optional): Conditional embeddings for timestep.
            attention_mask (torch.Tensor, optional): An attention mask of shape `(batch, key_tokens)`.
            cross_attention_kwargs (Dict[str, Any], optional): A kwargs dictionary for the AttentionProcessor.
            added_cond_kwargs (Dict[str, torch.Tensor], optional): Additional embeddings to add to the UNet blocks.
            down_block_additional_residuals (Tuple[torch.Tensor], optional): Residuals for down UNet blocks.
            mid_block_additional_residual (torch.Tensor, optional): Residual for the middle UNet block.
            down_intrablock_additional_residuals (Tuple[torch.Tensor], optional): Additional residuals within down blocks.
            encoder_attention_mask (torch.Tensor, optional): A cross-attention mask of shape `(batch, sequence_length)`.
            input_embedding (torch.Tensor, optional): Additional input embedding for preprocessing.
            add_sample (bool): Whether to add the sample to the processed embedding. Defaults to True.
            return_dict (bool): Whether to return a UNet2DConditionOutput. Defaults to True.
            use_condition_embedding (bool): Whether to use the condition embedding. Defaults to True.

        Returns:
            Union[UNet2DConditionOutput, Tuple]: The processed sample tensor, either as a UNet2DConditionOutput or tuple.
        """
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time

        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process (following the original modified logic)
        sample = self.conv_in(sample)  # [B, 4, H, W] -> [B, 320, H, W]
        if input_embedding is not None and self.condition_embedding is not None and self.information_transformer_layes is not None:
            if use_condition_embedding:
                input_embedding = self.condition_embedding(input_embedding)  # [B, 320, H, W]
            batch_size, channel, height, width = input_embedding.shape

            concat_feat = torch.cat([sample, input_embedding], dim=1)

            concat_feat = self.MoE(concat_feat)

            feat_alpha = self.spatial_ch_projs(concat_feat.view(batch_size, 2 * channel,
                                                                           height * width).transpose(1, 2)).transpose(1, 2).view(batch_size, channel, height, width)

            sample = sample + feat_alpha if add_sample else feat_alpha  # Update sample as in the original version

        # 2.5 GLIGEN position net (kept from the original version)
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down (continues the standard flow)
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = down_intrablock_additional_residuals is not None
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)
            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)
            if is_adapter and len(down_intrablock_additional_residuals) > 0 and sample.shape == \
                    down_intrablock_additional_residuals[0].shape:
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)
        return UNet2DConditionOutput(sample=sample)