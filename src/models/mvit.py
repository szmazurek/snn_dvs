import math
from dataclasses import dataclass
from functools import partial
from torchvision.models.video.mvit import (
    MViT_V2_S_Weights,
    MSBlockConfig,
    MultiscaleBlock,
    PositionalEncoding,
    WeightsEnum,
    _unsqueeze,
    _ovewrite_named_param,
    _log_api_usage_once,
)
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.fx
import torch.nn as nn


class MViT(nn.Module):
    def __init__(
        self,
        spatial_size: Tuple[int, int],
        temporal_size: int,
        n_channels: int,
        block_setting: Sequence[MSBlockConfig],
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        proj_after_attn: bool,
        dropout: float = 0.5,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 400,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        patch_embed_kernel: Tuple[int, int, int] = (3, 7, 7),
        patch_embed_stride: Tuple[int, int, int] = (2, 4, 4),
        patch_embed_padding: Tuple[int, int, int] = (1, 3, 3),
    ) -> None:
        """
        MViT main class.

        Args:
            spatial_size (tuple of ints): The spacial size of the input as ``(H, W)``.
            temporal_size (int): The temporal size ``T`` of the input.
            block_setting (sequence of MSBlockConfig): The Network structure.
            residual_pool (bool): If True, use MViTv2 pooling residual connection.
            residual_with_cls_embed (bool): If True, the addition on the residual connection will include
                the class embedding.
            rel_pos_embed (bool): If True, use MViTv2's relative positional embeddings.
            proj_after_attn (bool): If True, apply the projection after the attention.
            dropout (float): Dropout rate. Default: 0.0.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
            num_classes (int): The number of classes.
            block (callable, optional): Module specifying the layer which consists of the attention and mlp.
            norm_layer (callable, optional): Module specifying the normalization layer to use.
            patch_embed_kernel (tuple of ints): The kernel of the convolution that patchifies the input.
            patch_embed_stride (tuple of ints): The stride of the convolution that patchifies the input.
            patch_embed_padding (tuple of ints): The padding of the convolution that patchifies the input.
        """
        super().__init__()
        # This implementation employs a different parameterization scheme than the one used at PyTorch Video:
        # https://github.com/facebookresearch/pytorchvideo/blob/718d0a4/pytorchvideo/models/vision_transformers.py
        # We remove any experimental configuration that didn't make it to the final variants of the models. To represent
        # the configuration of the architecture we use the simplified form suggested at Table 1 of the paper.
        _log_api_usage_once(self)
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch Embedding module
        self.conv_proj = nn.Conv3d(
            in_channels=n_channels,
            out_channels=block_setting[0].input_channels,
            kernel_size=patch_embed_kernel,
            stride=patch_embed_stride,
            padding=patch_embed_padding,
        )

        input_size = [
            size // stride
            for size, stride in zip(
                (temporal_size,) + spatial_size, self.conv_proj.stride
            )
        ]

        # Spatio-Temporal Class Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=block_setting[0].input_channels,
            spatial_size=(input_size[1], input_size[2]),
            temporal_size=input_size[0],
            rel_pos_embed=rel_pos_embed,
        )

        # Encoder module
        self.blocks = nn.ModuleList()
        for stage_block_id, cnf in enumerate(block_setting):
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = (
                stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            )

            self.blocks.append(
                block(
                    input_size=input_size,
                    cnf=cnf,
                    residual_pool=residual_pool,
                    residual_with_cls_embed=residual_with_cls_embed,
                    rel_pos_embed=rel_pos_embed,
                    proj_after_attn=proj_after_attn,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
            )

            if len(cnf.stride_q) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, cnf.stride_q)
                ]
        self.norm = norm_layer(block_setting[-1].output_channels)

        # Classifier module
        self.head = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(block_setting[-1].output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)
        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        required_encoding_shape = thw[0] * thw[1] * thw[2]
        # print(f"Required encoding shape: {required_encoding_shape}")
        # print(f"X shape: {x.shape}")
        if x.shape[1] - 1 != required_encoding_shape:
            class_token, x = torch.tensor_split(x, indices=(1,), dim=1)

            # Permute the tensor to put the spatial dimension at the end
            x_permuted = x.permute(0, 2, 1)

            # Interpolate
            x_new = F.interpolate(x_permuted, size=(required_encoding_shape,))

            # Permute back to original dimension order
            x_new = x_new.permute(0, 2, 1)

            # Add class token back
            x = torch.cat((class_token, x_new), dim=1)
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.head(x)

        return x


def _mvit(
    spatial_size: Tuple[int, int],
    temporal_size: int,
    n_channels: int,
    block_setting: List[MSBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> MViT:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "spatial_size", weights.meta["min_size"])
        _ovewrite_named_param(
            kwargs, "temporal_size", weights.meta["min_temporal_size"]
        )

    model = MViT(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        n_channels=n_channels,
        block_setting=block_setting,
        residual_pool=kwargs.pop("residual_pool", False),
        residual_with_cls_embed=kwargs.pop("residual_with_cls_embed", True),
        rel_pos_embed=kwargs.pop("rel_pos_embed", False),
        proj_after_attn=kwargs.pop("proj_after_attn", False),
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


def mvit_v2_s(
    *,
    weights: Optional[MViT_V2_S_Weights] = None,
    progress: bool = True,
    n_channels: int = 3,
    spatial_size: Tuple[int, int] = (256, 450),
    temporal_size: int = 8,
    **kwargs: Any,
) -> MViT:
    """Constructs a small MViTV2 architecture from
    `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__ and
    `MViTv2: Improved Multiscale Vision Transformers for Classification
    and Detection <https://arxiv.org/abs/2112.01526>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V2_S_Weights
            :members:
    """

    config: Dict[str, List] = {
        "num_heads": [1, 2, 4, 4, 8],
        "input_channels": [96, 192, 384, 384, 384],
        "output_channels": [192, 384, 384, 384, 768],
        "kernel_q": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        "kernel_kv": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        "stride_q": [[1, 2, 2], [1, 2, 2], [1, 1, 1], [1, 1, 1], [1, 2, 2]],
        "stride_kv": [[1, 4, 4], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 1, 1]],
    }

    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )

    return _mvit(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        block_setting=block_setting,
        n_channels=n_channels,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=weights,
        progress=progress,
        **kwargs,
    )
