# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor
import math
from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .detr_layers import DetrTransformerEncoderLayer
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, get_text_sine_pos_embed,coordinate_to_encoding
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from typing import Tuple

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class GroundingDinoTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):

    def __init__(self,
                 cross_attn_text_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        self.cross_attn_text_cfg = cross_attn_text_cfg
        if 'batch_first' not in self.cross_attn_text_cfg:
            self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # cross attention between query and text
        query = self.cross_attn_text(
            query=query,
            query_pos=query_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask)
        query = self.norms[1](query)
        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            key=key,#None
            value=value,#memory
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[2](query)
        query = self.ffn(query)
        query = self.norms[3](query)

        return query


class GroundingDinoTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None,
                concated_feats:bool=False
                ):
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            if memory_text is None:
                bs, n_text, _ = (0, 0, 0)
            else:
                bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if not concated_feats:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                vi_out, ii_out = output.chunk(chunks=2, dim=1)
                vi_out = layer(
                    query=vi_out,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                ii_out = layer(
                    query=ii_out,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                output = torch.concat([vi_out, ii_out], dim=1)            
                
        return output, memory_text

def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return bbox_xyxy_to_cxcywh(bboxes)
from .utils import inverse_sigmoid
class GroundingDinoTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
    
    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                integral:nn.Module=None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_pred_corners = []
        intermediate_reference_points = [reference_points]
        IsCornerReg = (integral is not None)
        
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                pred_corners = reg_branches[lid](query)
                
                if IsCornerReg:
                    new_reference_points = distance2bbox(reference_points, integral(pred_corners), integral.reg_scale)
                else:
                    assert reference_points.shape[-1] == 4
                    new_reference_points = pred_corners + inverse_sigmoid(reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                if IsCornerReg:
                    intermediate_pred_corners.append(pred_corners)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.
        
        if self.return_intermediate:
            # if IsCornerReg:
            #     return torch.stack(intermediate),\
            #         torch.stack(intermediate_reference_points),\
            #             torch.stack(intermediate_pred_corners)
            # else:
                return torch.stack(intermediate),\
                    torch.stack(intermediate_reference_points)

        return query, reference_points


"""
    Text IF实现
"""
import torch.nn.functional as F
class Text_IF_Fusion(nn.Module):
    def __init__(self, in_channels=256, dim=48, text_dim=256, bias=False):
        super().__init__()
        
        # 将输入特征投影到目标维度 (e.g., 256 -> 48)
        self.proj_A = nn.Conv2d(in_channels, dim, kernel_size=1, bias=bias)
        self.proj_B = nn.Conv2d(in_channels, dim, kernel_size=1, bias=bias)

        # LIF-inspired illumination routing.
        # A-branch is treated as visible-light dominant guidance for weight map prediction.
        self.lif_pre = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, kernel_size=1, padding=0, bias=True)
        )
        self.lif_text_affine = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(inplace=True),
            nn.Linear(text_dim, 2)
        )
        self.lif_beta = 0.4

        self.cross_attention = Cross_attention(dim)
        self.attention_spatial = Attention_spatial(dim)
        self.feature_fusion = Fusion_Embed(dim)  # 拼接后是 dim*2
        self.prompt_guidance = FeatureWiseAffine(text_dim, dim)

        # 解码器（简化版）
        self.decoder = nn.Sequential(
            # TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2, bias=bias, LayerNorm_type='WithBias'),
            nn.Conv2d(dim, 256, kernel_size=1, bias=bias)
        )

        self._last_lif_weight_vis = None
        self._last_vis_proxy = None

    def _compute_lif_weights(self, vis_feat, text_feature):
        # Base weight map from visible-branch feature.
        base = torch.sigmoid(self.lif_pre(vis_feat))

        # LIFAdd-style remap: ((w - 0.31) / 0.63) -> clamp(max=0.5) -> beta * x + 0.5
        step1 = (base - 0.31) / 0.63
        step2 = torch.clamp(step1, max=0.5)
        lif_weight = self.lif_beta * step2 + 0.5

        # Text-conditioned affine (small range) to adapt low/high-light routing by prompt context.
        text_scale_bias = self.lif_text_affine(text_feature)
        text_scale = (1.0 + 0.2 * torch.tanh(text_scale_bias[:, 0:1])).view(-1, 1, 1, 1)
        text_bias = (0.1 * torch.tanh(text_scale_bias[:, 1:2])).view(-1, 1, 1, 1)
        lif_weight = torch.clamp(lif_weight * text_scale + text_bias, min=0.05, max=0.95)

        return lif_weight

    def forward(self, feat_A, feat_B, text_feature):
        """输入:
            - feat_A: [batch, in_channels, H, W] (单尺度特征)
            - feat_B: [batch, in_channels, H, W]
            - text_feature: [batch, text_dim]
        """

        feat_A = self.proj_A(feat_A)  # -> [batch, dim, H, W]
        feat_B = self.proj_B(feat_B)

        # LIF routing: high-light -> more VIS (A), low-light -> more IR (B).
        weight_vis = self._compute_lif_weights(feat_A, text_feature)
        weight_ir = 1.0 - weight_vis
        feat_A = feat_A * weight_vis
        feat_B = feat_B * weight_ir

        self._last_lif_weight_vis = weight_vis
        self._last_vis_proxy = torch.sigmoid(feat_A.mean(dim=1, keepdim=True).detach())

        feat_A, feat_B = self.cross_attention(feat_A, feat_B)
        fused = self.feature_fusion(feat_A, feat_B) 
        fused = self.attention_spatial(fused)

        modulated = self.prompt_guidance(fused, text_feature)

        output = self.decoder(modulated)
        output = torch.sigmoid(output)

        return output

    def MaskLoss(self):
        # Keep compatibility with detector loss path and regularize LIF weight map.
        if self._last_lif_weight_vis is None:
            return self.proj_A.weight.new_tensor(0.0)

        w = self._last_lif_weight_vis

        # Smoothness regularization (TV).
        tv_h = (w[:, :, 1:, :] - w[:, :, :-1, :]).abs().mean()
        tv_w = (w[:, :, :, 1:] - w[:, :, :, :-1]).abs().mean()
        loss_tv = tv_h + tv_w

        # Avoid collapse to trivial all-0 or all-1 routing.
        loss_balance = (w.mean() - 0.5).abs()

        # Encourage VIS weight to roughly follow VIS intensity proxy.
        vis_proxy = self._last_vis_proxy
        loss_align = F.l1_loss(w, vis_proxy)

        return 0.05 * loss_tv + 0.10 * loss_balance + 0.15 * loss_align

class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            # print("text parameters:", gamma.shape)
            # print("text parameters:", beta.shape)
            x = (1 + gamma) * x + beta
        return x

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x
