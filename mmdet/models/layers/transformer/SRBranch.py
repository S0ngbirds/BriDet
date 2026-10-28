# models/sr_branch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from mmdet.registry import MODELS


class CrossScaleAttention(nn.Module):
    def __init__(self,
                 low_channels: int,
                 high_channels: int,
                 reduction: int = 4):
        super().__init__()
        hidden_channels = max(min(low_channels, high_channels) // reduction, 16)
        self.low_proj = nn.Conv2d(low_channels, hidden_channels, kernel_size=1)
        self.high_proj = nn.Conv2d(high_channels, hidden_channels, kernel_size=1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, low_channels + high_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self,
                low_level_feat: torch.Tensor,
                high_level_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if high_level_feat.shape[2:] != low_level_feat.shape[2:]:
            high_level_feat = F.interpolate(
                high_level_feat,
                size=low_level_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        shared_context = torch.cat([
            self.low_proj(low_level_feat),
            self.high_proj(high_level_feat)
        ], dim=1)
        channel_gate = self.channel_gate(shared_context)
        low_gate, high_gate = torch.split(
            channel_gate,
            [low_level_feat.shape[1], high_level_feat.shape[1]],
            dim=1
        )

        spatial_context = torch.cat([
            low_level_feat.mean(dim=1, keepdim=True),
            high_level_feat.mean(dim=1, keepdim=True)
        ], dim=1)
        spatial_gate = self.spatial_gate(spatial_context)

        low_level_feat = low_level_feat * (1.0 + low_gate * spatial_gate)
        high_level_feat = high_level_feat * (1.0 + high_gate * spatial_gate)
        return low_level_feat, high_level_feat


@MODELS.register_module()
class SRBranch(nn.Module):
    """Super Resolution Branch Module for enhancing small object detection.
    
    This module takes low-level and high-level features from the backbone,
    fuses them, and reconstructs a high-resolution image. It's used during
    training to improve feature quality for small objects, and is removed
    during inference to maintain computational efficiency.
    """
    
    def __init__(self, 
                 in_channels: int = 256, 
                 out_channels: int = 6, 
                 factor: int = 3, 
                 num_layers: int = 3,
                 intermediate_channels: int = 64,
                 attn_reduction: int = 4):
        """
        Args:
            in_channels (int): Input channels from concatenated features.
            out_channels (int): Output channels (3 for RGB, 1 for grayscale).
            factor (int): Super resolution scale factor.
            num_layers (int): Number of intermediate processing layers.
            intermediate_channels (int): Number of channels in intermediate layers.
        """
        super(SRBranch, self).__init__()
        self.factor = factor
        self.low_channels = in_channels // 2
        self.high_channels = in_channels - self.low_channels

        self.cross_scale_attention = CrossScaleAttention(
            self.low_channels,
            self.high_channels,
            reduction=attn_reduction
        )
        
        # Encoder: Process fused features
        self.encoder_conv = nn.Conv2d(in_channels, intermediate_channels, 
                                    kernel_size=3, padding=1)
        self.encoder_relu = nn.ReLU(inplace=True)
        
        # Middle processing layers
        middle_layers = []
        for i in range(num_layers):
            middle_layers.extend([
                nn.Conv2d(intermediate_channels, intermediate_channels, 
                         kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
        self.middle_processing = nn.Sequential(*middle_layers)
        
        # Decoder: Progressive upsampling
        decoder_layers = []
        current_channels = intermediate_channels
        
        # Calculate number of upsampling steps needed
        num_upsample_steps = int(torch.log2(torch.tensor(factor)).item())
        
        for step in range(num_upsample_steps):
            decoder_layers.extend([
                nn.Conv2d(current_channels, current_channels * 4, 
                         kernel_size=3, padding=1),
                nn.PixelShuffle(2),  # Upsample by 2x
                nn.ReLU(inplace=True)
            ])
            # After PixelShuffle, channels are reduced by 4x but we keep the same
            # current_channels remains the same for next step
        
        # Final convolution to get desired output channels
        decoder_layers.append(
            nn.Conv2d(intermediate_channels, out_channels, 
                     kernel_size=3, padding=1)
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, low_level_feat: torch.Tensor, 
                high_level_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low_level_feat, high_level_feat = self.cross_scale_attention(
            low_level_feat,
            high_level_feat
        )
        
        # Concatenate features along channel dimension
        fused_feat = torch.cat([low_level_feat, high_level_feat], dim=1)

        x = self.encoder_conv(fused_feat)
        x = self.encoder_relu(x)

        x = self.middle_processing(x)
        sr_feat = x
        
        # Decoder (progressive upsampling)
        sr_output = self.decoder(x)
        if sr_output.dim() == 3:
            sr_output = sr_output.unsqueeze(0)  
        return sr_output, sr_feat

    def compute_loss(self, sr_output: torch.Tensor, 
                    target: torch.Tensor, 
                    loss_weight: float = 0.1) -> torch.Tensor:

        
        if sr_output.dim() != target.dim():

            if sr_output.dim() == 3 and target.dim() == 4:
                sr_output = sr_output.unsqueeze(0)
        
        if sr_output.shape[2:] != target.shape[2:]:

            
            target = F.interpolate(
                target, 
                size=sr_output.shape[2:],  # 使用SR输出的空间尺寸
                mode='bilinear', 
                align_corners=False
            )

        # if sr_output.shape[1] != target.shape[1]:


        #     if target.shape[1] == 6 and sr_output.shape[1] == 3:
        #         target = target[:, :3, :, :]  # 取RGB通道

        
        return F.l1_loss(sr_output, target)