# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead
import segmentation_models_pytorch as smp


@HEADS.register_module()
class SMP_Unet_Head(BaseDecodeHead):
    """SMP Unet Decode Head
    """

    def __init__(self,
                 encoder_channels,
                 decoder_channels,
                 n_blocks,
                 use_batchnorm,
                 attention_type,
                 **kwargs):
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.n_blocks = n_blocks
        self.use_batchnorm = use_batchnorm
        self.attention_type = attention_type
        super(SMP_Unet_Head, self).__init__(input_transform='multiple_select', **kwargs)
        self.decode_head = smp.unet.decoder.UnetDecoder(
            encoder_channels=self.encoder_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.n_blocks,
            use_batchnorm=self.use_batchnorm,
            center=self.align_corners,
            attention_type=self.attention_type,
        )

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        feats = self.decode_head(*inputs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
