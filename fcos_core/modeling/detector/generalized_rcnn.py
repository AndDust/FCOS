# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from fcos_core.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        """
            build_backbone()函数调用自fcos_core.modeling.backbone
            build_rpn()函数调用自fcos_core.modeling.rpn.rpn
            build_roi_heads()函数调用自fcos_core.modeling.roi_heads.roi_heads
        """
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        """
            images.tensors.shape : torch.Size([16, 3, 1344, 800])
            # 将图像batch数据转化成image_list类型数据
        """
        images = to_image_list(images)
        """
            将一个batch图像送入backbone得到预测特征层
            len(features) : 5
            type(features) : <class 'tuple'>
            
            features[0].shape : torch.Size([16, 256, 168, 100])
            features[1].shape : torch.Size([16, 256, 84, 50])
            features[2].shape : torch.Size([16, 256, 42, 25])
            features[3].shape : torch.Size([16, 256, 21, 13])
            features[4].shape : torch.Size([16, 256, 11, 7])
        """
        features = self.backbone(images.tensors)
        """
            
        """
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
