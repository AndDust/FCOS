# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from fcos_core.modeling import registry
from fcos_core.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import mobilenet
from . import swin


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("MNV2-FPN-RETINANET")
def build_mnv2_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)
    in_channels_stage2 = body.return_features_num_channels
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2[1],
            in_channels_stage2[2],
            in_channels_stage2[3],
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("SWIN-T-FPN")
def build_mnv2_fpn_backbone(cfg):
    body = swin.swin_tiny_patch4_window7_224()
    in_channels_stage2 = 96
    out_channels = 96
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    """
        cfg.MODEL.BACKBONE.CONV_BODY ： 'R-50-FPN-RETINANET'
        
        根据cfg配置的conv_body返回backbone,该backbone是之前就注册到registry.BACKBONES这个注册器中的一系列build方法，返回的都是model
        所以这里使用时，直接根据字符串'R-50-FPN-RETINANET'将model提取出来即可。
        
        registry.BACKBONES:
        {'R-101-C5': <function build_resnet_backbone at 0x7f70076e5620>, 
        'R-101-C4': <function build_resnet_backbone at 0x7f70076e5620>, 
        'R-50-C5': <function build_resnet_backbone at 0x7f70076e5620>, 
        'R-50-C4': <function build_resnet_backbone at 0x7f70076e5620>, 
        'R-152-FPN': <function build_resnet_fpn_backbone at 0x7f70076e5598>, 
        'R-101-FPN': <function build_resnet_fpn_backbone at 0x7f70076e5598>, 
        'R-50-FPN': <function build_resnet_fpn_backbone at 0x7f70076e5598>, 
        'R-101-FPN-RETINANET': <function build_resnet_fpn_p3p7_backbone at 0x7f70076e5510>, 
        'R-50-FPN-RETINANET': <function build_resnet_fpn_p3p7_backbone at 0x7f70076e5510>, 
        'MNV2-FPN-RETINANET': <function build_mnv2_fpn_backbone at 0x7f70076e5158>, 
        'FBNet': <function add_conv_body at 0x7f70076aa9d8>}
    """
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
