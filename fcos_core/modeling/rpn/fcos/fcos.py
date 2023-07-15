import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d

"""
    in_channels ： 256
"""
class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        """
            num_classes ： 80
            self.fpn_strides ： [8, 16, 32, 64, 128]
            self.norm_reg_targets ： True
            self.centerness_on_reg : True
            self.use_dcn_in_tower : False
        """
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER

        """
            # 分类分支
        """
        cls_tower = []
        """
            # 回归分支
        """
        bbox_tower = []
        """
            cfg.MODEL.FCOS.NUM_CONVS : 4 
            # 默认有4个卷积块
        """
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d
            """
                                ———————> 4 *(conv2d + GN +Relu) ———————> conv2d # h *w *80 80个类别分数
                               |
                               |
                features—————> |                                  
         (P3, P4, P5, P6, P7)  |                                 ———————> conv2d h *w *4 4个距离参数
                               |                                |
                                ———————> 4 *(conv2d + GN +Relu) ————>|
                                                                |
                                                                 ———————> conv2d h *w *1 远近程度参数（用于后处理）
            """
            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        # 1个卷积层预测目标类别
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        # 1个卷积层预测回归的4个量
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        # 1个卷积层预测centerness值
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization 权重初始化
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # 这个Scale module的前向过程就是将输入张量乘上相应的scale值(初始化为1.0)
        # scales是为各层设置的可学习标量，用于乘上回归预测量
        # 因为各层特征共享检测头而它们负责回归的尺度是不一样的，因此分别需要一个可学习变量去自适应相应的尺度
        # class Scale(nn.Module):
        #     def __init__(self, init_value=1.0):
        #         super(Scale, self).__init__()
        #         self.scale = nn.Parameter(torch.FloatTensor([init_value]))
        #
        #     def forward(self, input):
        #         return input * self.scale

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        """
            x[0].shape : torch.Size([16, 256, 100, 156])
            x[1].shape : torch.Size([16, 256, 50, 78])
            x[2].shape : torch.Size([16, 256, 25, 39])
            x[3].shape : torch.Size([16, 256, 13, 20])
            x[4].shape : torch.Size([16, 256, 7, 10])
        """
    def forward(self, x):
        # 各个特征层的分类、回归、中心度预测结果
        # 以i代表第几个特征层
        # 其中每个是(b,num_fg_classes,h_i,w_i)
        logits = []
        bbox_reg = []
        centerness = []
        # 对每个level的特征图进行遍历
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            # 中心度和回归分支一起预测
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))
            # 预测回归结果，网络输出要乘以特征层对应的scale值
            bbox_pred = self.scales[l](self.bbox_pred(box_tower))

            """
                需要“打起精神”的地方就是记得将网络回归部分的输出乘以scale，
                还有就是根据情况选择使用ReLU()还是exp()去将回归的4个量映射到正值。
            """
            if self.norm_reg_targets:
                # 在归一化回归标签的策略下，是将回归的预测结果经过ReLU()形成非负值
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    # 如果是推理阶段，则要将回归值乘以该特征层相应的下采样步长以解码到输入图像空间中
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                # naive版本下是通过exp()函数将回归的4个量映射到正值
                # exp()本身就拥有将小的数值映射到较大的数值的能力
                # 因此这中方式下标签也不需要归一化
                bbox_reg.append(torch.exp(bbox_pred))
        """
            logits[0].shape : torch.Size([16, 80, 100, 152])
            bbox_reg[0].shape : torch.Size([16, 4, 100, 152])
            centerness[0].shape : torch.Size([16, 1, 100, 152])
        """
        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()
        # 检测头部，通过它获取分类、回归以及centerness的预测结果
        head = FCOSHead(cfg, in_channels)
        # 推理时的后处理模块
        box_selector_test = make_fcos_postprocessor(cfg)
        # 训练时计算loss的模块
        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

        # FPN各特征层对应的下采样步长 [8,16,32,64,128]
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        """
            模型会将各层特征通过检测头(head)得到预测结果，然后，
            如果当前是训练阶段，则使用 loss_evaluator 计算损失；
            否则，使用 box_selector_test 进行后处理，得到筛选后的检测框。
        """

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        """
            # 3个list 分别代表分类、回归、centerness的预测结果，每个list包含各特征层对应的预测结果
            # 3个list中的每项shape分别是：(b,num_fg_classes,h_i,w_i) (b,4,h_i,w_i) (b,1,h_i,w_i)
            # i代表第i层特征
        """
        box_cls, box_regression, centerness = self.head(features)
        """
            # list 其中每项代表对应特征层中每点在输入图像中对应的位置: (x,y)形式
            # 每项的shape是(h_i*w_i,2)
        """
        """
            计算出各特征点对应于输入图像中的位置，以便于在训练中计算loss和推理中解码出预测框位置
            locations:
                tensor([
                [   4.,    4.],
                [  12.,    4.],
                [  20.,    4.],
                ...,
                [1228.,  796.],
                [1236.,  796.],
                [1244.,  796.]], device='cuda:0')
        """
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        # 3个loss，均为标量
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        # 之所以返回两项是为了和以下_forward_test()方法返回的形式保持一致
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        # 经过后处理筛选后得到的检测框
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        """
            分别计算各层特征中每点在输入图像中对应的位置。
            features是来自FPN的输出
        """
        # len=5 代表5层特征中每点对应到输入图像中的位置(x,y)
        locations = []
        for level, feature in enumerate(features):
            # 该层特征图的高与宽
            h, w = feature.size()[-2:]
            # 计算该特征中每个点在输入图像中对应的位置
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        """
            计算一层特征中每点在输入图像中对应的位置
            # stride是该层特征相对于输入图像的下采样步长，
            # 因此每点映射到输入图像中它们的x,y坐标应该相互间隔这个步长值
        """
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
