"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


INF = 100000000 # 1e8


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

"""
    Loss计算的操作使用一个类封装了起来，注意，但它就是普通的类，
    没有继承torch.nn.Module,没有forward方法，但是定义了__call__方法，所以可以直接调用
    为什么不继承torch.nn.Module ?
"""
class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """
    """
        在初始化方法中，主要就是设置计算分类、回归、centerness loss相关的东西
    """
    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,  # 默认2.0
            cfg.MODEL.FCOS.LOSS_ALPHA # 默认0.25
        )
        # 各层特征相对于输入图像的下采样倍数 [8, 16, 32, 64, 128]
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        # 中心采样半径 通常是1.5 它会乘以各层的下采样步长

        # 样本候选是位于gt
        # box内的位置点，而center
        # sampling这个策略就是在这一步中加强了约束：只有在x,y
        # 方向上与物体中心距离都在一定范围内的位置点才是正样本候选。实际产生的效果就是相当于把gt
        # box缩小了，只有位于这个缩小的box内的位置点才是正样本候选。
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        # IoU Loss 类型："iou", "linear_iou" or "giou"
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE

        """
            # 是否使用normalizing regression targets策略
            # (会对回归标签使用下采样步长进行归一化)
        """
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        # 注意这里的reduction，因为在真正计算这项loss时要除以正样本数量去求均值
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        在中心采样策略下，在一张图片中，判断(所有特征层)每个特征点属于哪些目标物体的正样本，返回一个shape为
        (num_points,num_gts)的mask，mask[i][j]=True代表特征点i是目标物体j的正样本。
        '''
        # (一张图中的)物体数量 gt的shape是(num_gts,4) 每个gt box是x1y1x2y2形式
        num_gts = gt.shape[0]
        # gt_xs, gt_ys是一张图片中所有特征层的特征点在输入图像上的位置
        # shape均是 (K,)
        # (所有特征层)特征点数量
        K = len(gt_xs)
        # 1. 变换维度，使得每个特征点与每个目标物体对应，方便计算

        # expand一下shape，使得每个特征点都可对应到每个目标物体 方便后续进行计算
        # (num_gts,4)->(1,num_gts,4)->(K,num_gts,4)
        gt = gt[None].expand(K, num_gts, 4)
        # 物体中心x坐标 (K,num_gts)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        # 物体中心y坐标 (K,num_gts)
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        # 若图中没有目标物体，则所有特征点都是负样本，返回一个全0的mask
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            # (K,)
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0

        # 分别对各特征层进行处理 得到每个物体在各特征层对应的采样区域
        # 其实设置的半径值就是一个数字，只不过将其乘上各特征层对应的下采样步长后就得到了不同的采样区域范围
        # 这样，同一个目标物体在不同的特征层都有不同的采样区域
        # level, n_p: 特征层索引、特征点数量
        for level, n_p in enumerate(num_points_per):
            # 一个指针，与beg类似，作为各层特征点的末端索引(不包括其本身指向的位置)
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    """
        # points是list len=num_levels 其中每项是一个特征层中所有点对应到输入图像的位置 shape是(h_i*w_i,2) 位置形式是(x,y)
            points[0].shape : torch.Size([16800, 2])
        # targets是list len=num_images(batch size) 其中每项是BoxList()实例
            len(targets) : 16
    """
    def prepare_targets(self, points, targets):
        """为特征点分配标签 返回两个list，len=num_levels，分别代表所各特征层所有点一个批次的类别和回归标签"""
        """为特征点分配标签 返回两个list，len=num_levels，
        分别代表所各特征层所有点一个批次的类别和回归标签"""

        # 1. 为每个特征点分配其负责回归的目标尺寸

        # 各特征层负责回归的目标尺寸(bbox边长)的上下限
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            # INF是1e8
            [512, INF],
        ]
        # 将以上扩展到shape与特征点一一对齐 len=num_levels

        # 即每个特征点负责回归的目标尺寸(bbox边长)的上下限
        """
            expanded_object_sizes_of_interest[0]
                tensor([[-1., 64.],
                        [-1., 64.],
                        [-1., 64.],
                            ...,
                        [-1., 64.]
            扩展上面的尺寸，存放每一层的变长上下限值
        """
        expanded_object_sizes_of_interest = []

        for l, points_per_level in enumerate(points):
            # 代码一行写不下可以加个\
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        # 2. concat 拼接所有特征层的结果，方便计算

        # (num_points_all_levels,2) 将所有层的结果拼接在一起
        """
            expanded_object_sizes_of_interest.shape : torch.Size([22400, 2])
        """
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)

        # len=num_levels 各层的特征点数量 h_i*w_i
        """
            各层的特征点数量 h_i*w_i
            num_points_per_level : [16800, 4200, 1050, 273, 77]
        """
        num_points_per_level = [len(points_per_level) for points_per_level in points]

        # 将各层的特征点数量记录到类属性中 因为后续标签分配的实现是在另一个方法里
        self.num_points_per_level = num_points_per_level

        # 所有层的特征点在输入图像的位置 (num_points_all_levels,2)
        """
            把所有层特征点进行拼接
            points_all_level.shape : torch.Size([22400, 2])
        """
        points_all_level = torch.cat(points, dim=0)

        # 3. assignents 标签分配 得到各张图片(所有特征点)的结果

        # 计算每个特征点对应的标签： classification、regression
        # len(labels) = len(reg_targets) = num_images(batch size)
        # 其中每个分别是(num_points_all_levels,) (num_points_all_levels,4)
        """
            points_all_level.shape : torch.Size([22400, 2])
            expanded_object_sizes_of_interest.shape : torch.Size([22400, 2])
        """
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        # 4. 在每张图片中，将各特征层的标签区分开

        # i代表第i张图片的结果
        # labels[i]、reg_targets[i] 都是list，其中每项代表对应特征层中每个点的类别和回归标签
        for i in range(len(labels)):
            # [(num_points_level_1,), (num_points_level_2,), ..]
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            # [(num_points_level_1,4), (num_points_level_2,4), ..]
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        # 5. 将同一层的(所有图片的)结果拼接在一起

        # 以下两个list, len=num_levels
        # 同一层中所有图片的特征点类别标签
        labels_level_first = []
        # 同一层中所有图片的特征点回归标签
        reg_targets_level_first = []

        # 对于各张图片，同一层特征的尺寸一致，也就是特征点的数量一致，因此可以在第一个维度拼接
        for level in range(len(points)):
            # 该特征层所有图片的特征点类别标签
            labels_level_first.append(
                # (num_images*num_points_this_level,)
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            # 该特征层所有图片的特征点回归标签
            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            # normalizing regression targets策略 使用下采样步长对回归标签进行归一化
            # 对应地，这时候网络预测的回归量乘以scale后会经过ReLU()而非exp()
            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    """
        locations.shape : torch.Size([22400, 2])
        object_sizes_of_interest.shape : torch.Size([22400, 2])
        
        # locations是所有层的特征点在输入图像的位置 (num_points_all_levels,2)
        # object_sizes_of_interest是各特征点负责回归的目标尺度(bbox边长)范围 (num_points_all_levels,2)
        # targets是一个批次中所有图像的标签 len=num_images(batch size) 其中每项是BoxList()实例
    """
    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):

        # 以下两个list，len=num_images
        # 里面每项是一张图中所有特征点的类别标签 (num_points_all_levels,)
        labels = []
        # 里面每项是一张图中所有特征点的回归标签 (num_points_all_levels,4)
        reg_targets = []

        # 各特征点的x,y坐标
        # (num_points_all_levels,) (num_points_all_levels,)
        xs, ys = locations[:, 0], locations[:, 1]

        # 分别处理每张图像 i代表第i张图像
        for im_i in range(len(targets)):
            # 1. 利用每个特征点在输入图像的坐标与一张图中所有gt boxes的坐标初步计算出回归标签

            # 一张图的标签是一个BoxList实例
            targets_per_im = targets[im_i]
            # bbox坐标形式需要是x1y1x2y2
            assert targets_per_im.mode == "xyxy"
            # (num_objects_i,4) gt boxes
            bboxes = targets_per_im.bbox
            # (num_objects_i,) gt labels
            labels_per_im = targets_per_im.get_field("labels")
            # (num_objects_i,) 该图中所有gt boxes的面积
            area = targets_per_im.area()

            # 以下为两两交互计算 得到的shape均为(num_points_all_levels,num_objects_i)

            """
                计算所有特征点（五层）与一张图片上所有GT box边框的距离：
            """
            # l = x - x1 每个特征点与每个bbox左边框的距离
            l = xs[:, None] - bboxes[:, 0][None]
            # t = y - y1 每个特征点与每个bbox上边框的距离
            t = ys[:, None] - bboxes[:, 1][None]
            # r = x2 - x 每个特征点与每个bbox右边框的距离
            r = bboxes[:, 2][None] - xs[:, None]
            # b = y2 - y 每个特征点与每个bbox下边框的距离
            b = bboxes[:, 3][None] - ys[:, None]

            """
                # (num_points_all_levels,num_objects_i,4) 一张图片中所有特征点的回归标签
                reg_targets_per_im.shape : torch.Size([22400, 3, 4])
            """
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            # 2. 筛选出正样本候选 依据特征点必须在gt boxes内 或 物体中心的采样范围内

            # 若开启了中心采样策略，则只有距离物体中心点一定半径范围内(x,y方向上均不超过一定距离)的点才是正样本
            # 这种做法加强了约束，要求正样本位置距离物体中心更近，而非仅仅在gt box内
            if self.center_sampling_radius > 0:
                # 其中每项代表特征点j是否在物体i的中心采样区域内
                # (num_points_all_levels,num_objects_i) bool
                is_in_boxes = self.get_sample_region(
                    # (num_objects_i,4) 一张图片中的gt boxes
                    bboxes,
                    # [8,16,32,64,128] 各层下采样步长
                    self.fpn_strides,
                    # [h_1*w_1, h_2*w_2, ..] 各层特征点数量
                    self.num_points_per_level,
                    # (num_points_all_levels,) (num_points_all_levels,)
                    xs, ys,
                    # 通常设置为1.5
                    radius=self.center_sampling_radius
                )
            # 否则，只要点位于gt boxes内的特征点均为正样本
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                # (num_points_all_levels,num_objects_i) bool
                # 其中每项代表特征点j是否在物体i的gt box内 只需要min(l,t,r,b) > 0
                """
                    is_in_boxes :
                    tensor([[False, False],
                            [False, False],
                            [True, False],
                            ...,
                            [False, False],
                            [False, False],
                            [False, False]]
                """
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            # 3. 筛选出正样本 依据各层特征负责回归不同尺度的目标物体

            # (num_points_all_levels,num_objects_i) 特征点距离gt box 4条边的最大距离
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]

            # limit the regression range for each location
            """
                # 特征点距离gt box 4条边的最大距离必须在其所属特征层负责回归的尺度范围内
                # 其实背后代表的意义就是那一层特征负责预测的目标尺寸(gt boxes边长)必须在一定范围内
                # (num_points_all_levels,num_objects_i)
                
                object_sizes_of_interest[:, [0]]
                                    tensor([[ -1.],
                                    [ -1.],
                                    [ -1.],
                                    ...,
                                    [512.],
                                    [512.],
                                    [512.]], device='cuda:0')
                
                object_sizes_of_interest[:, [1]]
                                    tensor([[6.4000e+01],
                                    [6.4000e+01],
                                    [6.4000e+01],
                                    ...,
                                    [1.0000e+08],
                                    [1.0000e+08],
                                    [1.0000e+08]], device='cuda:0')
                
                假设一个特征点对应某个bbox，那么判断这个特征点是否属于这个层级
                is_cared_in_the_level.shape : torch.Size([20267, 3])
                                    tensor([[False, False, False],
                                    [False, False, False],
                                    [False, False, False],
                                    ...,
                                    [False,  True,  True],
                                    [False,  True,  True]]
            """

            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            # 将每个物体的面积的shape扩展以方便计算 使得每个特征点对应一张图中所有物体面积
            # (num_points_all_levels,num_objects_i)
            """
                locations_to_gt_area：
                tensor([[ 10303.9453, 222432.5938,  67902.7734],
                        [ 10303.9453, 222432.5938,  67902.7734],
                        [ 10303.9453, 222432.5938,  67902.7734],
                        ...,
                        [ 10303.9453, 222432.5938,  67902.7734],
            """
            locations_to_gt_area = area[None].repeat(len(locations), 1)

            """
                # 将负样本（1.特征点不在任何bbox内 2.即使在某bbox内，但是最大距离超过了这个层级应该的限制。那么都是负样本）特征点对应的gt boxes面积置为无穷
                为什么要把特征点对应的gt面积置为无穷？
            """
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # 4. 若一个正样本对应到多个gt，则选择gt box面积最小的去负责预测

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            # 选出每个特征点对应的面积最小的gt box
            # 最小面积：(num_points_all_levels,） gt索引：(num_points_all_levels,）
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            # (num_points_all_levels,4) 每个特征点取面积最小的gt box作为回归目标 注意这里包含了负样本
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            # (num_points_all_levels,) 每个特征点对应的类别标签
            labels_per_im = labels_per_im[locations_to_gt_inds]
            # 负样本的类别标签设置为背景类
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

        """
            第一张图片所有特征点对应的标签参数（背景0，分类）、回归参数：
            labels[0].shape : torch.Size([22400])
            reg_targets[0].shape : torch.Size([22400, 4])
        """
        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    """
        生成一套locations就够了，给16张图片使用
        locations[0].shape : torch.Size([16800, 2])
        locations是每一层在原图尺寸上特征点对应的位置!
        
        box_cls[0].shape : torch.Size([16, 80, 100, 168])
        box_regression[0].shape : torch.Size([16, 4, 100, 168])
        centerness[0].shape : torch.Size([16, 1, 100, 168])
    """
    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        """
            box_cls[0].shape : torch.Size([16, 80, 100, 168])
            N : batch大小 : 16
            num_classes : 80
        """
        N = box_cls[0].size(0)
        # (前景)类别数量
        num_classes = box_cls[0].size(1)

        """
            # 1. 标签分配

            # 两个list，len=num_levels，
            # list中的每项是对应特征层所有图片的特征点的类别标签和回归标签 shape分别为:
            # (num_images * num_points_level_l,) (num_images * num_points_this_level_l,4)
            
            labels:
            labels[0].shape : torch.Size([243200])
            labels[1].shape : torch.Size([60800])
            labels[2].shape : torch.Size([15200])
            labels[3].shape : torch.Size([3952])
            labels[4].shape : torch.Size([1120])
            
            reg_targets:
            reg_targets[0].shape : torch.Size([243200, 4])
            reg_targets[1].shape : torch.Size([60800, 4])
            reg_targets[2].shape : torch.Size([15200, 4])
            reg_targets[3].shape : torch.Size([3952, 4])
            reg_targets[4].shape : torch.Size([1120, 4])
            
        """
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        # 以下这批list的长度都等于特征层数量 len=num_levels
        # box_cls_flatten = []
        # box_regression_flatten = []
        # centerness_flatten = []
        #
        # labels_flatten = []
        # reg_targets_flatten = []

        # 依次处理各个特征层 以下注释中i代表第i层特征
        # for l in range(len(labels)):
        #     # (N,num_classes,H_i,W_i)->(N,H_i,W_i,num_classes)->(N*H_i*W_i,num_classes)
        #     box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
        #     # (N,4,H_i,W_i)->(N,H_i,W_i,4)->(N*H_i*W_i,4)
        #     box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
        #     # (N,1,H_i,W_i)->(N*H_i*W_i)
        #     centerness_flatten.append(centerness[l].reshape(-1))
        #
        #     # (N*H_i*W_i)
        #     labels_flatten.append(labels[l].reshape(-1))
        #     # (N*H_i*W_i,4)
        #     reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))

        # 2.1 permute & flatten

        # 以上注释掉的部分是原作写法，个人感觉用列表生成式更简洁看起来舒服些
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))
        # 2.2 concat

        # 将所有特征层(所有图片)的预测结果拼接在一起
        # (num_points_all_levels_batches,num_classes)
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        # (num_points_all_levels_batches,4)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        # (num_points_all_levels_batches,)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)

        # 将所有特征层(所有图片)的标签拼接在一起
        # (num_points_all_levels_batches,)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        # (num_points_all_levels_batches,4)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        # 3. 获取正样本的回归预测和centerness预测(因为回归损失和centerness损失仅对正样本计算)

        # (num_pos,) 正样本(特征点)索引
        # torch.nonzero(labels_flatten > 0)返回的shape是(num_points_all_levels_batches,1)
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # (num_pos,4) 正样本对应的回归预测
        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        # (num_pos,) 正样本对应的centerness预测
        centerness_flatten = centerness_flatten[pos_inds]
        # (num_pos,4)
        # reg_targets_flatten = reg_targets_flatten[pos_inds]

        # 4. 计算分类损失(正负样本都要计算)

        # 在所有GPU上进行同步，使得每个GPU得到相同的正样本数量，是一个同步操作
        num_gpus = get_num_gpus()
        # sync num_pos from all gpus

        # 所有gpu上正样本数量的总和
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        # 所有gpu上正样本数量的均值
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        # 计算分类损失 多分类Focal Loss
        cls_loss = self.cls_loss_func(
            # (num_points_all_levels_batches,num_classes)
            box_cls_flatten,
            # (num_points_all_levels_batches,) 注意转换成int
            labels_flatten.int()
        ) / num_pos_avg_per_gpu
        # 若该批次中有正样本，则进一步计算回归与centerness损失
        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
