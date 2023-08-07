import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes

"""
    对FCOS Head的输出进行后处理
"""
class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        """ pre_nms_thresh : 0.05 score小于0.05的都不要 """
        self.pre_nms_thresh = pre_nms_thresh  # 阈值来筛选正负样本，根据论文中所述，大于0.05的为正样本，其余的为负样本。然后看看一共有多少个正样本
        """ """
        self.pre_nms_top_n = pre_nms_top_n
        """ nms_thresh : 0.6 """
        self.nms_thresh = nms_thresh
        """ fpn_post_nms_top_n : 100 """
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        """ min_size : 0"""
        self.min_size = min_size
        """ num_classes : 81 """
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        """
            在进行NMS前处理一个batch中单个特征层所有点对应的预测结果，
            得到整个batch在该层中的预测结果，其中bbox坐标对应到输入图像空间
        """
        N, C, H, W = box_cls.shape

        # 将预测结果的shape进行处理以方便与locations进行对齐
        # locations的shape为(H*W,2) 其中每个是(x,y)形式
        # 代表该特征层每个点在输入图像的位置
        """
            都变成和locations一样的形式:
            最终：
            第一层：
            box_cls.shape : torch.Size([1, 18400, 80])
            box_regression.shape ： torch.Size([1, 18400, 4])
            centerness.shape : torch.Size([1, 18400])
        """
        # put in the same format as locations
        # (N,H,W,C)
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        # (N,H*W,C) 将logits经过sigmoid函数做多个二分类
        """
            box_cls.shape : torch.Size([1, 18400, 80])
        """
        box_cls = box_cls.reshape(N, -1, C).sigmoid()

        # (N,H,W,4)
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        # (N,H*W,4)
        box_regression = box_regression.reshape(N, -1, 4)

        # (N,H,W,1)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        # (N,H*W)
        centerness = centerness.reshape(N, -1).sigmoid()

        """
            # 在NMS前先根据分类得分进行筛选，选取大于0.05的那批
            # (N,H*W,C) bool True or False
            
            tensor([[[False, False, False,  ..., False, False, False],
                    [False, False, False,  ..., False, False, False],
                    [False, False, False,  ..., False, False, False],
                    ...,
                    [False, False, False,  ..., False, False, False]
        """
        candidate_inds = box_cls > self.pre_nms_thresh

        # 经过筛选后每张图片保留下来的候选目标数量
        # (N,H*W,C)->(N,H*W*C)->(N,) bool->int
        # pre_nms_top_n = candidate_inds.view(N, -1).sum(1)kl
        # 亲测以上那句会由于内存不连续而报错 于是我这里改成了以下
        # 这里reshape()等价于contigous().view()
        """
            过滤出来score只剩下234个大于0.05的
            pre_nms_top_n : tensor([234], device='cuda:0')
        """
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1)
        # (N,) 限制候选目标数量，默认最多是1000个
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        # 将分类得分乘上centerness分数 (N,H*W,C) x (N,H*W,1) = (N,H*W,C)
        # 这样就可以在后续NMS时'down-weight'那些远离物体中心的点产生的低质量框
        """
            box_cls.shape : torch.Size([1, 18400, 80])
            centerness[:, :, None].shape : torch.Size([1, 18400, 1])
        """
        # box_cls.shape : torch.Size([1, 18400, 80])
        box_cls = box_cls * centerness[:, :, None]

        # 其中每项是BoxList()实例，代表每张图像经过处理后的bboxes(坐标对应到输入图像空间)
        results = []

        # 依次处理各图
        for i in range(N):
            # (H*W,C) 该图中各特征点的分类得分 (乘上centerness之后的)
            per_box_cls = box_cls[i]
            # (H*W,C) bool True or False 指示各点在哪些类别上是候选
            per_candidate_inds = candidate_inds[i]
            """
                per_candidate_inds.shape : torch.Size([18400, 80])
            """
            # (per_candidate_inds.sum(),) 候选得分(一个点可能在多个类别上都是候选)
            """
                根据bool矩阵取出tensor中对应位置元素 (大于0.05过滤后又乘了centerness的数值)
                per_box_cls:
                        tensor([0.0115, 0.0126, 0.0227, 0.0260, 0.0194, 0.0181, 0.0317, 0.0207, 0.0192,
                                0.0106, 0.0129, 0.0263, 0.0230, 0.0370, 0.0450, 0.0412, 0.0304, 0.0663,
                                0.0827, 0.0722, 0.0294, 0.0371, 0.0733, 0.0985, 0.0886, 0.0382,...
            """
            per_box_cls = per_box_cls[per_candidate_inds]

            # (per_candidate_inds.sum(),2)
            # 每项：第1个指示特征点索引(flattened)，第2个指示类别索引
            """
                per_candidate_nonzeros:
                tensor([[10221,     4],
                        [10222,     4],
                        [10405,     4],
                        ...
                        [13060,     7],
                        [13061,     4],
                        [13061, ...
                        
                 # 注意，一个特征点位置可能在多个类别上都成为候选
                 # 特征点索引(flattened) (per_candidate_inds.sum(),) bool 指示哪些特征点位置是候选
            """

            per_candidate_nonzeros = per_candidate_inds.nonzero()

            """
                per_box_loc:
                tensor([10221, 10222, 10405, 10406, 11576, 11577, 11760, 11761, 11768, 11943,
                        11944, 11952, 12322, 12323, 12324, 12325, 12506, 12507, 12508, 12509,
                        12510, 12690, 12691, 12692, 12693, 12694, 12874, 12875, 12876,  ...
            """
            per_box_loc = per_candidate_nonzeros[:, 0]
            # 前景类别索引 (per_candidate_inds.sum(),) bool 指示1个特征点在哪些类别上是候选

            # +1是因为网络结构中，在分类预测的卷积层输出通道数里并未包含背景类，因此这里将0留给背景
            per_class = per_candidate_nonzeros[:, 1] + 1

            # 该图中各点回归的预测结果 (H*W,4)
            per_box_regression = box_regression[i]

            # 注意，这里可能包含了多次同一个点的位置
            # (per_candidate_inds.sum(),4) 候选特征点回归的预测结果
            """
                per_box_regression.shape : torch.Size([234, 4])
            """
            per_box_regression = per_box_regression[per_box_loc]
            # (per_candidate_inds.sum(),2) 候选特征点在输入图像的位置 每个的形式是(x,y)
            """
                per_locations.shape : torch.Size([234, 2])
            """
            per_locations = locations[per_box_loc]

            # 该图候选目标数量的上限值 (1,) 234
            per_pre_nms_top_n = pre_nms_top_n[i]

            # 若当前候选目标(得分>0.05的那些)数量(包括一个点在多个类别上成为候选)超过了上限值则进行截断
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                # 进一步选择得分最高的那批
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                # 这里之前先判断了一下正样本数是否大于1000个，大于1000个则利用 torch.topk函数（参考pytorch -- topk()）来按顺序选取前1000个样本，不按大小排序。

                # torch.Size([234]) (per_pre_nms_top_n,) 候选目标点预测的类别 其中每个都是前景的类别(从1开始)
                per_class = per_class[top_k_indices]
                # torch.Size([234, 4]) (per_pre_nms_top_n,4) 候选目标点回归的预测结果
                per_box_regression = per_box_regression[top_k_indices]
                # torch.Size([234, 2]) (per_pre_nms_top_n,2) 候选目标点在输入图像的位置 其中每个都是(x,y)形式
                per_locations = per_locations[top_k_indices]

            # 由回归的4个量计算出bbox两个对角的坐标x1y1x2y2(对应到输入图像)
            # 记num_candidates_i = min(per_pre_nms_top_n, per_candidate_inds.sum())
            # (num_candidates_i,4)
            detections = torch.stack([
                # x1=x-l
                per_locations[:, 0] - per_box_regression[:, 0],
                # y1=y-t
                per_locations[:, 1] - per_box_regression[:, 1],
                # x2=x+r
                per_locations[:, 0] + per_box_regression[:, 2],
                # y2=y+b
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)
            # 输入图像的高、宽
            h, w = image_sizes[i]

            # 实例化BoxList()对象
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            # 预测类别
            boxlist.add_field("labels", per_class)
            # 预测分数
            # 开方是因为分类得分乘上了centerness分数，因此这里做了个数值尺度的还原(否则得分是1e-2级别的)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            # 将bbox坐标限制在输入图像尺寸范围内 remove_empty=False代表不对bbox的坐标作检查(即不要求x2>x1&y2>y1)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            # 过滤掉尺寸较小的bbox 实际的实现是保留边长不小于self.min_size的那批
            # self.min_size默认为0 于是在这个条件下 就相当于对以上那句设置了remove_empty=True
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """

        """
            得到的sampled_boxes 就是FPN每一层得到的bbox坐标，类别，置信度，封装在了 BoxList这个类中:
            0:BoxList(num_boxes=1000,image_width=1066,image_height=800,mode=xyxy) 
            1:BoxList(num_boxes=984,image_width=1066,image_height=800,mode=xyxy)
            2:BoxList(num_boxes=492,image_width=1066,image_height=800,mode=xyxy) 
            3:BoxList(num_boxes=138,image_width=1066,image_height=800,mode=xyxy) 
            4:BoxList(num_boxes=14,image_width=1066,image_height=800,mode=xyxy)
        """
        sampled_boxes = []
        """
            遍历每一个特征层，共5个
        """
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes
                )
            )

        """
            经过所有层（5层）的单个处理过程之后，结果都保存在了sampled_boxes当中，其为list=5的列表，里面是5个BoxList类。然后
            boxlists = [cat_boxlist(boxlist) for boxlist in boxlists] 引出下面cat_boxlist函数。
        """
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.

    """
        对处理后的result还要进一步过滤,一张图片检测出物体的限制在100以下: 
        就拿某张图片来说，result的长度是478代表检测出了478个物体，需要过滤:
    """
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        # 依次处理各张图像
        for i in range(num_images):
             # multiclass nms
            # NMS后返回的还是BoxList()实例
            """
                将所有层的bbox concate经过nms处理
            """
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            # NMS后剩余的目标数量
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            # 若经过NMS后该图的目标数量超出了上限(默认100)，则进行截断
            """
                对处理后的result还要进一步过滤,一张图片检测出物体的限制在100以下,
                就拿某张图片来说，result的长度是478代表检测出了478个物体，需要过滤:
            """
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                # 得分已排好序
                cls_scores = result.get_field("scores")
                # 取排在超出数量限制的那个点对应的得分作为阀值
                # 这样就能限制目标数量，最终获取得分最高的前100个
                """
                    其中#y, i = torch.kthvalue(x, k, n) 沿着n维度返回第k小值和下标，
                    找到这个排名 number_of_detections - self.fpn_post_nms_top_n + 1 小的分数，
                    也就是第101位大 的，得到最后得到100个bbox
                """
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    # 默认0.05 若一个预测结果在分类得分上连0.05都不足则剔除掉
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    # 默认1000 NMS前每张图仅保留得分最高的前1000个预测结果
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    # 默认0.6 NMS阀值，0.6以上则视为重叠框
    nms_thresh = config.MODEL.FCOS.NMS_TH
    # 默认100 NMS后每张图片最多保留多少个目标
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    # 是否采用测试增强(TTA) 默认False
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    # 初始化一个FCOSPostProcessor
    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,  # 默认81，包括背景
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
