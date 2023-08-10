# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
# 倒入提前配置的cfg文件
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

"""
    根据.yaml文件生成一个cfg文件，然后
    根据cfg指定训练过程中、模型中、以及测试中使用到的参数
    
cfg:

DATALOADER:
  ASPECT_RATIO_GROUPING: True
  NUM_WORKERS: 4
  SIZE_DIVISIBILITY: 32
DATASETS:
  TEST: ('coco_2017_val',)
  TRAIN: ('coco_2017_train',)
INPUT:
  MAX_SIZE_TEST: 1333
  MAX_SIZE_TRAIN: 1333
  MIN_SIZE_RANGE_TRAIN: (-1, -1)
  MIN_SIZE_TEST: 800
  MIN_SIZE_TRAIN: (800,)
  PIXEL_MEAN: [102.9801, 115.9465, 122.7717]
  PIXEL_STD: [1.0, 1.0, 1.0]
  TO_BGR255: True
  
MODEL:
  BACKBONE:
    CONV_BODY: R-50-FPN-RETINANET
    FREEZE_CONV_BODY_AT: 2
    USE_GN: False
  CLS_AGNOSTIC_BBOX_REG: False
  DEVICE: cuda
  FBNET:
    ARCH: default
    ARCH_DEF: 
    BN_TYPE: bn
    DET_HEAD_BLOCKS: []
    DET_HEAD_LAST_SCALE: 1.0
    DET_HEAD_STRIDE: 0
    DW_CONV_SKIP_BN: True
    DW_CONV_SKIP_RELU: True
    KPTS_HEAD_BLOCKS: []
    KPTS_HEAD_LAST_SCALE: 0.0
    KPTS_HEAD_STRIDE: 0
    MASK_HEAD_BLOCKS: []
    MASK_HEAD_LAST_SCALE: 0.0
    MASK_HEAD_STRIDE: 0
    RPN_BN_TYPE: 
    RPN_HEAD_BLOCKS: 0
    SCALE_FACTOR: 1.0
    WIDTH_DIVISOR: 1
  FCOS:
    CENTERNESS_ON_REG: True
    CENTER_SAMPLING_RADIUS: 1.5
    FPN_STRIDES: [8, 16, 32, 64, 128]
    INFERENCE_TH: 0.05
    IOU_LOSS_TYPE: giou
    LOSS_ALPHA: 0.25
    LOSS_GAMMA: 2.0
    NMS_TH: 0.6
    NORM_REG_TARGETS: True
    NUM_CLASSES: 81
    NUM_CONVS: 4
    PRE_NMS_TOP_N: 1000
    PRIOR_PROB: 0.01
    USE_DCN_IN_TOWER: False
  FCOS_ON: True
  FPN:
    USE_GN: False
    USE_RELU: False
  GROUP_NORM:
    DIM_PER_GP: -1
    EPSILON: 1e-05
    NUM_GROUPS: 32
  KEYPOINT_ON: False
  MASK_ON: False
  META_ARCHITECTURE: GeneralizedRCNN
  RESNETS:
    BACKBONE_OUT_CHANNELS: 256
    DEFORMABLE_GROUPS: 1
    NUM_GROUPS: 1
    RES2_OUT_CHANNELS: 256
    RES5_DILATION: 1
    STAGE_WITH_DCN: (False, False, False, False)
    STEM_FUNC: StemWithFixedBatchNorm
    STEM_OUT_CHANNELS: 64
    STRIDE_IN_1X1: True
    TRANS_FUNC: BottleneckWithFixedBatchNorm
    WIDTH_PER_GROUP: 64
    WITH_MODULATED_DCN: False
  RETINANET:
    ANCHOR_SIZES: (32, 64, 128, 256, 512)
    ANCHOR_STRIDES: (8, 16, 32, 64, 128)
    ASPECT_RATIOS: (0.5, 1.0, 2.0)
    BBOX_REG_BETA: 0.11
    BBOX_REG_WEIGHT: 4.0
    BG_IOU_THRESHOLD: 0.4
    FG_IOU_THRESHOLD: 0.5
    INFERENCE_TH: 0.05
    LOSS_ALPHA: 0.25
    LOSS_GAMMA: 2.0
    NMS_TH: 0.4
    NUM_CLASSES: 81
    NUM_CONVS: 4
    OCTAVE: 2.0
    PRE_NMS_TOP_N: 1000
    PRIOR_PROB: 0.01
    SCALES_PER_OCTAVE: 3
    STRADDLE_THRESH: 0
    USE_C5: False
  RETINANET_ON: False
  ROI_BOX_HEAD:
    CONV_HEAD_DIM: 256
    DILATION: 1
    FEATURE_EXTRACTOR: ResNet50Conv5ROIFeatureExtractor
    MLP_HEAD_DIM: 1024
    NUM_CLASSES: 81
    NUM_STACKED_CONVS: 4
    POOLER_RESOLUTION: 14
    POOLER_SAMPLING_RATIO: 0
    POOLER_SCALES: (0.0625,)
    PREDICTOR: FastRCNNPredictor
    USE_GN: False
  ROI_HEADS:
    BATCH_SIZE_PER_IMAGE: 512
    BBOX_REG_WEIGHTS: (10.0, 10.0, 5.0, 5.0)
    BG_IOU_THRESHOLD: 0.5
    DETECTIONS_PER_IMG: 100
    FG_IOU_THRESHOLD: 0.5
    NMS: 0.5
    POSITIVE_FRACTION: 0.25
    SCORE_THRESH: 0.05
    USE_FPN: False
  ROI_KEYPOINT_HEAD:
    CONV_LAYERS: (512, 512, 512, 512, 512, 512, 512, 512)
    FEATURE_EXTRACTOR: KeypointRCNNFeatureExtractor
    MLP_HEAD_DIM: 1024
    NUM_CLASSES: 17
    POOLER_RESOLUTION: 14
    POOLER_SAMPLING_RATIO: 0
    POOLER_SCALES: (0.0625,)
    PREDICTOR: KeypointRCNNPredictor
    RESOLUTION: 14
    SHARE_BOX_FEATURE_EXTRACTOR: True
  ROI_MASK_HEAD:
    CONV_LAYERS: (256, 256, 256, 256)
    DILATION: 1
    FEATURE_EXTRACTOR: ResNet50Conv5ROIFeatureExtractor
    MLP_HEAD_DIM: 1024
    POOLER_RESOLUTION: 14
    POOLER_SAMPLING_RATIO: 0
    POOLER_SCALES: (0.0625,)
    POSTPROCESS_MASKS: False
    POSTPROCESS_MASKS_THRESHOLD: 0.5
    PREDICTOR: MaskRCNNC4Predictor
    RESOLUTION: 14
    SHARE_BOX_FEATURE_EXTRACTOR: True
    USE_GN: False
  RPN:
    ANCHOR_SIZES: (32, 64, 128, 256, 512)
    ANCHOR_STRIDE: (16,)
    ASPECT_RATIOS: (0.5, 1.0, 2.0)
    BATCH_SIZE_PER_IMAGE: 256
    BG_IOU_THRESHOLD: 0.3
    FG_IOU_THRESHOLD: 0.7
    FPN_POST_NMS_TOP_N_TEST: 2000
    FPN_POST_NMS_TOP_N_TRAIN: 2000
    MIN_SIZE: 0
    NMS_THRESH: 0.7
    POSITIVE_FRACTION: 0.5
    POST_NMS_TOP_N_TEST: 1000
    POST_NMS_TOP_N_TRAIN: 2000
    PRE_NMS_TOP_N_TEST: 6000
    PRE_NMS_TOP_N_TRAIN: 12000
    RPN_HEAD: SingleConvRPNHead
    STRADDLE_THRESH: 0
    USE_FPN: False
  RPN_ONLY: True
  USE_SYNCBN: False
  WEIGHT: catalog://ImageNetPretrained/MSRA/R-50
  
OUTPUT_DIR: .
PATHS_CATALOG: /home/nku524/dl/codebase/FCOS/fcos_core/config/paths_catalog.py
SOLVER:
  BASE_LR: 0.01
  BIAS_LR_FACTOR: 2
  CHECKPOINT_PERIOD: 2500
  DCONV_OFFSETS_LR_FACTOR: 1.0
  GAMMA: 0.1
  IMS_PER_BATCH: 16
  MAX_ITER: 90000
  MOMENTUM: 0.9
  STEPS: (60000, 80000)
  WARMUP_FACTOR: 0.3333333333333333
  WARMUP_ITERS: 500
  WARMUP_METHOD: constant
  WEIGHT_DECAY: 0.0001
  WEIGHT_DECAY_BIAS: 0
TEST:
  BBOX_AUG:
    ENABLED: False
    H_FLIP: False
    MAX_SIZE: 4000
    SCALES: ()
    SCALE_H_FLIP: False
  DETECTIONS_PER_IMG: 100
  EXPECTED_RESULTS: []
  EXPECTED_RESULTS_SIGMA_TOL: 4
  IMS_PER_BATCH: 8
"""
def train(cfg, local_rank, distributed):

    # 根据配置文件构建检测模型
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    # 模型数据移到GPU
    model.to(device)

    if cfg.MODEL.USE_SYNCBN:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 定义优化器
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    # 文件输出路径
    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # 开始训练
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
#     FCOS with improvements
#     MODEL:
#     META_ARCHITECTURE: "GeneralizedRCNN"
#     WEIGHT: "catalog://ImageNetPretrained/MSRA/R-50"
#     RPN_ONLY: True
#     FCOS_ON: True
#     BACKBONE:
#     CONV_BODY: "R-50-FPN-RETINANET"
#
# RESNETS:
# BACKBONE_OUT_CHANNELS: 256
# RETINANET:
# USE_C5: False  # FCOS uses P5 instead of C5
# FCOS:
# # normalizing the regression targets with FPN strides
# NORM_REG_TARGETS: True
# # positioning centerness on the regress branch.
# # Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042
# CENTERNESS_ON_REG: True
# # using center sampling and GIoU.
# # Please refer to https://github.com/yqyao/FCOS_PLUS
# CENTER_SAMPLING_RADIUS: 1.5
# IOU_LOSS_TYPE: "giou"
# DATASETS:
# TRAIN: ("coco_2017_train",)
# TEST: ("coco_2017_val",)
# INPUT:
# MIN_SIZE_TRAIN: (800,)
# MAX_SIZE_TRAIN: 1333
# MIN_SIZE_TEST: 800
# MAX_SIZE_TEST: 1333
# DATALOADER:
# SIZE_DIVISIBILITY: 32
# SOLVER:
# BASE_LR: 0.01
# WEIGHT_DECAY: 0.0001
# STEPS: (60000, 80000)
# MAX_ITER: 90000
# IMS_PER_BATCH: 16
# WARMUP_METHOD: "constant"

    parser.add_argument(
        "--config-file",
        default="/home/nku524/dl/codebase/FCOS/fcos/configs/fcos_imprv_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    """
        # 如果是多机多卡的机器，WORLD_SIZE代表使用的机器数，RANK对应第几台机器
        # 如果是单机多卡的机器，WORLD_SIZE代表有几块GPU，RANK和LOCAL_RANK代表第几块GPU
    """
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    """
        指定参数来源yaml文件 ： /home/nku524/dl/codebase/FCOS/fcos/configs/fcos_imprv_R_50_FPN_1x.yaml
        cfg融合文件中指定的参数
    """
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 输出文件目录： . 当前文件路径
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    """
        构建模型开始训练，并返回模型
    """
    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    """
        程序入口
    """
    main()
