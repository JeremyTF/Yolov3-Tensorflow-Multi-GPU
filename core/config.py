#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "/media/jiangshengjie/Elements/tensorflow-yolov3/trainv2.names"
__C.YOLO.ANCHORS                = "/media/jiangshengjie/Elements/tensorflow-yolov3/data/anchors/basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "/home/jiangshengjie/tensorflow-yolov3/tensorflow-yolov3/checkpoint_train/yolov3_test_loss=601.2548.ckpt-1"
__C.YOLO.DEMO_WEIGHT            = "/home/jiangshengjie/tensorflow-yolov3/tensorflow-yolov3/checkpoint_train/yolov3_test_loss=601.2548.ckpt-1"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "/media/jiangshengjie/Elements/New_yizhuang_MC3D/home/wangning/Desktop/data/New_yizhuang/labels_3d.txt"
__C.TRAIN.BATCH_SIZE            = 8
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE            = 480
__C.TRAIN.DATA_AUG              = False
__C.TRAIN.LEARN_RATE_INIT       = 1e-3
__C.TRAIN.LEARN_RATE_END        = 1e-5
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FIRST_STAGE_EPOCHS    = 0
__C.TRAIN.SECOND_STAGE_EPOCHS   = 400
__C.TRAIN.INITIAL_WEIGHT        = "/media/wangning/Elements/tensorflow-yolov3/check_point_tl/weights_traffic_light/yolov3_test_loss=4.5359.ckpt-5"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "/media/jiangshengjie/Elements/tensorflow-yolov3/label_previewv2.txt"
__C.TEST.BATCH_SIZE             = 1
__C.TEST.INPUT_SIZE             = 1600
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = False
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection_train/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "/media/jiangshengjie/Elements/tl/yolov3_test_loss_v100_16g_1600=7.7765.ckpt-2"
__C.TEST.SCORE_THRESHOLD        = 0.15
__C.TEST.SHOW_LABEL             = True
__C.TEST.IOU_THRESHOLD          = 0.15






