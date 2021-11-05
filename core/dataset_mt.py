#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-03-15 18:05:03
#   Description :
#
#================================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
import time
from concurrent.futures import ThreadPoolExecutor


class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            txt = [line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3') for line
                   in txt]
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            # annotations = annotations[0:int(len(annotations) * 0.8)]
        np.random.shuffle(annotations)
        return annotations


    def __iter__(self):
        return self

    def __next__(self):
        time1 = time.time()
        pool = ThreadPoolExecutor(max_workers=23)

        # start threads to load and preprocess
        thread0 = pool.submit(self.get_data, 0, 1)
        thread1 = pool.submit(self.get_data, 1, 0)
        thread2 = pool.submit(self.get_data, 2, 0)
        thread3 = pool.submit(self.get_data, 3, 0)
        thread4 = pool.submit(self.get_data, 4, 0)
        thread5 = pool.submit(self.get_data, 5, 0)
        thread6 = pool.submit(self.get_data, 6, 0)
        thread7 = pool.submit(self.get_data, 7, 0)
        thread8 = pool.submit(self.get_data, 8, 0)
        thread9 = pool.submit(self.get_data, 9, 0)
        thread10 = pool.submit(self.get_data, 10, 0)
        thread11 = pool.submit(self.get_data, 11, 0)
        thread12 = pool.submit(self.get_data, 12, 0)
        thread13 = pool.submit(self.get_data, 13, 0)
        thread14 = pool.submit(self.get_data, 14, 0)
        thread15 = pool.submit(self.get_data, 15, 0)
        # thread16 = pool.submit(self.get_data, 16, 0)
        # # thread17 = pool.submit(self.get_data, 17, 0)
        # # thread18 = pool.submit(self.get_data, 18, 0)
        # # thread19 = pool.submit(self.get_data, 19, 0)
        # # thread20 = pool.submit(self.get_data, 20, 0)
        # # thread21 = pool.submit(self.get_data, 21, 0)
        # # thread22 = pool.submit(self.get_data, 22, 0)
        # # thread23 = pool.submit(self.get_data, 23, 0)

        batch_image0, batch_label_sbbox0, batch_label_mbbox0, batch_label_lbbox0, \
        batch_sbboxes0, batch_mbboxes0, batch_lbboxes0 = thread0.result()

        batch_image1, batch_label_sbbox1, batch_label_mbbox1, batch_label_lbbox1, \
        batch_sbboxes1, batch_mbboxes1, batch_lbboxes1 = thread1.result()

        batch_image2, batch_label_sbbox2, batch_label_mbbox2, batch_label_lbbox2, \
        batch_sbboxes2, batch_mbboxes2, batch_lbboxes2 = thread2.result()

        batch_image3, batch_label_sbbox3, batch_label_mbbox3, batch_label_lbbox3, \
        batch_sbboxes3, batch_mbboxes3, batch_lbboxes3 = thread3.result()

        batch_image4, batch_label_sbbox4, batch_label_mbbox4, batch_label_lbbox4, \
        batch_sbboxes4, batch_mbboxes4, batch_lbboxes4 = thread4.result()

        batch_image5, batch_label_sbbox5, batch_label_mbbox5, batch_label_lbbox5, \
        batch_sbboxes5, batch_mbboxes5, batch_lbboxes5 = thread5.result()

        batch_image6, batch_label_sbbox6, batch_label_mbbox6, batch_label_lbbox6, \
        batch_sbboxes6, batch_mbboxes6, batch_lbboxes6 = thread6.result()

        batch_image7, batch_label_sbbox7, batch_label_mbbox7, batch_label_lbbox7, \
        batch_sbboxes7, batch_mbboxes7, batch_lbboxes7 = thread7.result()

        batch_image8, batch_label_sbbox8, batch_label_mbbox8, batch_label_lbbox8, \
        batch_sbboxes8, batch_mbboxes8, batch_lbboxes8 = thread8.result()

        batch_image9, batch_label_sbbox9, batch_label_mbbox9, batch_label_lbbox9, \
        batch_sbboxes9, batch_mbboxes9, batch_lbboxes9 = thread9.result()

        batch_image10, batch_label_sbbox10, batch_label_mbbox10, batch_label_lbbox10, \
        batch_sbboxes10, batch_mbboxes10, batch_lbboxes10 = thread10.result()

        batch_image11, batch_label_sbbox11, batch_label_mbbox11, batch_label_lbbox11, \
        batch_sbboxes11, batch_mbboxes11, batch_lbboxes11 = thread11.result()

        batch_image12, batch_label_sbbox12, batch_label_mbbox12, batch_label_lbbox12, \
        batch_sbboxes12, batch_mbboxes12, batch_lbboxes12 = thread12.result()

        batch_image13, batch_label_sbbox13, batch_label_mbbox13, batch_label_lbbox13, \
        batch_sbboxes13, batch_mbboxes13, batch_lbboxes13 = thread13.result()

        batch_image14, batch_label_sbbox14, batch_label_mbbox14, batch_label_lbbox14, \
        batch_sbboxes14, batch_mbboxes14, batch_lbboxes14 = thread14.result()

        batch_image15, batch_label_sbbox15, batch_label_mbbox15, batch_label_lbbox15, \
        batch_sbboxes15, batch_mbboxes15, batch_lbboxes15 = thread15.result()

        # batch_image16, batch_label_sbbox16, batch_label_mbbox16, batch_label_lbbox16, \
        # batch_sbboxes16, batch_mbboxes16, batch_lbboxes16 = thread16.result()
        #
        # batch_image17, batch_label_sbbox17, batch_label_mbbox17, batch_label_lbbox17, \
        # batch_sbboxes17, batch_mbboxes17, batch_lbboxes17 = thread17.result()
        #
        # batch_image18, batch_label_sbbox18, batch_label_mbbox18, batch_label_lbbox18, \
        # batch_sbboxes18, batch_mbboxes18, batch_lbboxes18 = thread18.result()
        #
        # batch_image19, batch_label_sbbox19, batch_label_mbbox19, batch_label_lbbox19, \
        # batch_sbboxes19, batch_mbboxes19, batch_lbboxes19 = thread19.result()
        #
        # batch_image20, batch_label_sbbox20, batch_label_mbbox20, batch_label_lbbox20, \
        # batch_sbboxes20, batch_mbboxes20, batch_lbboxes20 = thread20.result()
        #
        # batch_image21, batch_label_sbbox21, batch_label_mbbox21, batch_label_lbbox21, \
        # batch_sbboxes21, batch_mbboxes21, batch_lbboxes21 = thread21.result()
        #
        # batch_image22, batch_label_sbbox22, batch_label_mbbox22, batch_label_lbbox22, \
        # batch_sbboxes22, batch_mbboxes22, batch_lbboxes22 = thread22.result()
        #
        # batch_image23, batch_label_sbbox23, batch_label_mbbox23, batch_label_lbbox23, \
        # batch_sbboxes23, batch_mbboxes23, batch_lbboxes23 = thread23.result()

        time2 = time.time()
        load_data_time = time2 - time1
        print('load_data_time{}'.format(load_data_time))

        # start threads to add numpy array

        thread24 = pool.submit(self.add_numpy, batch_image0, batch_image1, batch_image2, batch_image3, batch_image4,
                               batch_image5, batch_image6,
                               batch_image7, batch_image8, batch_image9, batch_image10, batch_image11, batch_image12,
                               batch_image13, batch_image14,
                               batch_image15)

        thread25 = pool.submit(self.add_numpy, batch_label_sbbox0, batch_label_sbbox1, batch_label_sbbox2,
                               batch_label_sbbox3, batch_label_sbbox4, batch_label_sbbox5,
                               batch_label_sbbox6, batch_label_sbbox7, batch_label_sbbox8, batch_label_sbbox9,
                               batch_label_sbbox10, batch_label_sbbox11,
                               batch_label_sbbox12, batch_label_sbbox13, batch_label_sbbox14, batch_label_sbbox15,
                               )

        thread26 = pool.submit(self.add_numpy, batch_label_mbbox0, batch_label_mbbox1, batch_label_mbbox2,
                               batch_label_mbbox3, batch_label_mbbox4, batch_label_mbbox5,
                               batch_label_mbbox6, batch_label_mbbox7, batch_label_mbbox8, batch_label_mbbox9,
                               batch_label_mbbox10, batch_label_mbbox11,
                               batch_label_mbbox12, batch_label_mbbox13, batch_label_mbbox14, batch_label_mbbox15,
                               )

        thread27 = pool.submit(self.add_numpy, batch_label_lbbox0, batch_label_lbbox1, batch_label_lbbox2,
                               batch_label_lbbox3, batch_label_lbbox4, batch_label_lbbox5,
                               batch_label_lbbox6, batch_label_lbbox7, batch_label_lbbox8, batch_label_lbbox9,
                               batch_label_lbbox10, batch_label_lbbox11,
                               batch_label_lbbox12, batch_label_lbbox13, batch_label_lbbox14, batch_label_lbbox15,
                               )

        thread28 = pool.submit(self.add_numpy, batch_sbboxes0, batch_sbboxes1, batch_sbboxes2, batch_sbboxes3,
                               batch_sbboxes4, batch_sbboxes5,
                               batch_sbboxes6, batch_sbboxes7, batch_sbboxes8, batch_sbboxes9, batch_sbboxes10,
                               batch_sbboxes11,
                               batch_sbboxes12, batch_sbboxes13, batch_sbboxes14, batch_sbboxes15, )

        thread29 = pool.submit(self.add_numpy, batch_mbboxes0, batch_mbboxes1, batch_mbboxes2, batch_mbboxes3,
                               batch_mbboxes4, batch_mbboxes5,
                               batch_mbboxes6, batch_mbboxes7, batch_mbboxes8, batch_mbboxes9, batch_mbboxes10,
                               batch_mbboxes11,
                               batch_mbboxes12, batch_mbboxes13, batch_mbboxes14, batch_mbboxes15, )

        thread30 = pool.submit(self.add_numpy, batch_lbboxes0, batch_lbboxes1, batch_lbboxes2, batch_lbboxes3,
                               batch_lbboxes4, batch_lbboxes5,
                               batch_lbboxes6, batch_lbboxes7, batch_lbboxes8, batch_lbboxes9, batch_lbboxes10,
                               batch_lbboxes11,
                               batch_lbboxes12, batch_lbboxes13, batch_lbboxes14, batch_lbboxes15, )

        batch_image = thread24.result()

        batch_label_sbbox = thread25.result()

        batch_label_mbbox = thread26.result()

        batch_label_lbbox = thread27.result()

        batch_sbboxes = thread28.result()

        batch_mbboxes = thread29.result()

        batch_lbboxes = thread30.result()

        time3 = time.time()
        add_time = time3 - time2
        print('add_time{}'.format(add_time))
        pool.shutdown()
        return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes

    def add_numpy(self, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16):
        return a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16

    def get_data(self, num, batch_count):
        self.train_input_size = self.train_input_sizes
        self.train_output_sizes = self.train_input_size // self.strides

        batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

        batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                      self.anchor_per_scale, 5 + self.num_classes))
        batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                      self.anchor_per_scale, 5 + self.num_classes))
        batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                      self.anchor_per_scale, 5 + self.num_classes))

        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

        num = num
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:
                index = int(self.batch_count * self.batch_size + num)
                if index >= self.num_samples: index -= self.num_samples
                annotation = self.annotations[index]
                image, bboxes = self.parse_annotation(annotation)

                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                batch_image[num, :, :, :] = image
                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
                num += 16
            self.batch_count += batch_count

            return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                   batch_sbboxes, batch_mbboxes, batch_lbboxes
        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations)
            raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split(',')
        image_path = line[0]
	
        np.fromfile(image_path, dtype = np.uint8)
       # if not os.path.exists(image_path):
       #     raise KeyError("%s does not exist ... " %image_path)
        image = np.array(cv2.imread(image_path))
        # print image_path
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        bboxes = np.array([list(map(int, box.split())) for box in line[1:]])
        # print('stop')
       # if self.data_aug:
           # image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
           # image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
           # image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                       5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            if (0<bbox[0]<self.train_input_size) and (0<bbox[1]<self.train_input_size)and  (0<bbox[2]<self.train_input_size) and (0<bbox[3]<self.train_input_size):
                bbox_coor = bbox[:4]
                bbox_class_ind = 0
            # bbox_class_ind = bbox[4]

                onehot = np.zeros(self.num_classes, dtype=np.float)
                onehot[bbox_class_ind] = 1.0
                uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
                deta = 0.01
                smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

                bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)

                bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

                iou = []
                exist_positive = False
                for i in range(3):
                    anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                    anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                    anchors_xywh[:, 2:4] = self.anchors[i]

                    iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                    iou.append(iou_scale)
                    iou_mask = iou_scale > 0.3

                    if np.any(iou_mask):
                        xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                        label[i][yind, xind, iou_mask, :] = 0
                        label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                        label[i][yind, xind, iou_mask, 4:5] = 1.0
                        label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                        bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                        bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                        bbox_count[i] += 1

                        exist_positive = True
            else:
                break

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs




