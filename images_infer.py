#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import os
import sys
import time

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

def main(model_path, images_dir, save_segment_dir):
    """
    main
    """

    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    pb_file         = model_path
    num_classes     = 7
    input_size      = 608
    graph           = tf.Graph()

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)

    images = sorted(os.listdir(images_dir))
    with tf.Session(graph=graph) as sess:
        idx = 0
        for imgfile in images:
            idx = idx + 1
            if idx < 5:
                continue
            # print imgfile
            start_time = time.time()

            original_image = cv2.imread(os.path.join(images_dir, imgfile))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]
            image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...]

            pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
                [return_tensors[1], return_tensors[2], return_tensors[3]],
                        feed_dict={ return_tensors[0]: image_data})

            pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
            bboxes = utils.nms(bboxes, 0.45, method='nms')
            image = utils.draw_bbox(original_image, bboxes)
            image = Image.fromarray(image)
            pngfile_path = os.path.join(save_segment_dir, imgfile)
            image.save(pngfile_path)

            end_time = time.time()
            print("process image use time %f" % (end_time - start_time))
        print('Done.')

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print ("The number of parameters is not 3, please check!")
    model_path = sys.argv[1]
    images_dir = sys.argv[2]
    save_segment_dir = sys.argv[3]
    main(model_path, images_dir, save_segment_dir)




