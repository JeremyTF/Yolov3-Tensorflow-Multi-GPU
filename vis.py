import cv2 as cv
import numpy as np
from core.config import cfg

# failure_path = '/home/jiangshengjie/tensorflow-yolov3/tensorflow-yolov3/yizhuang/failurecase.txt'

# with open(failure_path,'r') as f:
#     txt = f.readlines()
#     annotations = [line.strip() for line in txt if len(line.strip().split(',')[1:]) != 0]
#     for line in annotations:
#         line = line.split(',')
#         path = str(line[0])
#         box= np.array([list(map(int, box.split())) for box in line[1:-1]])
#         bboxes_gt, classes_gt = box[:, :4], box[:, 4]
#         for i in range(1):
#             xmin, ymin, xmax, ymax = list(map(int, bboxes_gt[i]))
#             # box = line[1:-1]
#             # box1 = box[0:1:1]
#             img = cv.imread(path)
#             cv.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
#             img = cv.putText(img,str(classes_gt),(xmin,ymin),cv.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 3)
#             cv.imshow('img',img)
#             cv.waitKey(0)

import numpy as np
import core.utils as utils
import cv2 as cv
#obtain prediction bboxes
import colorsys

# failure_gt_path = '/home/jiangshengjie/tensorflow-yolov3/tensorflow-yolov3/yizhuang/faliure_groundtruth.txt'
def get_pred_dict(path):
    annotation_path = path
    pred_dict = {}
    pred_class_dict={}
    with open(annotation_path, 'r') as annotation_file:
        annotation_file = annotation_file.readlines()   
        annotation_file = [line.replace('./', '/media/wangning/Elements/tensorflow-yolov3/') for line
                   in annotation_file]
        annotation_file = [line.strip() for line in annotation_file if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotation_file)
        for num, line in enumerate(annotation_file):
            # for num, line in enumerate(annotation_file[250:400]):
            # line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3')
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            # image = cv2.imread(image_path)
            bbox_data_gt  = np.array([list(map(int, box.split(','))) for box in annotation[1:]])
            # print(annotation)

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :5], bbox_data_gt[:, 4]

            num_bbox_gt = len(bboxes_gt)


            pred_dict[image_path] = bboxes_gt
            pred_class_dict[image_path] = classes_gt
            # print(gt)
    return pred_dict


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

classes=read_class_names('/media/jiangshengjie/Elements/tensorflow-yolov3/trainv2.names')
num_classes = len(classes)
# image_h, image_w, _ = image.shape# class_ind = int(bbox[5])
hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]# bbox_mess = '%s: %.2f' % (classes[class_ind], score)
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
tl_path = '/media/jiangshengjie/Elements/tensorflow-yolov3/label_previewv2.txt'
tl_dict = get_pred_dict(tl_path)
# for k1,v1 in failure_dict.items():
for k2,v2 in tl_dict.items():
    # if k1 == k2:
    img = cv.imread(k2)
        # for box in v1:
        #     bboxes_gt, classes_gt = box[ :4], box[4]
        #     xmin, ymin, xmax, ymax = list(map(int, bboxes_gt))
        #     cv.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
        #     cv.putText(img,str(classes_gt),(xmax,ymax),cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    for box in v2:
        class_ind = int(box[4])
        bbox_color = colors[class_ind]
        bboxes_gt1, classes_gt1 = box[:4], box[4]
        class_ind = int(classes_gt1)
        bbox_mess = '%s:' % (classes[class_ind])
        xmin, ymin, xmax, ymax = list(map(int, bboxes_gt1))
        cv.rectangle(img, (xmin, ymin), (xmax, ymax), bbox_color, 2)
        t_size = cv.getTextSize(bbox_mess, cv.FONT_HERSHEY_COMPLEX, 0.7, thickness=2)[0]
        cv.rectangle(img, (xmin,ymin), (xmin + t_size[0], ymin - t_size[1]), bbox_color, -1)
        cv.putText(img, bbox_mess, (xmin, ymin), cv.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 1)
    cv.namedWindow('img',cv.WINDOW_NORMAL|cv.WINDOW_KEEPRATIO)
    cv.imshow('img',img)
    cv.waitKey(0)


