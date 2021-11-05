import cv2 as cv
import numpy as np

failure_path = './failurecase.txt'
label_path = './hdmap_test_label.txt'

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

failure_gt_path = './faliure_groundtruth.txt'
def get_pred_dict(path):
    annotation_path = path
    pred_dict = {}
    pred_class_dict={}
    with open(annotation_path, 'r') as annotation_file:
        annotation_file = annotation_file.readlines()
        annotation_file = [line.replace('jiangshengjie/tensorflow-yolov3/tensorflow-yolov3','wangning/Desktop/data') for
                           line in annotation_file]
        for num, line in enumerate(annotation_file):
            # for num, line in enumerate(annotation_file[250:400]):
            # line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3')
            annotation = line.strip().split(',')
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            # image = cv2.imread(image_path)
            bbox_data_gt  = np.array([list(map(int, box.split())) for box in annotation[1:-1]])
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


failure_dict = get_pred_dict(label_path)
failure_gt_dict = get_pred_dict(failure_gt_path)

for k1,v1 in failure_dict.items():
    for k2,v2 in failure_gt_dict.items():
        if k1 == k2:
            img = cv.imread(k2)
            for box in v1:
                bboxes_gt, classes_gt = box[ :4], box[4]
                xmin, ymin, xmax, ymax = list(map(int, bboxes_gt))
                cv.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),1)
                cv.putText(img,str(classes_gt),(xmax,ymax),cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                for box in v2:
                    bboxes_gt1, classes_gt1 = box[:4], box[4]
                    xmin, ymin, xmax, ymax = list(map(int, bboxes_gt1))
                    cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
                    cv.putText(img, str(classes_gt1), (xmin, ymin), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            cv.namedWindow('yolov3', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.imshow('yolov3',img)
            cv.waitKey(0)


