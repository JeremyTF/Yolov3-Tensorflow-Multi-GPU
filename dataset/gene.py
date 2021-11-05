import numpy as np
import core.utils as utils
import cv2 as cv
#obtain prediction bboxes
prediciton_path = './failurecase.txt'
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

# print(a)

#obtain simulation bboxes
simulation_path = "./val_new_total.txt"
def get_simu_dict(path):
    simulation_path = path
    simu_dict = {}
    simu_class_dict = {}
    with open(simulation_path, 'r') as simulation_file:
        simulation_file = simulation_file.readlines()
        simulation_file = [line.replace('jiangshengjie/tensorflow-yolov3/tensorflow-yolov3' ,'wangning/Desktop/data' ) for
                           line in simulation_file]
        simulation_file = [line.replace('on','') for line in simulation_file]
        simulation_file = [line.replace('  ',' ') for line in simulation_file]
        for num, line in enumerate(simulation_file):
            # for num, line in enumerate(annotation_file[250:400]):
            # line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3')
            annotation = line.strip().split('.jpg,')
            image_path = annotation[0]+'.jpg'
            rest = annotation[1]
            image_name = image_path.split('/')[-1]
            # image = cv2.imread(image_path)
            # bbox_data_gt = np.array([list(map(int, box.split())) for box in annotation[2:] if len(box)>5])

            # bboxes_gt, classes_gt = bbox_data_gt[:, :5], bbox_data_gt[:, 4]


                # print('a')


            # num_bbox_gt = len(bboxes_gt)



            # simu_dict[image_path]= bboxes_gt
            # simu_class_dict[image_path]=classes_gt
            simu_class_dict[image_path] = rest
            # simu_dict.
            # simu_dict.
    return simu_class_dict
a = get_pred_dict(prediciton_path)
b = get_simu_dict(simulation_path)


path = './failure_with_mask.txt'

#
# print(a.keys())
# print(b.keys())
for k1,v1 in a.items():
    for k2,v2 in b.items():
        if k1 == k2:
            with open(path,'a') as f:
                mess_path = ''.join(k2)+','
                f.write(mess_path)
                rest = ''.join(v2)
                f.write(rest)
                f.write('\n')
