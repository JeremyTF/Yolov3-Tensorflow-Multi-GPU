import numpy as np

#tain prediction bboxes
prediciton_path = '/home/jiangshengjie/tensorflow-yolov3/tensorflow-yolov3/yizhuang/prediction_yolo.txt'
simulation_path = '/home/jiangshengjie/tensorflow-yolov3/tensorflow-yolov3/yizhuang/val_new_no1.txt'
def get_pred_dict(path):
    annotation_path = path
    pred_dict = {}
    with open(annotation_path, 'r') as annotation_file:
        annotation_file = annotation_file.readlines()
        annotation_file = [line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3') for
                           line in annotation_file]
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split(',')
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            # image = cv2.imread(image_path)
            bbox_data_gt = np.array([list(map(int, box.split())) for box in annotation[1:-1]])
            # print(annotation)

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]

            num_bbox_gt = len(bboxes_gt)

            for i in range(num_bbox_gt):
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
            pred_dict[image_path] = bboxes_gt
            # print(gt)
    return pred_dict

a = get_pred_dict(prediciton_path)
# print(a)

#obtain simulation bboxes


def get_simuclass_dict1(path):
    be_corrected_gt = 0
    to_add_gt = 0
    to_delete_gt = 0
    simulation_path = path
    simu_dict = {}
    simu_class_dict = {}
    simu_dict1={}
    bboxes_gt = []
    with open(simulation_path, 'r') as simulation_file:
        simulation_file = simulation_file.readlines()
        simulation_file = [line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3') for
                           line in simulation_file]
        simulation_file = [line.replace('on','') for line in simulation_file]
        simulation_file = [line.replace('  ',' ') for line in simulation_file]
        for num, line in enumerate(simulation_file):
            # for num, line in enumerate(annotation_file[250:400]):
            # line.replace('wangning/Desktop/data', 'jiangshengjie/tensorflow-yolov3/tensorflow-yolov3')
            annotation = line.strip().split(',')
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            # image = cv2.imread(image_path)
            bbox_data_gt = np.array([list(map(int, box.split())) for box in annotation[2:] if len(box)>5])
            # print(annotation)

            # if len(bbox_data_gt) == 0:
            # else:
            # for i in bbox_data_gt[:,4]:
            #     if i ==2 or i == 0 :
            #         bbox_tem=bbox_data_gt[i,:]
            #         bboxes_gt.append(bbox_tem)
            #     else:
            #         # print(i)
            #         continue
            # classes_gt = bbox_data_gt[:, 4]

            simu_dict1[image_path]= bbox_data_gt
            # simu_class_dict[image_path]=classes_gt
            # simu_dict.
            # simu_dict.
        # print(to_delete_gt)
        # print(to_add_gt)
        # print(to_delete_gt)
    return simu_dict1

b = get_simuclass_dict1(simulation_path)


def compute_iou(box1, box2,expand=True):
    try:
        width = abs(box1[0] - box1[2])
        height = abs(box1[1] - box1[3])
        if expand == True:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0]- 1 * width, box1[1]- 1 * height, box1[2]+ 1 * width, box1[3] + 1 * height
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0]- 1 * width, box2[1]- 1 * height, box2[2]+1 * width, box2[3]+ 1 * height
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    except:
        pass

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)



    width = max(inter_rect_x2 - inter_rect_x1 + 1,0)
    high = max(inter_rect_y2 - inter_rect_y1 + 1,0)
    inter_area = width * high

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


be_corrected = 0
to_add = 0
to_delete = 0

be_corrected_gt = 0
to_add_gt = 0
to_delete_gt = 0

# for k2,v2 in b.items():
#     for box in v2:
#         if box[4] == 0:
#             be_corrected_gt+=1
#         if box[4] == 1:
#             to_add_gt+=1
#         if box[4] == 2:
#             to_delete_gt+=1

pred_path = './yizhuang/hdmap_test_label.txt'
with open(pred_path,'w') as f:
    print('Comparing...')
    for k1,v1 in a.items():
        for k2,v2 in b.items():
                if k1==k2:
                    mess_path =''.join(k1) + ','
                    f.write(mess_path)

                    for box1 in v1:
                        for box2 in v2:
                            box2 = np.array(box2)
                            iou = compute_iou(box1,box2[0:4],expand=False)
                            if iou > 0.05:
                                xmin, ymin, xmax, ymax = list(map(str, box1))
                                mess_correct = ' '.join([xmin, ymin, xmax, ymax,str(0)]) + ','
                                f.write(mess_correct)
                                be_corrected+=1
                                # to_add += 1

                                break
                        else:
                            xmin, ymin, xmax, ymax = list(map(str, box1))
                            mess_add = ' '.join(([xmin, ymin, xmax, ymax,str(1)])) + ','
                            f.write(mess_add)
                            to_add +=1
                            # be_corrected += 1

                    for box2 in v2:
                        for box1 in v1:
                            box2 = np.array(box2)
                            iou = compute_iou(box1,box2[0:4],expand=False)
                            if iou > 0.05:
                                break
                        else:
                            xmin, ymin, xmax, ymax = list(map(str, box2[0:4]))
                            mess_delete = ' '.join(([xmin, ymin, xmax, ymax,str(2)])) + ','
                            f.write(mess_delete)
                            to_delete+=1
                    f.write('\n')
                    # c = compute_correct_add_delete(v1,v2)
                    # list1.append(c)
                    # continue


print(be_corrected)
print(to_add)
print(to_delete)
print(be_corrected_gt)
print(to_add_gt)
print(to_delete_gt)

# hdmap_path = '/home/jiangshengjie/tensorflow-yolov3/tensorflow-yolov3/yizhuang/hdmap_test_label.txt'
# with open(hdmap_path,'w') as f:
#     print('Comparing...')
#     for k1,v1 in a.items():
#         for k2,v2 in b.items():
#                 if k1==k2:
#                     mess_path =''.join(k1) + ','
#                     f.write(mess_path)
#                     for box in v2:
#                         xmin, ymin, xmax, ymax = list(map(str, box[0:4]))
#                         if box[4] == 2 or box[4] == 0:
#                             print(box)
#                             mess_correct = ' '.join([xmin, ymin, xmax, ymax, str(box[4])]) + ','
#                             f.write(mess_correct)
#                     f.write('\n')
