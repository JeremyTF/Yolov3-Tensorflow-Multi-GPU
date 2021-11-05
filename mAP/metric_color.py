import glob
import os

# class_dict = {'be_corrected':0,
#               'to_add':1,
#               'to_del':2}
class_dict = {'unknown_red':0,
'unknown_yellow':1,
'unknown_green':2,
'unknown_black':3,
'round_unknown':4,
'round_red':5,
'round_yellow':6,
'round_green':7,
'round_black':8,
'Left_red':9,
'Left_yellow':10,
'Left_green':11,
'Right_red':12,
'Right_yellow':13,
'Right_green':14,
'Up_red':15,
'Up_yellow':16,
'Up_green':17,
'Down_green':18,
'X_red':19,
'TA_red':20,
'TA_yellow':21,
'TA_green':22,
'figure_unknown':23,
'figure_red':24,
'figure_yellow':25,
'figure_green':26,
'figure_black':27,
'LUdouble_red':28,
'LUdouble_green':29,
'line_red':30,
'square_red':31,
'square_yellow':32,
'square_green':33,
'bike_red':34,
'bike_yellow':35,
'bike_green':36,
'pedestrain_red':37,
'pedestrain_green':38}

def read_data(imageid_elem, file_path):
    """
    """
    key = file_path.split('/')[-1]

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            elem = line.split(' ')
            if key not in imageid_elem.keys():
                imageid_elem[key] = set([])
            imageid_elem[key].add(line)

    return 0

def obtain_label(file_dir, file_suffix):
    """
    """
    imageid_elem = {}
    target_file = os.path.join(file_dir, file_suffix)
    file_names = glob.glob(target_file)
    for per_file in file_names:
        read_data(imageid_elem, per_file)

    return imageid_elem

def str_parse_gt(gt_str_list):
    bbox_info_list = []
    for box_str in gt_str_list:
        box_elem_str_list = box_str.split(' ')
        box_elem_str_list = [elem for elem in box_elem_str_list if len(elem)>1]
        class_str = box_elem_str_list[0]
        bbox = [int(elem) for elem in box_elem_str_list[1:]]
        class_num = class_dict[class_str]
        tmp_box_info={'class':class_num,'box':bbox}
        bbox_info_list.append(tmp_box_info)

    return bbox_info_list

def str_parse_pred(pred_str_list):
    bbox_info_list = []
    for box_str in pred_str_list:
        box_elem_str_list = box_str.split(' ')
        box_elem_str_list = [elem for elem in box_elem_str_list if len(elem) > 1]
        class_str = box_elem_str_list[0]
        bbox = [int(elem) for elem in box_elem_str_list[2:]]
        if len(bbox)<4:
            return None
        class_num = class_dict[class_str]
        tmp_box_info = {'class': class_num, 'box': bbox}
        bbox_info_list.append(tmp_box_info)

    return bbox_info_list

def get_info_dict(gt_info, pre_info):
    metric_info_dict = {}
    for key in gt_info.keys():
        ground_truth = None
        pred = None
        if key in gt_info.keys():
            ground_truth = str_parse_gt(gt_info[key])


        if key  in pre_info.keys():
            pred = str_parse_pred(pre_info[key])

        if (ground_truth is not None) and (pred is not None):
            tmp_dict = {key:{'gt':ground_truth,
                             'pred':pred}}
            metric_info_dict.update(tmp_dict)
    return metric_info_dict


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

def compute_area(box1):
    width = abs(box1[0] - box1[2])
    height = abs(box1[1] - box1[3])
    area = width*height
    return area


def compute_recall_precision(metric_info_dict):
    """
    metric_info_dict:{'name.txt':{'gt':[],'pred':[]}}
    """

    gt_count = 0
    correct_count= 0

    red = 0
    red_count = 0

    yellow = 0
    yellow_count = 0

    green = 0
    green_count = 0

    black = 0
    black_count = 0

    unknown = 0
    unknown_count = 0

    for key in metric_info_dict.keys():
        try:
            pair_info = metric_info_dict[key]
            groundtruth = pair_info['gt']
            prediction = pair_info['pred']
            for gt_box_dict in groundtruth:
                area1 = compute_area(gt_box_dict['box'])
                if area1 >= 800:
                    gt_count+=1
                    gt_class = gt_box_dict['class']



                    if gt_class in [0 , 5 , 9 , 12 , 15 , 19 , 20 ,24 , 28 , 30 , 31 , 34 , 37]:
                        red += 1
                    if gt_class in [1 , 6 , 10 , 13 , 16 , 21 , 25 , 32 , 35]:
                        yellow += 1
                    if gt_class in [2, 7, 11, 14, 17, 18, 22, 26, 29, 33, 36, 38]:
                        green += 1
                    if gt_class in [3 , 8 , 27]:
                        black +=1
                    if gt_class in [4 , 23]:
                        unknown += 1

                    for pred_box_dict in prediction:
                        pred_class = pred_box_dict['class']
                        # if True:
                        if gt_class == pred_class:
                            area1 = compute_area(gt_box_dict['box'])
                            area2 = compute_area(pred_box_dict['box'])
                            if area1 >=0 and area2 >=800:
                                iou = compute_iou(gt_box_dict['box'],pred_box_dict['box'],expand=True)
                                if iou >=0.3:

                                    if gt_class in [0 , 5 , 9 , 12 , 15 , 19 , 20 ,24 , 28 , 30 , 31 , 34 , 37]:
                                        red_count+=1
                                        correct_count += 1
                                        break
                                    if gt_class in [1 , 6 , 10 , 13 , 16 , 21 , 25 , 32 , 35]:
                                        yellow_count += 1
                                        correct_count += 1
                                        break
                                    if gt_class in [2, 7, 11, 14, 17, 18, 22, 26, 29, 33, 36, 38]:
                                        green_count+=1
                                        correct_count += 1
                                        break
                                    if gt_class in [3 , 8 , 27]:
                                        black_count += 1
                                        correct_count += 1
                                    if gt_class in [4 , 23]:
                                        unknown_count += 1
                                        correct_count += 1
                                        break
                            else:
                                continue
        except:
            continue


    recall            = correct_count/gt_count
    recall_red =  red_count/red
    recall_yellow     = yellow_count / yellow
    recall_green     = green_count/green
    recall_black = black_count/black
    recall_unknown = unknown_count/unknown
    



    total_num = 0
    pred_red_num = 0
    pred_yellow_num = 0
    pred_green_num = 0
    pred_black_num = 0
    pred_unknown_num = 0
    for pred_key in metric_info_dict.keys():
        pred_info = metric_info_dict[pred_key]['pred']
        for bbox in pred_info:
            area2 = compute_area(bbox['box'])
            if area2 >= 800:
                box_class = bbox['class']
                if box_class in [0 , 5 , 9 , 12 , 15 , 19 , 20 ,24 , 28 , 30 , 31 , 34 , 37]:
                    pred_red_num += 1
                if box_class in [1 , 6 , 10 , 13 , 16 , 21 , 25 , 32 , 35]:
                    pred_yellow_num += 1
                if box_class in [2, 7, 11, 14, 17, 18, 22, 26, 29, 33, 36, 38]:
                    pred_green_num += 1
                if box_class in [3 , 8 , 27]:
                    pred_black_num += 1
                if box_class in [4 , 23]:
                    pred_unknown_num += 1

    total_num = pred_red_num + pred_yellow_num + pred_green_num+ pred_black_num + pred_unknown_num

    precision             = correct_count/total_num
    precision_red  =  red_count/pred_red_num
    precision_yellow      = yellow_count / pred_yellow_num
    precision_green      = green_count/pred_green_num
    precision_black = black_count/pred_black_num
    # precision_unknown = unknown_count/pred_unknown_num

    # print(precision_red,precision_yellow,precision_green)
    print('total_recall    :{:.2f}%({}/{}), recall_red    :{:.2f}%({}/{}), recall_yellow    :{:.2f}%({}/{}),   recall_green   :{:.2f}%({}/{}),    recall_black   :{:.2f}%({}/{})'.format(recall * 100, correct_count, gt_count, recall_red * 100, red_count, red,
                                                                                                                                                                                                                              recall_yellow * 100, yellow_count, yellow, recall_green * 100, green_count, green,
                                                                                                                                                                                                                              recall_black * 100, black_count, black, ))
    print('total_precision :{:.2f}%({}/{}), precision_red :{:.2f}%({}/{}), precision_yellow :{:.2f}%({}/{}),  precision_green :{:.2f}%({}/{}), precision_black :{:.2f}%({}/{})'.format(precision * 100, correct_count, total_num, precision_red * 100, red_count, pred_red_num,
                                                                                                                                                                                                                           precision_yellow * 100, yellow_count, pred_yellow_num, precision_green * 100, green_count, pred_green_num,
                                                                                                                                                                                                                           precision_black * 100, black_count, pred_black_num,  ))
    # print(
    #     'total_recall    :{:.2f}%({}/{}), recall_be_correct    :{:.2f}%({}/{})'.format(
    #         recall * 100, correct_count, gt_count, recall_red * 100, red_count, red,
    #         ))
    # print(
    #     'total_precision :{:.2f}%({}/{}), precision_be_correct :{:.2f}%({}/{})'.format(
    #         precision * 100, correct_count, total_num, precision_be_correct * 100, red_count,
    #         pred_red_num, ))


def main():
    gt_info = obtain_label('./ground-truth', '*.txt')
    pre_info = obtain_label('./predicted', '*.txt')
    metric_info_dict = get_info_dict(gt_info,pre_info)
    precision= compute_recall_precision(metric_info_dict)
    # recall= compute_recal(metric_info_dict)



if __name__ == '__main__':
    main()
