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

unknown_list = [0,1,2,3]
round_list = [4,5,6,7,8]
Left_list = [9,10,11]
Right_list = [12,13,14]
Up_list = [15,16,17]
Down_list = [18]
X_list = [19]
TA_list = [20,21]
figure_list = [23,24,25,26,27]
LUdouble_list = [28,29,]
line_list = [30]
square_list = [31,32,33]
bike_list = [34,35,36]
pedestrain_list = [37,38]

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

    unknown = 0
    unknown_count = 0

    round = 0
    round_count = 0

    left = 0
    left_count = 0

    right = 0
    right_count = 0

    up = 0
    up_count = 0

    down = 0
    down_count = 0

    x = 0
    x_count = 0

    ta = 0
    ta_count = 0

    figure = 0
    figure_count = 0

    LUdouble = 0
    LUdouble_count = 0

    line = 0
    line_count = 0

    square = 0
    square_count = 0

    bike = 0
    bike_count = 0

    pedestrain = 0
    pedestrain_count = 0



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

                    if gt_class in unknown_list:
                        unknown += 1
                    if gt_class in round_list:
                        round += 1
                    if gt_class in Left_list:
                        left += 1
                    if gt_class in Right_list:
                        right +=1
                    if gt_class in Up_list:
                        up += 1
                    if gt_class in Down_list:
                        down += 1
                    if gt_class in X_list:
                        x += 1
                    if gt_class in TA_list:
                        ta += 1
                    if gt_class in figure_list:
                        figure +=1
                    if gt_class in LUdouble_list:
                        LUdouble += 1
                    if gt_class in line_list:
                        line += 1
                    if gt_class in square_list:
                        square += 1
                    if gt_class in bike_list:
                        bike += 1
                    if gt_class in pedestrain_list:
                        pedestrain += 1


                    for pred_box_dict in prediction:
                        pred_class = pred_box_dict['class']
                        # if True:
                        if gt_class == pred_class:
                            area1 = compute_area(gt_box_dict['box'])
                            area2 = compute_area(pred_box_dict['box'])
                            if area1 >= 0 and area2 >= 800:
                                iou = compute_iou(gt_box_dict['box'],pred_box_dict['box'],expand=True)
                                if iou >=0.3:

                                    if gt_class in unknown_list:
                                        unknown_count+=1
                                        correct_count += 1
                                        break
                                    if gt_class in round_list:
                                        round_count += 1
                                        correct_count += 1
                                        break
                                    if gt_class in Left_list:
                                        left_count+=1
                                        correct_count += 1
                                        break
                                    if gt_class in Right_list:
                                        right_count += 1
                                        correct_count += 1
                                    if gt_class in Up_list:
                                        up_count += 1
                                        correct_count += 1
                                        break

                                    if gt_class in Down_list:
                                        down_count += 1
                                        correct_count += 1
                                        break
                                    if gt_class in X_list:
                                        x_count+=1
                                        correct_count += 1
                                        break
                                    if gt_class in TA_list:
                                        ta_count += 1
                                        correct_count += 1
                                    if gt_class in figure_list:
                                        figure_count += 1
                                        correct_count += 1
                                        break

                                    if gt_class in LUdouble_list:
                                        LUdouble_count += 1
                                        correct_count += 1
                                        break
                                    if gt_class in line_list:
                                        line_count+=1
                                        correct_count += 1
                                        break
                                    if gt_class in square_list:
                                        square_count += 1
                                        correct_count += 1
                                    if gt_class in bike_list:
                                        bike_count += 1
                                        correct_count += 1
                                        break
                                    if gt_class in pedestrain_list:
                                        pedestrain_count += 1
                                        correct_count += 1
                                        break
                            else:
                                continue

        except:
            continue


    recall            = correct_count/gt_count

    recall_unknown =  unknown_count/unknown
    recall_round     = round_count / round
    recall_left     = left_count/left
    recall_right = right_count/right
    # recall_up = up_count / up
    # recall_down = down_count / down
    # recall_x = x_count / x
    # recall_ta = ta_count / ta
    recall_figure = figure_count / figure
    # recall_LUdouble = LUdouble_count / LUdouble
    # recall_line = line_count / line
    # recall_square = square_count / square
    # recall_bike = bike_count / bike
    # recall_pedestrain = pedestrain_count / pedestrain

    



    total_num = 0
    pred_unknown_num = 0
    pred_round_num = 0
    pred_left_num = 0
    pred_right_num = 0
    pred_up_num = 0
    pred_down_num = 0
    pred_x_num = 0
    pred_ta_num = 0
    pred_figure_num = 0
    pred_LUdouble_num = 0
    pred_line_num = 0
    pred_square_num = 0
    pred_bike_num = 0
    pred_pedestrain_num = 0

    for pred_key in metric_info_dict.keys():
        pred_info = metric_info_dict[pred_key]['pred']
        for bbox in pred_info:
            area2 = compute_area(bbox['box'])
            if area2 >= 800:
                box_class = bbox['class']
                if box_class in unknown_list:
                    pred_unknown_num += 1
                if box_class in round_list:
                    pred_round_num += 1
                if box_class in Left_list:
                    pred_left_num += 1
                if box_class in Right_list:
                    pred_right_num += 1
                if box_class in Up_list:
                    pred_up_num += 1
                if box_class in Down_list:
                    pred_down_num += 1
                if box_class in X_list:
                    pred_x_num += 1
                if box_class in TA_list:
                    pred_ta_num += 1
                if box_class in figure_list:
                    pred_figure_num += 1
                if box_class in LUdouble_list:
                    pred_LUdouble_num += 1
                if box_class in line_list:
                    pred_line_num += 1
                if box_class in square_list:
                    pred_square_num += 1
                if box_class in bike_list:
                    pred_bike_num += 1
                if box_class in pedestrain_list:
                    pred_pedestrain_num += 1

    total_num = pred_unknown_num + pred_round_num + pred_left_num+ pred_right_num + pred_up_num + pred_down_num + pred_x_num +pred_ta_num\
    +pred_figure_num + pred_LUdouble_num + pred_line_num + pred_square_num +pred_bike_num + pred_pedestrain_num

    precision             = correct_count/total_num
    precision_unknown  =  unknown_count/pred_unknown_num
    precision_round      = round_count / pred_round_num
    precision_left      = left_count/pred_left_num
    # precision_right = right_count/pred_right_num

    # precision_up = up_count / pred_up_num
    # precision_down = down_count / pred_down_num
    precision_x = x_count / pred_x_num
    # precision_ta = ta_count / pred_ta_num

    precision_figure = figure_count / pred_figure_num
    # precision_LUdouble = LUdouble_count / pred_LUdouble_num
    # precision_line = line_count / pred_line_num
    # precision_square = square_count / pred_square_num

    # precision_bike = bike_count / pred_bike_num
    # precision_pedestrain = pedestrain_count / pred_pedestrain_num



    # print(precision_red,precision_yellow,precision_green)
    print('total_recall    :{:.2f}%({}/{}), recall_unknown    :{:.2f}%({}/{}), recall_round    :{:.2f}%({}/{}),   recall_left   :{:.2f}%({}/{}),    recall_right   :{:.2f}%({}/{})'.format(recall * 100, correct_count, gt_count, recall_unknown * 100, unknown_count, unknown,
                                                                                                                                                                                                                              recall_round * 100, round_count, round, recall_left * 100, left_count, left,
                                                                                                                                                                                                                              recall_right * 100, right_count, right, ))
    print(
        'recall_up:0({}/{}),  recall_down:0({}/{}),recall_x:0({}/{}),recall_ta:0({}/{}) , recall_figure    :{:.2f}%({}/{}), recall_LUdouble:0({}/{}),recall_line    :0({}/{}),'
        'recall_square:0({}/{}),recall_bike:0({}/{}),recall_pedestrain:0({}/{})'.format(
            up_count,up,
            down_count,down,
            x_count,x,ta_count,ta,
            recall_figure * 100, figure_count,figure,
            LUdouble_count,LUdouble,
            line_count,line,
            square_count,square,
            bike_count,bike,
            pedestrain_count,pedestrain ))

    print("----------------------------------------------------------------------------------------------------------------------------------------------------------")

    print('total_precision :{:.2f}%({}/{}), precision_unknown :{:.2f}%({}/{}), precision_round :{:.2f}%({}/{}),  '
          'precision_left :{:.2f}%({}/{}), precision_right :0({}/{})'.format(precision * 100, correct_count, total_num, precision_unknown * 100,
                                                                             unknown_count, pred_unknown_num,precision_round * 100, round_count, pred_round_num, precision_left * 100, left_count, pred_left_num,right_count, pred_right_num,  ))

    print(
         'precision_up:0({}/{}), precision_down:0({}/{}),precision_x:{:.2f}%({}/{}),precision_ta:0({}/{}) , precision_figure    :{:.2f}%({}/{}), '
         'precision_LUdouble:0({}/{}),precision_line :0%({}/{}),'
        'precision_square:0({}/{}),precision_bike:0({}/{}),precision_pedestrain:0({}/{})'.format(
            up_count,pred_up_num,
            down_count,pred_down_num,
            precision_x*100,x_count,pred_x_num,
            ta_count,pred_ta_num,
            precision_figure * 100, figure_count,pred_figure_num,
            LUdouble_count,pred_LUdouble_num,
            line_count,pred_line_num,
            square_count,pred_square_num,
            bike_count,pred_bike_num,
            pedestrain_count,pred_pedestrain_num ))

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
