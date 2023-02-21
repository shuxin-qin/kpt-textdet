import cv2
import numpy as np
import math
import os
from tps.ThinPlateSpline import ThinPlateSpline as stn

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]

def image_preporcess(image, target_size, means, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image - means

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_h = ih
    pad_w = iw
    
    pad_size = [[0,pad_h-nh], [0,pad_w-nw], [0,0]]
    img_pad = np.pad(image_resized, pad_size, 'constant')

    if gt_boxes is None:
        return img_pad

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
        return img_pad, gt_boxes


# [b,k,4+1+34+1]
def post_process_text(detections, org_img_shape, input_size, down_ratio, score_threshold):
    bboxes = detections[0, :, 0:4]
    scores = detections[0, :, 4]
    kps = detections[0, :, 5:19]
    kp_wid = detections[0, :, 19:-1]
    classes = detections[0, :, -1]
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size[1] / org_w, input_size[0] / org_h)

    bboxes = 1.0 * (bboxes * down_ratio) / resize_ratio

    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, org_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, org_h)

    kps = 1.0 * (kps * down_ratio) / resize_ratio

    kps[:, 0::2] = np.clip(kps[:, 0::2], 0, org_w)
    kps[:, 1::2] = np.clip(kps[:, 1::2], 0, org_h)

    kp_wid = 1.0 * (kp_wid * down_ratio) / resize_ratio
    
    score_mask = scores >= score_threshold
    bboxes, socres, kps, kp_wid, classes = bboxes[score_mask], scores[score_mask], kps[score_mask], kp_wid[score_mask], classes[score_mask]
    return np.concatenate([bboxes, socres[:, np.newaxis], kps, kp_wid, classes[:, np.newaxis]], axis=-1)


def text_draw_on_img(img, scores, bboxes, kps, kp_wids, thickness=2, drawkp=True):
    colors_tableau = [(158, 218, 229), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207)]
    
    h, w = img.shape[:2]
    thickness = int(max(h, w)*2 / 512 + 0.5)
    scale = 0.4
    text_thickness = 1
    line_type = 8
    polys = []
    img_list = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors_tableau[0]
        # Draw bounding boxes
        x1_src = int(bbox[0])
        y1_src = int(bbox[1])
        x2_src = int(bbox[2])
        y2_src = int(bbox[3])

        #cv2.rectangle(img, (x1_src, y1_src), (x2_src, y2_src), color, 1)
        # Draw text
        #s = '%.2f' % scores[i]
        # text_size is (width, height)
        #text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        #p1 = (y1_src - text_size[1], x1_src)

        #cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        #cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

        kpts = kps[i]
        kpts = np.array(kpts).reshape(-1, 2)
        #kpts = kpts + np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
        color = colors_tableau[2]
        
        kp_wid = kp_wids[i]
        pplist = cal_width(kpts, kp_wid)

        th = np.mean(kp_wid)
        tw = cal_text_width(kpts)

        ratio = tw / th

        if len(pplist) > 3:
            #print(pplist)
            pplist = np.asarray(pplist)

            #img_list.append(text_rectify(pplist, img, ratio))

            cv2.drawContours(img, [pplist.astype(int)], -1, (0,255,0), 2)

            polys.append(pplist)

        if drawkp:
            show_skelenton_text(img, kpts)

    return polys, img_list


def cal_text_width(kpts):
    dis = 0
    for i in range(6):
        p1 = kpts[i]
        p2 = kpts[i+1]

        dis += cal_dis(p1, p2)

    return dis

def cal_dis(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt(((x1-x2)**2)+((y1-y2)**2))


def cal_width(kpts, kp_wid):
    kp1, kp2, kp3, kp4, kp5, kp6, kp7 = kpts
    d1, d2, d3, d4, d5, d6, d7 = kp_wid
    #print(kp_wid)
    #print(kpts)
    p1u, p1d = cal_two_p(kp1, kp2, d1)
    p2u, p2d = cal_three_p(kp2, kp1, kp3, d2)
    p3u, p3d = cal_three_p(kp3, kp2, kp4, d3)
    p4u, p4d = cal_three_p(kp4, kp3, kp5, d4)
    p5u, p5d = cal_three_p(kp5, kp4, kp6, d5)
    p6u, p6d = cal_three_p(kp6, kp5, kp7, d6)
    p7u, p7d = cal_two_p1(kp6, kp7, d7)
    #print(p1u, p1d)
    #print(p2u, p2d)
    #print(p3u, p3d)
    #print(p4u, p4d)
    #print(p5u, p5d)
    #print(p6u, p6d)
    #print(p7u, p7d)
    pplist = [p1u, p2u, p3u, p4u, p5u, p6u, p7u, p7d, p6d, p5d, p4d, p3d, p2d, p1d]
    #print(pplist)
    return pplist

def cal_two_p(p1, p2, d): #计算过P1点与P1P2垂直的两个点，距离为d
    d = abs(d)
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        pp1 = [int(x1-d/2), y1]
        pp2 = [int(x1+d/2), y1]
        return pp1, pp2

    elif y1 == y2:
        pp1 = [x1, int(y1-d/2)]
        pp2 = [x1, int(y1+d/2)]
        return pp1, pp2

    else:
        dis = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        x0 = x1 - ((y1-y2)*d*0.5/dis)
        y0 = y1 - ((x2-x1)*d*0.5/dis)
        pp1 = [int(x0), int(y0)]
        pp2 = [int(2*x1-x0), int(2*y1-y0)]
        return pp1, pp2

def cal_two_p1(p1, p2, d): #计算过P2点与P1P2垂直的两个点，距离为d
    d = abs(d)
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        pp1 = [int(x2-d/2), y2]
        pp2 = [int(x2+d/2), y2]
        return pp1, pp2

    elif y1 == y2:
        pp1 = [x2, int(y2-d/2)]
        pp2 = [x2, int(y2+d/2)]
        return pp1, pp2

    else:
        dis = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        x0 = x2 - ((y1-y2)*d*0.5/dis)
        y0 = y2 - ((x2-x1)*d*0.5/dis)
        pp1 = [int(x0), int(y0)]
        pp2 = [int(2*x2-x0), int(2*y2-y0)]
        return pp1, pp2


def cal_three_p(p1, p2, p3, d): #计算过P1点与P2P3垂直的两个点，距离为d
    d = abs(d)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    if x2 == x3:
        pp1 = [int(x1-d/2), y1]
        pp2 = [int(x1+d/2), y1]
        return pp1, pp2

    elif y2 == y3:
        pp1 = [x1, int(y1-d/2)]
        pp2 = [x1, int(y1+d/2)]
        return pp1, pp2

    else:
        dis = math.sqrt((y3-y2)**2 + (x3-x2)**2)
        x0 = x1 - ((y2-y3)*d*0.5/dis)
        y0 = y1 - ((x3-x2)*d*0.5/dis)
        pp1 = [int(x0), int(y0)]
        pp2 = [int(2*x1-x0), int(2*y1-y0)]

        return pp1, pp2


def show_skelenton_text(img, kpts, color = (0,0,255)):

    for i in range(kpts.shape[0]):
        x,y = int(kpts[i][0]), int(kpts[i][1])

        cv2.circle(img, (x,y), 3, color, -1)

    skelenton = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]

    for sk in skelenton:
        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

        if pos1[0]>0 and pos1[1] >0 and pos2[0] >0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 1, 8)

    return img


def write_result_as_txt(txt_name, bboxes, path):
    if not os.path.exists(path):
        os.makedirs(path)

    filename = os.path.join(path, txt_name)
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox.reshape(-1)]
        # line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        line = "%d"%values[0]
        for v_id in range(1, len(values)):
            line += ",%d"%values[v_id]
        line += '\n'
        lines.append(line)

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)


def text_rectify(kpts, img, ratio):

    """
    使用cv2自带的tps处理
    """
    tps = cv2.createThinPlateSplineShapeTransformer()
    target = np.array([[[0,0], [50,0], [100,0], [150,0], [200,0], [250,0], [300,0], [300,50], [250,50], [200,50], [150,50], [100,50], [50,50], [0,50]]])
    source_cv2 = kpts.reshape(1, -1, 2)
    target_cv2 = target.reshape(1, -1, 2)

    #print(source_cv2)

    matches = list()
    for i in range(0, len(source_cv2[0])):
        matches.append(cv2.DMatch(i,i,0))

    tps.estimateTransformation(target_cv2, source_cv2, matches)
    new_img_cv2 = tps.warpImage(img.copy())

    new_img_cv2 = new_img_cv2[:50, :300, :]

    nw = int(ratio * 50 + 0.5)
    new_img_cv2 = cv2.resize(new_img_cv2, (nw, 50))

    return new_img_cv2