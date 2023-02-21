import os
import cv2
import sys
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
sys.path.append('../')
from utils.utils import cal_width

from utils import data_reader_ctw, dataset_text
import cfg_text

def heatmap_overlay(heatmap, image):
    # 灰度化heatmap
    heatmap_g = heatmap.astype(np.uint8)
    # 热力图伪彩色
    heatmap_color = cv2.applyColorMap(heatmap_g, cv2.COLORMAP_JET)
    # overlay热力图
    merge_img = image.copy()
    heatmap_img = heatmap_color.copy()
    overlay = image.copy()
    alpha = 0.5 # 设置覆盖图片的透明度
    #cv2.rectangle(overlay, (0, 0), (merge_img.shape[1], merge_img.shape[0]), (0, 0, 0), -1) # 设置蓝色为热度图基本色
    cv2.addWeighted(overlay, alpha, merge_img, 1-alpha, 0, merge_img) # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap_img, alpha, merge_img, 1-alpha, 0, merge_img) # 将热度图覆盖到原图

    return heatmap_color

def show_skelenton(img, kpts, color = (255,0,0)):

    kpts = np.array(kpts).reshape(-1, 2)
    
    skelenton = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]

    for sk in skelenton:

        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

        if pos1[0]>0 and pos1[1] >0 and pos2[0] >0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 2, 8)
            cv2.circle(img, pos1, 1, (0,255,0), 8)
            cv2.circle(img, pos2, 1, (0,255,0), 8)

    return img


if __name__ == "__main__":
    
    configs = cfg_text.Config()
    img_dir = 'data/ctw1500/train'
    data_source = data_reader_ctw.DataReader(img_dir, config=configs)

    print(len(data_source))

    datasets = dataset_text.Dataset(data_source, batch_size=1)

    n = configs.down_ratio

    print('##################')
    imgs, texts, hms, whs, reg_masks, inds, hm_hps, kpss, kps_masks, kpwidths = next(datasets)
    print(imgs.shape)
    print(hms.shape)
    print(whs.shape)
    print(reg_masks.shape)
    print(inds.shape)

    print(hm_hps.shape)
    print(texts.shape)

    text = texts[0]

    index = 0
    img_array = imgs[index]
    hm = hms[index]
    wh = whs[index]
    reg_mask = reg_masks[index]
    ind = inds[index]
    ind = ind.astype(np.int32)
    hmm = hm[..., 0]

    kpwidth = kpwidths[index]
    
    #print(hmm.shape)
    
    kps = kpss[index]

    hm_hp = hm_hps[index]
    hm_hp = np.sum(hm_hp, axis=-1)

    hmm = hm_hp
    hmm = np.clip(hmm, 0, 1)*255
    hmm = hmm.astype(np.uint8)

    h, w = hmm.shape[:2]
    ddd = np.zeros((h, w, 3), np.uint8)
    ddd[..., 0] = hmm
    ddd[..., 1] = hmm
    ddd[..., 2] = hmm
    '''
    for i in range(ind.shape[0]):
        #print(i, '--', ind[i])
        if ind[i] == 0: 
            break
        ch = ind[i] // w
        cw = ind[i] - ch*w
        #cv2.circle(ddd, (cw, ch), 5, (0,0,255))
        bw, bh, dx, dy, ww = wh[i]
        x1 = int(cw - dx)
        y1 = int(ch - dy)
        x2 = int(x1 + bw)
        y2 = int(y1 + bh)
        cv2.rectangle(ddd, (x1,y1), (x2,y2), (0,255,0), 1)
    '''
    cv2.imwrite('1-0.jpg', ddd)

    img = img_array + configs.MEAN_PIXEL
    img = img.astype(np.uint8)
    img = np.clip(img, 0, 255)
    cv2.imwrite('0.jpg', img[...,::-1])

    #merge = heatmap_overlay(hmm, img[...,::-1])
    #cv2.imwrite('1.jpg', merge)

    for i in range(ind.shape[0]):
        #print(i, '--', ind[i])
        if ind[i] == 0: 
            break
        ch = ind[i] // w
        cw = ind[i] - ch*w
        #cv2.circle(ddd, (cw, ch), 5, (0,0,255))

        bw, bh, dx, dy = wh[i]
        x1 = int(cw - dx)*n
        y1 = int(ch - dy)*n
        x2 = int(cw - dx + bw)*n
        y2 = int(ch - dy + bh)*n
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)

        # draw keypoints
        kp = kps[i]
        kp = kp.reshape(-1, 2)
        print('#############')
        #print(kp)
        #kp = (kp + np.array([cw, ch]))*2
        
        kp0 = kp[0]
        kp1 = kp[1]
        kp2 = kp[2]
        kp3 = kp[3] + np.array([cw, ch])
        kp4 = kp[4]
        kp5 = kp[5]
        kp6 = kp[6]
        kp2 = kp2 + kp3
        kp4 = kp4 + kp3
        kp1 = kp1 + kp2
        kp5 = kp5 + kp4
        kp0 = kp0 + kp1
        kp6 = kp6 + kp5
        kp = np.vstack([kp0, kp1, kp2, kp3, kp4, kp5, kp6]) * n
        #print(kp)
        print('@@@@@@@@@@@')
        #draw width
        width = kpwidth[i]*n
        #wy1 = int(kp[3][1] - width[3]*1./2)
        #wy2 = int(kp[3][1] + width[3]*1./2)
        #cv2.line(img, (int(kp[3][0]), wy1), (int(kp[3][0]), wy2), (0,0,255), 2, 8)

        print(width)
        #show_skelenton(img, kp)
        pplist = cal_width(kp.tolist(), width)
        pplist = np.asarray(pplist)
        cv2.drawContours(img, [pplist.astype(int)], -1, (0,255,0), 2)


    cv2.imwrite('2.jpg', img[...,::-1])

    #cv2.namedWindow("img", 0)
    #cv2.resizeWindow("img", 512, 512)
    #cv2.imshow('img', img[...,::-1])

    #cv2.namedWindow("test",1)
    #cv2.resizeWindow("test", 512, 512)
    #cv2.imshow('test', hmm)
    #cv2.namedWindow("text", 1)
    #cv2.resizeWindow("text", 512, 512)
    #cv2.imshow('text', text)
    cv2.imwrite('4.jpg', text[...,0]*255)
    #cv2.imshow('kernal_1', kernal_1)
    cv2.imwrite('5.jpg', text[...,1]*255)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

