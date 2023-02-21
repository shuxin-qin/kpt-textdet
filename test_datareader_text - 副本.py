import os
import cv2
import sys
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
sys.path.append('../')

from utils import data_reader_text, dataset_text
import cfg_text

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
    data_source = data_reader_text.DataReader(img_dir, config=configs)

    print(len(data_source))

    datasets = dataset_text.Dataset(data_source, batch_size=2)

    print('##################')
    imgs, hms, whs, regs, reg_masks, inds, hm_hps, hp_offsets, hp_inds, hp_masks, kpss, kps_masks, kpwidths, texts = next(datasets)
    print(imgs.shape)
    print(hms.shape)
    print(whs.shape)
    print(regs.shape)
    print(reg_masks.shape)
    print(inds.shape)

    print(hm_hps.shape)
    print(hp_offsets.shape)
    print(hp_inds.shape)
    print(hp_masks.shape)
    print(kpss.shape)
    print(kps_masks.shape)
    print(texts.shape)

    text = texts[0]

    index = 0
    img_array = imgs[index]
    hm = hms[index]
    #wh = whs[index]
    reg = regs[index]
    reg_mask = reg_masks[index]
    ind = inds[index]
    ind = ind.astype(np.int32)
    hmm = hm[..., 0]

    kpwidth = kpwidths[index]
    
    #print(hmm.shape)
    
    kps = kpss[index]

    hm_hp = hm_hps[index]
    hm_hp = np.sum(hm_hp, axis=-1)

    hmm = hmm + hm_hp
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
    cv2.imwrite('1.jpg', ddd)

    #img = draw_box_in_img.draw_boxes_with_label_and_scores_lp(img_array, box, label, np.ones_like(label), label_map)
    img = img_array + configs.MEAN_PIXEL
    img = img.astype(np.uint8)
    img = np.clip(img, 0, 255)
    for i in range(ind.shape[0]):
        #print(i, '--', ind[i])
        if ind[i] == 0: 
            break
        ch = ind[i] // w
        cw = ind[i] - ch*w
        #cv2.circle(ddd, (cw, ch), 5, (0,0,255))
        '''
        bw, bh, dx, dy, ww = wh[i]
        x1 = int(cw - dx)*4
        y1 = int(ch - dy)*4
        x2 = int(cw - dx + bw)*4
        y2 = int(ch - dy + bh)*4
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
        '''

        # draw keypoints
        kp = kps[i]
        kp = kp.reshape(-1, 2)
        kp = (kp + np.array([cw, ch]))
        kp = np.vstack((kp, np.array([cw, ch])))

        #draw width
        width = kpwidth[i]
        wy1 = int(kp[3][1] - width[3]*1./2)
        wy2 = int(kp[3][1] + width[3]*1./2)
        cv2.line(img, (int(kp[3][0]), wy1), (int(kp[3][0]), wy2), (0,0,255), 2, 8)

        show_skelenton(img, kp)


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
    cv2.imwrite('3.jpg', text*255)

    one = np.ones_like(text, dtype=np.uint8)
    zero = np.zeros_like(text, dtype=np.uint8)
    kernal_0 = np.where(text>0.55, one, zero)*255
    kernal_1 = np.where(text>0.01, one, zero)*255

    #cv2.imshow('kernal_0', kernal_0)
    cv2.imwrite('4.jpg', kernal_0)
    #cv2.imshow('kernal_1', kernal_1)
    cv2.imwrite('5.jpg', kernal_1)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

