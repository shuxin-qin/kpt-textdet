#! /usr/bin/env python
# coding=utf-8

import os
import random
import numpy as np

class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset, batch_size=4, shuffle=True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_count = 0
        self.num_samples = len(self.dataset)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):

        imgs, gt_texts, hms, whs, reg_masks, inds, hm_hps, kpss, kps_masks, kpwidths \
             = [], [], [], [], [], [], [], [], [], []

        num = 0
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples: 
                    index -= self.num_samples
                
                img, gt_text, hm, wh, reg_mask, ind, hm_hp, kps, kps_mask, kpwidth\
                     = self.dataset[index]

                imgs.append(img)
                hms.append(hm)
                whs.append(wh)
                reg_masks.append(reg_mask)
                inds.append(ind)

                hm_hps.append(hm_hp)
                kpss.append(kps)
                kps_masks.append(kps_mask)
                kpwidths.append(kpwidth)
                gt_texts.append(gt_text)

                num += 1

            self.batch_count += 1
            imgs = np.asarray(imgs)
            hms = np.asarray(hms)
            whs = np.asarray(whs)
            reg_masks = np.asarray(reg_masks)
            inds = np.asarray(inds)

            hm_hps = np.asarray(hm_hps)
            kpss = np.asarray(kpss)
            kps_masks = np.asarray(kps_masks)
            kpwidths = np.asarray(kpwidths)
            gt_texts = np.asarray(gt_texts)

            return imgs, gt_texts, hms, whs, reg_masks, inds, hm_hps, kpss, kps_masks, kpwidths
        
        else:
            self.batch_count = 0
            if self.shuffle:
                self.dataset.shuffle()
            raise StopIteration

    def __len__(self):
        return self.num_batchs

    