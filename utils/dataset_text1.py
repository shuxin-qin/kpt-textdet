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

        imgs, gt_texts  = [], []

        num = 0
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples: 
                    index -= self.num_samples
                
                img, gt_text = self.dataset[index]

                imgs.append(img)
                gt_texts.append(gt_text)

                num += 1

            self.batch_count += 1
            imgs = np.asarray(imgs)
            gt_texts = np.asarray(gt_texts)

            return imgs, gt_texts
        
        else:
            self.batch_count = 0
            if self.shuffle:
                self.dataset.shuffle()
            raise StopIteration

    def __len__(self):
        return self.num_batchs

    