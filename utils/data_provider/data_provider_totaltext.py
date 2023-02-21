# encoding:utf-8
import sys
import os
sys.path.append(os.path.abspath('../../')) 
import time
import traceback
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.utils_tool import logger
from utils.data_provider.data_util import GeneratorEnqueuer
import tensorflow as tf
import pyclipper
import Polygon as plg
import random

tf.app.flags.DEFINE_string('training_data_path', '/home/public/share/01Datasets/scene_text_detection/total_text/train',
                           'training dataset to use')
tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                            'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, then we resize'
                            'the image according to this')
tf.app.flags.DEFINE_integer('min_text_area_size', 10,
                            'if the text area size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')

FLAGS = tf.app.flags.FLAGS


def get_files(exts):
    files = []
    path = os.path.join(FLAGS.training_data_path, 'images')
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    #fn, _ = os.path.splitext(filename)
                    files.append(filename)
                    break
    return files

def load_annoataion(img, txt_fn):
    '''
    load annotation from the text file
    :param:
    :return:
    '''
    text_polys = []
    tags = []
    h, w = img.shape[0:2]
    with open(txt_fn, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            #print(im_name)
            split_line = line.strip().split(',')
            #print(split_line)

            xline = split_line[0].strip().split('[')[-1].split(']')[0].split(' ')
            yline = split_line[1].strip().split('[')[-1].split(']')[0].split(' ')

            xs = []
            ys = []
            for i in range(len(xline)):
                if xline[i] != '':
                    xs.append(int(xline[i]))
            for i in range(len(yline)):
                if yline[i] != '':
                    ys.append(int(yline[i]))

            bbox = [[x*1.0/w, y*1.0/h] for x, y in zip(xs, ys)]
            poly = np.array(bbox).reshape(-1)
            text_polys.append(poly)
            tags.append(True)

        return text_polys, tags

def check_and_validate_polys(polys, im_shape):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :return:
    '''
    (h, w) = im_shape
    if polys.shape[0] == 0:
        return []
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    for poly in polys:
        if abs(pyclipper.Area(poly)) < 1:
            continue
        #clockwise
        if pyclipper.Orientation(poly):
            poly = poly[::-1]

        validated_polys.append(poly)
    return np.array(validated_polys)


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 30
    angle = random.random() * 2 * max_angle - max_angle
    random_angles = np.array([angle, angle, angle, 90, 180, 270])
    angle = np.random.choice(random_angles)
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_scale(img, min_size, random_scale):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    #random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs
    
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis = 1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis = 1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=1000):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue

        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)


def generator(input_size=512, 
              batch_size=8,
              rand_scale=np.array([0.5, 1, 2.0, 3.0]),
              scale_ratio=np.array([0.33, 0.43, 0.52, 0.61, 0.70, 0.78, 0.86, 0.94])):

    image_list = np.array(get_files(['jpg', 'png', 'jpeg', 'JPG']))

    logger.info('{} training images in {}'.format(
        image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])

    while True:
        np.random.shuffle(index)
        images = []
        #image_fns = []
        seg_maps = []
        training_masks = []
        for i in index:
            try:
                #im_fn = image_list[i]
                im_fn = os.path.join(FLAGS.training_data_path, 'images', image_list[i])
                im = cv2.imread(im_fn)
                if im is None:
                    logger.info(im_fn)
                h, w, _ = im.shape
                #txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
                txt_fn = os.path.join(FLAGS.training_data_path, 'labels', 'poly_gt_'+image_list[i].split('.')[0]+'.txt')
                if not os.path.exists(txt_fn):
                    continue

                text_polys, tags = load_annoataion(im, txt_fn)
                if len(text_polys) == 0:
                    continue
                #text_polys = check_and_validate_polys(text_polys, (h, w))

                im = random_scale(im, input_size, rand_scale)
                
                gt_text = np.zeros(im.shape[0:2], dtype='uint8')
                training_mask = np.ones(im.shape[0:2], dtype='uint8')
                text_polys_new = []
                if len(text_polys) > 0:
                    for i in range(len(text_polys)):
                        text_poly = text_polys[i]
                        text_poly = np.reshape(text_poly*([im.shape[1], im.shape[0]]*int(text_poly.shape[0]/2)), (int(text_poly.shape[0]/2), 2)).astype('int32')
                        text_polys_new.append(text_poly)
                        cv2.drawContours(gt_text, [text_poly], -1, i + 1, -1)
                        if not tags[i]:
                            cv2.drawContours(training_mask, [text_poly], -1, 0, -1)
                '''
                gt_kernals = []
                kernel_num = scale_ratio.shape[0]
                for i in range(0, kernel_num-1):
                    rate = scale_ratio[i]
                    gt_kernal = np.zeros(im.shape[0:2], dtype='uint8')
                    kernal_bboxes = shrink(text_polys_new, rate)
                    for i in range(len(text_polys_new)):
                        cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
                    gt_kernals.append(gt_kernal)
                '''
                gt_text[gt_text > 0] = 1
                gt_kernals = []
                border_dist = gt_text.astype('float32')
                #center_dist = np.zeros(im.shape[0:2], dtype='float32')
                kernel_num = scale_ratio.shape[0]
                lamda = 1.0/kernel_num
                for i in range(0, kernel_num):
                    #rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
                    rate = scale_ratio[i]
                    gt_kernal = np.zeros(im.shape[0:2], dtype='uint8')
                    kernal_bboxes = shrink(text_polys_new, rate)
                    for j in range(len(text_polys_new)):
                        cv2.drawContours(gt_kernal, [kernal_bboxes[j]], -1, 1, -1)
                    if i > 1:
                        gt_kernals.append(gt_kernal)
                    border_dist = border_dist - lamda*gt_kernal

                center_dist = (1 - border_dist) * gt_text
                center_dist = center_dist.astype('float32')

                imgs = [im, gt_text, training_mask, border_dist, center_dist]
                imgs.extend(gt_kernals)

                imgs = random_horizontal_flip(imgs)
                imgs = random_rotate(imgs)
                imgs = random_crop(imgs, (input_size, input_size))

                img, gt_text, training_mask, border_dist, center_dist, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4], imgs[5:]

                gt_center = gt_kernals[0]
                gt_kernals = []
                gt_kernals.append(gt_center)
                gt_kernals.append(gt_text)
                gt_kernals.append(border_dist)
                gt_kernals.append(center_dist)
                gt_kernals = np.stack(gt_kernals, axis=-1)

                images.append(img[:, :, ::-1].astype(np.float32))
                #image_fns.append(im_fn)
                seg_maps.append(gt_kernals.astype(np.float32))
                training_masks.append(training_mask[:, :, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, seg_maps, training_masks
                    images = []
                    seg_maps = []
                    training_masks = []
                    #border_dists = []
            except Exception as e:
                traceback.print_exc()
                continue

def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=16, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.001)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

if __name__ == '__main__':

    gen = get_batch(num_workers=1)

    images, seg_maps, training_masks = next(gen)

    for i in range(1):
        image = images[i]
        seg_map = seg_maps[i]
        training_mask = training_masks[i]

        print('image: ', image.shape)
        print('seg_map: ', seg_map.shape)
        print('training_mask: ', training_mask.shape)

        cv2.namedWindow("test",0)
        cv2.resizeWindow("test", 512, 512)
        cv2.imshow('test', image[:, :, ::-1].astype('uint8'))
        
        cv2.namedWindow("test0",0)
        cv2.resizeWindow("test0", 512, 512)
        cv2.imshow('test0', seg_map[:, :, 0].astype(np.float32))

        cv2.namedWindow("test1",0)
        cv2.resizeWindow("test1", 512, 512)
        cv2.imshow('test1', seg_map[:, :, 1].astype(np.float32))

        cv2.namedWindow("test2",0)
        cv2.resizeWindow("test2", 512, 512)
        cv2.imshow('test2', seg_map[:, :, -2].astype(np.float32))

        cv2.namedWindow("test3",0)
        cv2.resizeWindow("test3", 512, 512)
        cv2.imshow('test3', seg_map[:, :, -1].astype(np.float32))
        
        cv2.waitKey(0)

    cv2.destroyAllWindows()

