import os
import cv2
import numpy as np
import io
import random, math
import xml.etree.ElementTree as ET
from utils.data_aug_text import random_crop, random_translate, random_scale, scale_wh, random_rotate

class DataReader(object):
    def __init__(self, img_dir, config=None):
        '''Load a subset of the COCO dataset.
        '''
        self.config = config
        self.img_dir = img_dir
        self.max_objs = 20
        self.num_classes = 1
        self.num_joints = 7

        self.images = self.get_img_list(os.path.join(self.img_dir, 'images'))
        self.num_samples = len(self.images)
        self.shuffle()

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        
        img_id = self.images[idx]
        txt_fn = os.path.join(self.img_dir, 'labels', img_id.replace('.jpg','.xml'))
        #bboxes, polys, kpts = self.load_annoatation(txt_fn)
        bboxes, kpts, thicks, _, polys = self.load_annoatation_xml(txt_fn)
        img_path = os.path.join(self.img_dir, 'images', img_id)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if polys.shape[0] > 0:
            polys = np.reshape(polys, (polys.shape[0], 14, 2))
            gt_text = self.gen_masks(polys, img.shape[0:2])
        else:
            gt_text = np.zeros(img.shape[0:2], dtype='uint8')

        # generate gt_text
        #gt_text = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        #if polys.shape[0] > 0:
        #    polys = np.reshape(polys, (polys.shape[0], 14, 2))
        #    for polygon in polys:
        #        cv2.drawContours(gt_text, [polygon], -1, 1, -1)

        gt_text = gt_text[..., np.newaxis]
        gt_text = gt_text.astype(np.float32)
        img = img.astype(np.float32)
        img = np.concatenate([img, gt_text], axis=-1)

        # image resize and cut to 512x512
        size = (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT)
        img, bboxes, kpts, thicks = scale_wh(img, bboxes, kpts, thicks, size)
        # data augmentation np.random.randint(5) 0,1,2,3,4
        '''
        flag = np.random.randint(5)
        if flag < 2:
            img, bboxes, kpts, thicks = random_scale(img, bboxes, kpts, thicks)
        elif flag == 2:
            img, bboxes, kpts = random_crop(img, bboxes, kpts)
            img, bboxes, kpts, thicks = scale_wh(img, bboxes, kpts, thicks, size)
        elif flag == 3:
            img, bboxes, kpts = random_translate(img, bboxes, kpts)
        elif flag == 4:
            max_angle = 30
            angle = random.random() * 2 * max_angle - max_angle
            #print(img.shape, bboxes.shape, kpts.shape)
            img, bboxes, kpts = random_rotate(img, bboxes, kpts, angle=angle)
            img, bboxes, kpts, thicks = scale_wh(img, bboxes, kpts, thicks, size)
        '''
        #print(img.shape, bboxes.shape, kpts.shape)
        # 处理图像，缩放到指定大小
        img = self.impad_to_wh(img, self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT)
        gt_text = img[..., -1]
        img = img[..., :-1]
        scale_factor = 1.0
        img = self.imnormalize(img, self.config.MEAN_PIXEL, self.config.STD_PIXEL)

        one = np.ones_like(gt_text, dtype=np.uint8)
        zero = np.zeros_like(gt_text, dtype=np.uint8)
        gt_center = np.where(gt_text>0.6, one, zero)
        gt_text = np.where(gt_text>0.01, one, zero)

        gt_kernals = []
        gt_kernals.append(gt_text)
        gt_kernals.append(gt_center)

        gt_kernals = np.stack(gt_kernals, axis=-1)
        gt_kernals = gt_kernals.astype(np.float32)

        return img, gt_kernals


    def shuffle(self):
        random.shuffle(self.images)

    def get_img_list(self, img_path, exts=['jpg']):
        
        img_list = os.listdir(img_path)
        new_list = []
        for img_name in img_list:
            for ext in exts:
                if img_name.endswith(ext):
                    new_list.append(img_name)
                    break
        return new_list


    def load_annoatation_xml(self, xml_file):

        tree = ET.parse(xml_file)
        objs = tree.find('image').findall('box')
        bboxes = []
        polys = []
        kpts = []
        labels = []
        thicks = []
        for ix, obj in enumerate(objs):

            left = int(obj.get('left'))
            top = int(obj.get('top'))
            width = int(obj.get('width'))
            height = int(obj.get('height'))

            label = obj.find('label').text
            segs = obj.find('segs').text
            slist = segs.strip().split(',')
            poly = [np.int(slist[i]) for i in range(0, 28)]
            poly = np.asarray(poly)
            kpt = []
            thick = []
            for i in range(7):
                xu = int(slist[i*2])
                yu = int(slist[1+i*2]) 
                xd = int(slist[26-i*2])
                yd = int(slist[27-i*2])
                kp = [int((xu+xd)/2), int((yu+yd)/2)]
                kpt.append(kp)
                thick.append(math.sqrt(((xd-xu)**2)+((yd-yu)**2)))
            kpt = np.asarray(kpt)
            #kpt = kpt[[0,1,2,4,5,6,3]]
            thick = np.asarray(thick)

            bboxes.append([left, top, left+width, top+height])
            labels.append(label)
            polys.append(poly)
            kpts.append(kpt)
            thicks.append(thick)

        bboxes = np.asarray(bboxes)
        polys = np.asarray(polys)
        kpts = np.asarray(kpts)
        thicks = np.asarray(thicks)

        return bboxes, kpts, thicks, labels, polys


    def load_annoatation(self, txt_fn):
        '''
        load annotation from the text file
        :param p:
        :return:
        '''
        bboxes = []
        polys = []
        kpts = []
        with open(txt_fn, 'r', encoding='utf-8-sig') as f:
            for line in f.readlines():
                #print(im_name)
                slist = line.strip().split(',')

                x1 = np.int(slist[0])
                y1 = np.int(slist[1])
                x2 = np.int(slist[2])
                y2 = np.int(slist[3])
                poly = [np.int(slist[i]) for i in range(4, 32)]
                poly = np.asarray(poly) + ([x1, y1] * 14)

                bbox = np.asarray([x1, y1, x2, y2])
                kpt = []
                for i in range(7):
                    xu = int(slist[4+i*2]) + x1
                    yu = int(slist[5+i*2]) + y1
                    xd = int(slist[30-i*2]) + x1
                    yd = int(slist[31-i*2]) + y1
                    kp = [int((xu+xd)/2), int((yu+yd)/2)]
                    kpt.append(kp)
                kpt = np.asarray(kpt)

                bboxes.append(bbox)
                polys.append(poly)
                kpts.append(kpt)

        bboxes = np.asarray(bboxes)
        polys = np.asarray(polys)
        kpts = np.asarray(kpts)

        return bboxes, polys, kpts


    def imrescale(self, img, scale):
        h, w = img.shape[:2]
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        new_size = (int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5))
        rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return rescaled_img, scale_factor

    def imrescale_wh(self, img, width, height):
        h, w = img.shape[:2]

        scale_factor = min(width*1.0/w, height*1.0/h)
        new_size = (int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5))
        rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return rescaled_img, scale_factor


    def imnormalize(self, img, mean, std):
        img = (img - mean) / std    
        return img.astype(np.float32)

    def impad_to_square(self, img, pad_size):
        h,w = img.shape[:2]
        if len(img.shape) == 2:
            pad_size = [[0,pad_size-h], [0,pad_size-w]]
        else:
            pad_size = [[0,pad_size-h], [0,pad_size-w], [0,0]]
        pad = np.pad(img, pad_size, 'constant')
        return pad

    def impad_to_wh(self, img, width, height):
        h,w = img.shape[:2]
        if len(img.shape) == 2:
            pad_size = [[0,height-h], [0,width-w]]
        else:
            pad_size = [[0,height-h], [0,width-w], [0,0]]
        pad = np.pad(img, pad_size, 'constant')
        return pad

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size
        ra = 0.1155*height
        rb = 0.1155*width
        return ra, rb

    def gaussian2D(self, shape, sigmah=1, sigmaw=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x / (2*sigmaw*sigmaw) + y * y / (2*sigmah*sigmah)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, heatmap, center, rw, rh, k=1):
        diameterw = 2 * rw + 1
        diameterh = 2 * rh + 1
        gaussian = self.gaussian2D((diameterh, diameterw), sigmah=diameterh/6, sigmaw=diameterw/6)

        x, y = center

        height, width = heatmap.shape[0:2]

        left, right = min(x, rw), min(width - x, rw + 1)
        top, bottom = min(y, rh), min(height - y, rh + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[rh - top:rh + bottom, rw - left:rw + right]
        if min(masked_gaussian.shape)>0 and min(masked_heatmap.shape)>0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap

    def gaussian_radius_c(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 - sq1) / (2 * a1)

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 - sq2) / (2 * a2)

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)

    def gaussian2D_c(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian_c(self, heatmap, center, radius, k=1):

        diameter = 2 * radius + 1
        gaussian = self.gaussian2D_c((diameter, diameter), sigma=diameter / 6)

        x, y = center

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    def gen_masks(self, polygons, shape):

        #center_dist = np.zeros(shape, dtype='float32')
        mask = np.zeros(shape, dtype='uint8')
        gt_text = np.zeros(shape, dtype='float32')
        for i in range(polygons.shape[0]):
            #cv2.drawContours(gt_text, [polygons[i]], -1, 1, -1)

            mask = (mask * 0).astype('uint8')
            cv2.drawContours(mask, [polygons[i]], -1, 1, -1)
            mask = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3)
            if np.max(mask) > 0:
                mask = mask/np.max(mask)

            #mask = np.where(mask>=0.55, 1., 0)
            #mask = mask.astype('uint8')
            gt_text = np.where(gt_text > mask, gt_text, mask)

        return gt_text