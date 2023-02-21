import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import time
import cfg_text
from nets import model_n
from utils.decode import decode_text
from utils.image import get_affine_transform, affine_transform
from utils.utils import image_preporcess, py_nms, post_process_text, text_draw_on_img, write_result_as_txt

from pse import pse

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

def detect(seg_maps, min_area_thresh=10, seg_map_thresh=0.5, ratio = 1):

    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)

    kernal_0 = np.where(seg_maps[..., 0]>seg_map_thresh, one, zero)
    kernal_1 = np.where(seg_maps[..., 1]>seg_map_thresh, one, zero)
    kernals.append(kernal_1)
    kernals.append(kernal_0)

    mask_res, label_values = pse(kernals, min_area_thresh)

    score_map = seg_maps[..., 1].astype(np.float32)

    mask_res = np.array(mask_res)

    boxes = []
    for label_value in label_values:

        score_i = np.mean(score_map[mask_res == label_value])

        if score_i < 0.8:
            continue
        #print('#########')
        points_map = np.where(mask_res==label_value, 1, 0)
        points_map = points_map[:, :, np.newaxis].astype('uint8')
        #points = points[:, (1,0)]
        contours, _ = cv2.findContours(points_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        #rect = cv2.minAreaRect(points)
        #box = cv2.boxPoints(rect)
        if contour.shape[0] <= 2:
            continue
        boxes.append(contour)

    return boxes, kernals, score_map


def inference():

    ckpt_path='./checkpoint/ctw/weight50_ctw2-23750'
    sess = tf.Session()
    cfgs = cfg_text.Config()
    inputs = tf.placeholder(shape=[None, None, None, 3],dtype=tf.float32)

    seg_maps_pred, hm, wh, hm_hp, hp_kp, kp_wid = model_n.model(inputs, is_training=False)

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    img_folder = 'data/ctw1500/test/images'
    out_path = 'data/ctw1500/test/out'

    img_names = os.listdir(img_folder)
    for i, img_name in enumerate(img_names):
        img_path = os.path.join(img_folder, img_name)
        original_image = cv2.imread(img_path)
        original_image_size = original_image.shape[:2]
        image_data = image_preporcess(np.copy(original_image), [cfgs.IMAGE_HEIGHT, cfgs.IMAGE_WIDTH], cfgs.MEAN_PIXEL)
        image_data = image_data[np.newaxis, ...]

        t0 = time.time()
        seg_maps = sess.run(seg_maps_pred, feed_dict={inputs: image_data})
        
        polygons, kernels, score_map = detect(seg_maps=seg_maps)


        print('No.%d Inferencce took %.1f ms (%.2f fps)' % (i, (time.time()-t0)*1000, 1/(time.time()-t0)), img_name)
        polygons_r = []
        if len(polygons) > 0:
            h, w, _ = original_image.shape
            ratio = min(cfgs.IMAGE_WIDTH / w, cfgs.IMAGE_HEIGHT / h)

            for polygon in polygons:
                polygon = np.squeeze(polygon).astype(np.float32)
                #print(polygon.shape)
                polygon[:, 0] = polygon[:, 0] / ratio
                polygon[:, 1] = polygon[:, 1] / ratio
                polygon[:, 0] = np.clip(polygon[:, 0], 0, w)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, h)
                polygons_r.append(polygon)


        # save to file
        if len(polygons_r) > 0:
            txt_name = img_name.replace('.jpg', '.txt')
            write_result_as_txt(txt_name, polygons_r, 'data/ctw1500/test/txt')
            print("save: ", txt_name)

        cv2.imwrite(os.path.join(out_path, img_name), original_image)

        # center
        segs = seg_maps[0]
        segs = np.clip(segs, 0, 1)*255
        segs = segs.astype(np.uint8)
        cmap = segs[..., 0]
        tmap = segs[..., 1]
        cname = os.path.join(out_path, img_name.replace('.jpg', '_2.jpg'))
        tname = os.path.join(out_path, img_name.replace('.jpg', '_3.jpg'))
        cv2.imwrite(cname, cmap)
        cv2.imwrite(tname, tmap)


if __name__ == '__main__':

    inference()