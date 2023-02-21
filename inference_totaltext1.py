import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import time
import cfg_text
from text_net import TextNet
from utils.decode import decode_text
from utils.image import get_affine_transform, affine_transform
from utils.utils import image_preporcess, py_nms, post_process_text, text_draw_on_img, write_result_as_txt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def inference():

    ckpt_path='./checkpoint/weight50_totaltext-27170'
    sess = tf.Session()
    cfgs = cfg_text.Config()
    inputs = tf.placeholder(shape=[None, None, None, 3],dtype=tf.float32)
    heads={'hm':1, 'wh':4, 'offset':2, 'hm_hp':7, 'hp_kp':14, 'hp_offset':2, 'kpwidth':7}
    model = TextNet(inputs, heads, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    hm, wh, hm_hp, hp_kp, kp_wid, seg = model.predict()
    det = decode_text(hm, wh, hm_hp, hp_kp, kp_wid, K=20)

    img_folder = 'data/total_text/test/images'
    out_path = 'data/total_text/test/out'

    img_names = os.listdir(img_folder)
    for i, img_name in enumerate(img_names):
        img_path = os.path.join(img_folder, img_name)
        original_image = cv2.imread(img_path)
        original_image_size = original_image.shape[:2]
        image_data = image_preporcess(np.copy(original_image), [cfgs.IMAGE_HEIGHT, cfgs.IMAGE_WIDTH], cfgs.MEAN_PIXEL)
        image_data = image_data[np.newaxis, ...]

        t0 = time.time()
        detections, hps, hms = sess.run([det, hm_hp, hm], feed_dict={inputs: image_data})
        detections = post_process_text(detections, original_image_size, [cfgs.IMAGE_HEIGHT, cfgs.IMAGE_WIDTH], cfgs.down_ratio, cfgs.score_threshold)
        print('No.%d Inferencce took %.1f ms (%.2f fps)' % (i, (time.time()-t0)*1000, 1/(time.time()-t0)), img_name)
        polygons = []
        if cfgs.use_nms:
            results = []
            classified_bboxes = detections[:, :4]
            classified_scores = detections[:, 4]
            inds = py_nms(classified_bboxes, classified_scores, max_boxes=20, iou_thresh=0.4)
            results.extend(detections[inds])
            results = np.asarray(results)
            if len(results) > 0:
                bboxes = results[:,0:4]
                scores = results[:,4]
                kps = results[:,5:19]
                kp_wid = results[:,19:-1]
                polygons = text_draw_on_img(original_image, scores, bboxes, kps, kp_wid, thickness=2)
            
        else:
            bboxes = detections[:,0:4]
            scores = detections[:,4]
            kps = detections[:,5:19]
            kp_wids = detections[:,19:-1]
            polygons = text_draw_on_img(original_image, scores, bboxes, kps, kp_wids, thickness=2)


        # save to file
        if len(polygons) > 0:
            txt_name = img_name.replace('.jpg', '.txt')
            write_result_as_txt(txt_name, polygons, 'data/totaltext/test/txt')

        cv2.imwrite(os.path.join(out_path, img_name), original_image)

        # hp
        hps = hps[0]
        hps = np.sum(hps, axis=-1)
        hps = np.clip(hps, 0, 1)*255
        hmm = hps.astype(np.uint8)
        hname = os.path.join(out_path, img_name.replace('.jpg', '_0.jpg'))
        cv2.imwrite(hname, hmm)
        # hm
        hms = hms[0]
        hms = np.clip(hms, 0, 1)*255
        hms = hms.astype(np.uint8)
        hmname = os.path.join(out_path, img_name.replace('.jpg', '_1.jpg'))
        cv2.imwrite(hmname, hms)

if __name__ == '__main__':

    inference()