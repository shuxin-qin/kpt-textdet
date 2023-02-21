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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

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
    #cv2.addWeighted(overlay, alpha, merge_img, 1-alpha, 0, merge_img) # 将背景热度图覆盖到原图
    cv2.addWeighted(heatmap_img, alpha, merge_img, 1-alpha, 0, merge_img) # 将热度图覆盖到原图

    #heatmap_g = heatmap_g[..., np.newaxis]
    #merge_img = np.where(heatmap_g>0, merge_img, image)

    return heatmap_color, merge_img

def inference():

    ckpt_path='./checkpoint/ctw/weight50_ctw3-15000'
    sess = tf.Session()
    cfgs = cfg_text.Config()
    inputs = tf.placeholder(shape=[None, None, None, 3],dtype=tf.float32)

    seg_maps_pred, hm, wh, hm_hp, hp_kp, kp_wid = model_n.model(inputs, is_training=False)

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    center, text = tf.split(value=seg_maps_pred, num_or_size_splits=2, axis=3)
    zero = tf.zeros_like(center)
    one = tf.ones_like(center)
    Wc = tf.where(center >= 0.5, x=one, y=zero)
    hm = hm * Wc

    Wt = tf.where(text >= 0.5, x=one, y=zero)
    hm_hp = hm_hp * Wt

    det = decode_text(hm, wh, hm_hp, hp_kp, kp_wid, K=100)

    img_folder = 'data/ctw1500/test/images1'
    out_path = 'data/ctw1500/test/out1'

    img_names = os.listdir(img_folder)
    for i, img_name in enumerate(img_names):
        img_path = os.path.join(img_folder, img_name)
        original_image = cv2.imread(img_path)
        original_image_size = original_image.shape[:2]
        image_data = image_preporcess(np.copy(original_image), [cfgs.IMAGE_HEIGHT, cfgs.IMAGE_WIDTH], cfgs.MEAN_PIXEL)
        image_data = image_data[np.newaxis, ...]

        t0 = time.time()
        detections, hps, hms, seg_maps = sess.run([det, hm_hp, hm, seg_maps_pred], feed_dict={inputs: image_data})
        detections = post_process_text(detections, original_image_size, [cfgs.IMAGE_HEIGHT, cfgs.IMAGE_WIDTH], cfgs.down_ratio, cfgs.score_threshold)
        print('No.%d Inferencce took %.1f ms (%.2f fps)' % (i, (time.time()-t0)*1000, 1/(time.time()-t0)), img_name)
        polygons = []
        if cfgs.use_nms:
            results = []
            classified_bboxes = detections[:, :4]
            classified_scores = detections[:, 4]
            inds = py_nms(classified_bboxes, classified_scores, max_boxes=50, iou_thresh=0.3)
            results.extend(detections[inds])
            results = np.asarray(results)
            if len(results) > 0:
                bboxes = results[:,0:4]
                scores = results[:,4]
                kps = results[:,5:19]
                kp_wids = results[:,19:-1]
                #print(kp_wids)
                #print(bboxes)
                #polygons, _ = text_draw_on_img(original_image, scores, bboxes, kps, kp_wids, thickness=2, drawkp=False)
            
        else:
            bboxes = detections[:,0:4]
            scores = detections[:,4]
            kps = detections[:,5:19]
            kp_wids = detections[:,19:-1]
            #polygons, _ = text_draw_on_img(original_image, scores, bboxes, kps, kp_wids, thickness=2, drawkp=False)

        # save to file
        if len(polygons) > 0:
            txt_name = img_name.replace('.jpg', '.txt')
            write_result_as_txt(txt_name, polygons, 'data/ctw1500/test/txt')

        cv2.imwrite(os.path.join(out_path, img_name), original_image)

        # 还原到原始图像大小
        org_h, org_w = original_image_size
        ratio = min(cfgs.IMAGE_WIDTH / org_w, cfgs.IMAGE_HEIGHT / org_h)
        nw, nh  = int(cfgs.IMAGE_WIDTH/ratio + 0.5), int(cfgs.IMAGE_HEIGHT/ratio + 0.5)

        # hp
        hps = hps[0]
        hps = np.max(hps, axis=-1)
        hps = np.clip(hps, 0, 1)*255
        hmm = hps.astype(np.uint8)
        hmm = cv2.resize(hmm, (nw, nh))
        hmm = hmm[:org_h, :org_w]

        hmm, hmm_m = heatmap_overlay(hmm, original_image)

        hname = os.path.join(out_path, img_name.replace('.jpg', '_06W.jpg'))
        cv2.imwrite(hname, hmm)
        h_mname = os.path.join(out_path, img_name.replace('.jpg', '_03.jpg'))
        cv2.imwrite(h_mname, hmm_m)
        '''
        # hm
        hms = hms[0]
        hms = np.clip(hms, 0, 1)*255
        hms = hms.astype(np.uint8)
        hms = cv2.resize(hms, (nw, nh))
        hms = hms[:org_h, :org_w]

        hms, hms_m = heatmap_overlay(hms, original_image)

        hmname = os.path.join(out_path, img_name.replace('.jpg', '_10.jpg'))
        cv2.imwrite(hmname, hms)
        #hm_mname = os.path.join(out_path, img_name.replace('.jpg', '_11.jpg'))
        #cv2.imwrite(hm_mname, hms_m)

        # center
        segs = seg_maps[0]
        segs = np.clip(segs, 0, 1)*255
        segs = segs.astype(np.uint8)
        cmap = segs[..., 0]
        tmap = segs[..., 1]
        cmap = cv2.resize(cmap, (nw, nh))
        cmap = cmap[:org_h, :org_w]
        tmap = cv2.resize(tmap, (nw, nh))
        tmap = tmap[:org_h, :org_w]
        cname = os.path.join(out_path, img_name.replace('.jpg', '_2.jpg'))
        tname = os.path.join(out_path, img_name.replace('.jpg', '_3.jpg'))
        cv2.imwrite(cname, cmap)
        cv2.imwrite(tname, tmap)
        '''

if __name__ == '__main__':

    inference()