import tensorflow as tf
import loss
from net import resnet
from net.layers import _conv, upsampling
import numpy as np

def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
    
    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1], 1).astype('float32')
        return selected_mask
    
    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))
    
    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1], 1).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1], 1).astype('float32')
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :, :], gt_texts[i, :, :, :], training_masks[i, :, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)

    return selected_masks


class TextNet():
    def __init__(self, inputs, heads, is_training): # heads={'hm':1, 'offset':2, 'hm_hp':6, 'hp_kp':12, 'hp_offset':2, 'kpwidth':6}
        self.is_training = is_training
        self.heads = heads
        try:
            self.pred_hm, self.pred_wh, self.pred_hm_hp, self.pred_hp_kp, self.pred_kp_wid, \
                    self.segment, self.pred_det, self.pred_logits, self.features = self._build_model(inputs)
        except:
            raise NotImplementedError("Can not build up textnet network!")

    def _build_model(self, inputs):
        
        with tf.variable_scope('resnet'):
            c2, c3, c4, c5 = resnet.resnet50(is_training=self.is_training).forward(inputs)

            # FPN
            p5 = _conv(c5, 256, [1,1], is_training=self.is_training)

            up_p5 = upsampling(p5, rate=2, method='resize')
            reduce_dim_c4 = _conv(c4, 256, [1,1], is_training=self.is_training)
            p4 = 0.5*up_p5 + 0.5*reduce_dim_c4

            up_p4 = upsampling(p4, rate=2, method='resize')
            reduce_dim_c3 = _conv(c3, 256, [1,1], is_training=self.is_training)
            p3 = 0.5*up_p4 + 0.5*reduce_dim_c3

            up_p3 = upsampling(p3, rate=2, method='resize')
            reduce_dim_c2 = _conv(c2, 256, [1,1], is_training=self.is_training)
            p2 = 0.5*up_p3 + 0.5*reduce_dim_c2

        with tf.variable_scope('concat'):
            # concat
            P_concat = []
            P_concat.append(p2)
            P_concat.append(upsampling(p3, rate=2, method='resize'))
            P_concat.append(upsampling(p4, rate=4, method='resize'))
            P_concat.append(upsampling(p5, rate=8, method='resize'))
            F = tf.concat(P_concat, axis=-1)

            features = _conv(F, 256, [3,3], is_training=self.is_training)

        with tf.variable_scope('segmenter'):
            segment = _conv(features, 2, [3,3], is_training=self.is_training)
            segment = upsampling(segment, rate=4, method='resize')
            segment = tf.nn.sigmoid(segment)
            
        with tf.variable_scope('detector'):

            hm = _conv(features, 128, [3,3], is_training=self.is_training)
            hm = tf.layers.conv2d(hm, self.heads['hm'], 1, 1, padding='valid', activation = tf.nn.sigmoid, 
                                  bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm')
            hm = upsampling(hm, rate=4, method='resize')
            # wh
            wh = _conv(features, 128, [3,3], is_training=self.is_training)
            wh = tf.layers.conv2d(wh, self.heads['wh'], 1, 1, padding='valid', activation = None, name='wh')
            wh = upsampling(wh, rate=4, method='resize')

            # pose heat map
            hm_hp = _conv(features, 128, [3,3], is_training=self.is_training)
            hm_hp = tf.layers.conv2d(hm_hp, self.heads['hm_hp'], 1, 1, padding='valid', activation = tf.nn.sigmoid, 
                                  bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm_hp')
            hm_hp = upsampling(hm_hp, rate=4, method='resize')

            # pose keypoints
            hp_kp = _conv(features, 128, [3,3], is_training=self.is_training)
            hp_kp = tf.layers.conv2d(hp_kp, self.heads['hp_kp'], 1, 1, padding='valid', activation = None, name='hp_kp')
            hp_kp = upsampling(hp_kp, rate=4, method='resize')

            # kpwidth
            kp_wid = _conv(features, 128, [3,3], is_training=self.is_training)
            kp_wid = tf.layers.conv2d(kp_wid, self.heads['kpwidth'], 1, 1, padding='valid', activation = None, name='kp_wid')
            kp_wid = upsampling(kp_wid, rate=4, method='resize')


        with tf.variable_scope('recognizer'):
            #for training, using labels
            if self.is_training:
                det = labels
            else:
                det = decode_text(hm, wh, hm_hp, hp_kp, kp_wid, K=20)

            ch, cw = self.cfgs.roi_h, self.cfgs.roi_w
            batch = self.cfgs.BATCH_SIZE
            img_shape = tf.shape(features)
            #det = decode_hp(hm, wh, reg, hm_hp, hp_kp, hp_off, K=1)
            # get cropped rois as the lp area
            boxes = det[:,:,0:4]
            boxes = tf.reshape(boxes, (-1,4))
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h
            N = img_shape[0]
            normalized_rois = tf.transpose(tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]))

            #features = tf.stop_gradient(features)
            k = self.k
            ind = tf.cast(tf.range(0, N*k)/k, tf.int32)
            cropped_roi_features = tf.image.crop_and_resize(features, normalized_rois,
                                                            box_ind=ind,
                                                            crop_size=[ch, cw],
                                                            name='CROP_AND_RESIZE')
            # affine transform by detected four kpts of lp
            kpts = det[:,:,5:13]
            kpts = tf.reshape(kpts, (-1,8))
            kx1, ky1, kx2, ky2, kx3, ky3, kx4, ky4 = \
                    kpts[:,0],kpts[:,1],kpts[:,2],kpts[:,3],kpts[:,4],kpts[:,5],kpts[:,6],kpts[:,7]
            #normalize kpts to box position
            kx1 = (kx1-x1)*cw/(x2-x1+1)
            kx2 = (kx2-x1)*cw/(x2-x1+1)
            kx3 = (kx3-x1)*cw/(x2-x1+1)
            kx4 = (kx4-x1)*cw/(x2-x1+1)
            ky1 = (ky1-y1)*ch/(y2-y1+1)
            ky2 = (ky2-y1)*ch/(y2-y1+1)
            ky3 = (ky3-y1)*ch/(y2-y1+1)
            ky4 = (ky4-y1)*ch/(y2-y1+1)
            kpts = tf.transpose(tf.stack([kx1, ky1, kx2, ky2, kx3, ky3, kx4, ky4]))
            kpts = tf.reshape(kpts, (batch*k, 4, 2))
            kpts = kpts[:,:,::-1]
            #tgt_pts = tf.convert_to_tensor([[0, 0], [0, cw], [ch, cw], [ch, 0]])
            #tgt_pts = tf.broadcast_to(tgt_pts, shape=[batch, 4, 2])
            # lp feature after transform
            tgt_image = four_point_transform_2D_batch(cropped_roi_features, kpts, self.tgt_pts)

            # recognition net
            # N*36*96*128
            con1 = _conv(tgt_image, 128, [3,3], is_training=self.is_training)
            # N*36*48*128
            pool1 = tf.nn.max_pool(con1, ksize=[1, 3, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
            # N*36*48*128
            con2 = _conv(pool1, 128, [3,3], is_training=self.is_training)
            # N*36*24*128
            pool2 = tf.nn.max_pool(con2, ksize=[1, 3, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
            # N*36*24*128
            con3 = _conv(pool2, 128, [3,3], is_training=self.is_training)
            # N*36*24*128
            con4 = _conv(con3, 128, [3,3], is_training=self.is_training)
            # N*36*24*67
            con5 = _conv(con4, NUM_CHARS+1, [1,1], is_training=self.is_training)
            # N*24*67
            logits = tf.reduce_mean(con5, axis=1)
            # 24*N*67
            logits = tf.transpose(logits, (1, 0, 2), name='logits')

        return hm, wh, hm_hp, hp_kp, kp_wid, segment, det, logits, features

    def detect_loss(self, true_hm, true_wh, reg_mask, ind, true_hm_hp, true_kpt, kpt_mask, true_kp_wid, gt_mask):

        text, center = tf.split(value=self.segment, num_or_size_splits=2, axis=3)
        gtext, gcenter = tf.split(value=gt_mask, num_or_size_splits=2, axis=3)

        training_mask = tf.ones_like(gtext)

        selected_masks = tf.py_func(ohem_batch, [text, gtext, training_mask], tf.float32)

        # seg loss
        seg_loss = loss.dice_coefficient(text, gtext, selected_masks)
        seg_loss = seg_loss + loss.dice_coefficient(center, gcenter, selected_masks)
        #seg_loss = loss.cross_entropy_loss(gtext, text, selected_masks)
        #seg_loss += loss.cross_entropy_loss(gcenter, center, selected_masks)

        #seg_loss = 5 * seg_loss

        one = tf.ones_like(text)
        zero = tf.zeros_like(text)
        W = tf.where(text >= 0.5, x=one, y=zero)

        hm_loss = loss.focal_loss_mask(self.pred_hm, true_hm, selected_masks*W)
        wh_loss = 0.05*loss.reg_l1_loss(self.pred_wh, true_wh, ind, reg_mask)

        #pose hm_hp
        hm_hp_loss = loss.focal_loss_mask(self.pred_hm_hp, true_hm_hp, selected_masks*W)
        kpt_loss = 0.05*loss.reg_l1_loss_kpt(self.pred_hp_kp, true_kpt, ind, kpt_mask)
        kpt_wid_loss = 0.05*loss.reg_l1_loss(self.pred_kp_wid, true_kp_wid, ind, reg_mask)

        return hm_loss, wh_loss, hm_hp_loss, kpt_loss, kpt_wid_loss, seg_loss

    def recog_loss(self, targets):

        #print('######################')
        #print(self.logits)
        seq_len = tf.constant(np.ones(self.cfgs.BATCH_SIZE*self.k, dtype=np.int32) * 24)
        loss = tf.nn.ctc_loss(labels=targets, inputs=self.pred_logits, sequence_length=seq_len)
        loss = tf.reduce_mean(loss)

        return loss


    def predict(self):
        
        text, center = tf.split(value=self.segment, num_or_size_splits=2, axis=3)
        zero = tf.zeros_like(center)
        one = tf.ones_like(center)
        W = tf.where(center >= 0.5, x=one, y=zero)
        self.pred_hm = self.pred_hm * W

        zero1 = tf.zeros_like(text)
        one1 = tf.ones_like(text)
        W1 = tf.where(text >= 0.5, x=one, y=zero)
        self.pred_hm_hp = self.pred_hm_hp * W1
        return self.pred_hm, self.pred_wh, self.pred_hm_hp, self.pred_hp_kp, self.pred_kp_wid, self.segment
