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
            self.pred_hm, self.pred_wh, self.pred_hm_hp, self.pred_hp_kp, \
                    self.pred_kp_wid, self.segment = self._build_model(inputs)
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

        return hm, wh, hm_hp, hp_kp, kp_wid, segment

    def compute_loss(self, true_hm, true_wh, reg_mask, ind, true_hm_hp, 
                     true_kpt, kpt_mask, true_kp_wid, gt_mask):

        text, center = tf.split(value=self.segment, num_or_size_splits=2, axis=3)
        gtext, gcenter = tf.split(value=gt_mask, num_or_size_splits=2, axis=3)

        training_mask = tf.ones_like(gtext)

        #selected_masks = tf.py_func(ohem_batch, [text, gtext, training_mask], tf.float32)

        # seg loss
        seg_loss1 = loss.dice_coefficient(text, gtext, training_mask)
        seg_loss2 = loss.dice_coefficient(center, gcenter, training_mask)
        #seg_loss = loss.cross_entropy_loss(gtext, text, selected_masks)
        #seg_loss += loss.cross_entropy_loss(gcenter, center, selected_masks)

        seg_loss = 5 * (seg_loss1 + seg_loss2)

        #one = tf.ones_like(text)
        #zero = tf.zeros_like(text)
        #W = tf.where(text >= 0.5, x=one, y=zero)

        hm_loss = loss.focal_loss(self.pred_hm, true_hm)
        wh_loss = 0.05*loss.reg_l1_loss(self.pred_wh, true_wh, ind, reg_mask)

        #pose hm_hp
        hm_hp_loss = loss.focal_loss(self.pred_hm_hp, true_hm_hp)
        kpt_loss = 0.05*loss.reg_l1_loss_kpt(self.pred_hp_kp, true_kpt, ind, kpt_mask)
        kpt_wid_loss = 0.05*loss.reg_l1_loss(self.pred_kp_wid, true_kp_wid, ind, reg_mask)

        return hm_loss, wh_loss, hm_hp_loss, kpt_loss, kpt_wid_loss, seg_loss

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
