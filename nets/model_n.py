#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets.resnet import resnet_v1
from nets.layers import _conv, upsampling
from nets.loss import focal_loss_mask, reg_l1_loss, reg_l1_loss_kpt

FLAGS = tf.app.flags.FLAGS

#TODO:bilinear or nearest_neighbor?
def unpool(inputs, rate):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*rate,  tf.shape(inputs)[2]*rate])

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def build_feature_pyramid(C, weight_decay):

    '''
    reference: https://github.com/CharlesShang/FastMaskRCNN
    build P2, P3, P4, P5
    :return: multi-scale feature map
    '''
    feature_pyramid = {}
    with tf.variable_scope('build_feature_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)):
            feature_pyramid['P5'] = slim.conv2d(C['C5'],
                                                num_outputs=256,
                                                kernel_size=[1, 1],
                                                stride=1,
                                                scope='build_P5')

            # feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
            #                                         kernel_size=[2, 2], stride=2, scope='build_P6')
            # P6 is down sample of P5

            for layer in range(4, 1, -1):
                p, c = feature_pyramid['P' + str(layer + 1)], C['C' + str(layer)]
                up_sample_shape = tf.shape(c)
                #up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                #                                             name='build_P%d/up_sample_nearest_neighbor' % layer)
                up_sample = tf.image.resize_bilinear(p, [up_sample_shape[1], up_sample_shape[2]],
                                                             name='build_P%d/up_sample_nearest_neighbor' % layer)

                c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                scope='build_P%d/reduce_dimension' % layer)
                p = up_sample + c
                p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1,
                                padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
                feature_pyramid['P' + str(layer)] = p
    return feature_pyramid


def model(images, outputs = 2, weight_decay = 1e-5, is_training = True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    #images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    #no non-linearities in FPN article
    feature_pyramid = build_feature_pyramid(end_points, weight_decay=weight_decay)
    #unpool sample P
    P_concat = []
    for i in range(3, 0, -1):
        P_concat.append(unpool(feature_pyramid['P'+str(i+2)], 2**i))
    P_concat.append(feature_pyramid['P2'])
    #F = C(P2,P3,P4,P5)
    F = tf.concat(P_concat, axis=-1)

    #reduce to 256 channels
    with tf.variable_scope('feature_results'):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            F = slim.conv2d(F, 256, 3)
        # -2s
        F = unpool(F, 2)

    return F
        
    #     with slim.arg_scope([slim.conv2d],
    #                         weights_regularizer=slim.l2_regularizer(weight_decay),
    #                         activation_fn=None):
    #         S = slim.conv2d(F, outputs, 1)
        
    # #S = unpool(S, 2)
    # result = tf.nn.sigmoid(S)

    # heads={'hm':1, 'wh':4, 'hm_hp':7, 'hp_kp':14, 'kpwidth':7}
    
    # F = tf.concat([F, result], axis=-1)
    # F = _conv(F, 128, [3,3], is_training=is_training)

    # # hm
    # hm = _conv(F, 128, [3,3], is_training=is_training)
    # hm = tf.layers.conv2d(hm, heads['hm'], 1, 1, padding='valid', activation = tf.nn.sigmoid, 
    #                       bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm')
    # #hm = upsampling(hm, rate=2, method='resize')
    # # wh
    # wh = _conv(F, 128, [3,3], is_training=is_training)
    # wh = tf.layers.conv2d(wh, heads['wh'], 1, 1, padding='valid', activation = None, name='wh')
    # #wh = upsampling(wh, rate=2, method='resize')

    # # pose heat map
    # hm_hp = _conv(F, 128, [3,3], is_training=is_training)
    # hm_hp = tf.layers.conv2d(hm_hp, heads['hm_hp'], 1, 1, padding='valid', activation = tf.nn.sigmoid, 
    #                       bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm_hp')
    # #hm_hp = upsampling(hm_hp, rate=2, method='resize')

    # # pose keypoints
    # hp_kp = _conv(F, 128, [3,3], is_training=is_training)
    # hp_kp = tf.layers.conv2d(hp_kp, heads['hp_kp'], 1, 1, padding='valid', activation = None, name='hp_kp')
    # #hp_kp = upsampling(hp_kp, rate=2, method='resize')

    # # kpwidth
    # kp_wid = _conv(F, 128, [3,3], is_training=is_training)
    # kp_wid = tf.layers.conv2d(kp_wid, heads['kpwidth'], 1, 1, padding='valid', activation = None, name='kp_wid')
    # #kp_wid = upsampling(kp_wid, rate=2, method='resize')
    
    # return result, hm, wh, hm_hp, hp_kp, kp_wid


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    eps = 1e-5
    y_true_cls = y_true_cls * training_mask
    y_pred_cls = y_pred_cls * training_mask
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls)
    union = tf.reduce_sum(y_true_cls * y_true_cls) + tf.reduce_sum(y_pred_cls * y_pred_cls) + eps
    dice = 2 * intersection / union
    loss = 1. - dice
    # tf.summary.scalar('classification_dice_loss', loss)
    return loss

def dice_coefficient_1(y_true_cls, y_pred_cls, training_mask):

    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    dice = 2 * intersection / union
    loss = 1. - dice
    # tf.summary.scalar('classification_dice_loss', loss)
    return dice, loss

def smooth_l1(y_true_cls, y_pred_cls, gt_text, training_mask, sigma=1.0):
    y_true_cls = y_true_cls * training_mask
    y_pred_cls = y_pred_cls * training_mask
    sigma_2 = sigma ** 2
    diff = tf.abs(y_true_cls - y_pred_cls)
    #smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(diff, 1.0)))
    smoothL1_sign = tf.to_float(tf.less(diff, 1. / sigma_2))
    #smoothL1_sign = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = tf.pow(diff, 2) * (sigma_2 / 2.0) * smoothL1_sign + (diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    loss = tf.reduce_sum(loss) / tf.maximum(1.0, tf.reduce_sum(gt_text))
    return loss

def loss_l1(y_true_cls, y_pred_cls, gt_text, training_mask):
    y_true_cls = y_true_cls * training_mask
    y_pred_cls = y_pred_cls * training_mask * gt_text
    diff = tf.abs(y_true_cls - y_pred_cls)
    loss = tf.reduce_sum(diff) / tf.maximum(1.0, tf.reduce_sum(gt_text))
    return loss

def cross_entropy_loss(y_true_cls, y_pred_cls, training_mask):
    y_true_cls = y_true_cls * training_mask
    y_pred_cls = y_pred_cls * training_mask
    loss =  tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls)
    loss = tf.reduce_mean(loss)
    return loss

def loss_n(y_pred_cls, y_true_cls, pred_hm, pred_wh, pred_hm_hp, pred_hp_kp, pred_kp_wid, 
           true_hm, true_wh, reg_mask, ind, true_hm_hp, true_kpt, kpt_mask, true_kp_wid):
    
    g1, g2= tf.split(value=y_true_cls, num_or_size_splits=2, axis=3)
    s1, s2 = tf.split(value=y_pred_cls, num_or_size_splits=2, axis=3)
    Gn = [g1, g2]
    Sn = [s1, s2]

    training_mask = tf.ones_like(g1)
    selected_masks = tf.py_func(ohem_batch, [Sn[1], Gn[1], training_mask], tf.float32)

    Ls = dice_coefficient(Gn[0], Sn[0], training_mask=selected_masks)
    #tf.summary.scalar('Ls_loss', Ls)

    Lt = dice_coefficient(Gn[1], Sn[1], training_mask=selected_masks)
    #tf.summary.scalar('Lc_loss', Lt)

    one = tf.ones_like(Sn[1])
    zero = tf.zeros_like(Sn[1])
    Wt = tf.where(Sn[1] >= 0.5, x=one, y=zero)
    Wc = tf.where(Sn[0] >= 0.5, x=one, y=zero)

    hm_loss = focal_loss_mask(pred_hm, true_hm, selected_masks*Wc)
    wh_loss = 0.1*reg_l1_loss(pred_wh, true_wh, ind, reg_mask)

    #pose hm_hp
    hm_hp_loss = focal_loss_mask(pred_hm_hp, true_hm_hp, selected_masks*Wt)
    kpt_loss = 0.1*reg_l1_loss_kpt(pred_hp_kp, true_kpt, ind, kpt_mask)
    kpt_wid_loss = 0.1*reg_l1_loss(pred_kp_wid, true_kp_wid, ind, reg_mask)
    
    L1 = 5*(Lt + Ls)
    L2 = hm_loss + wh_loss + hm_hp_loss + kpt_loss + kpt_wid_loss
    return L1, L2, hm_loss, wh_loss, hm_hp_loss, kpt_loss, kpt_wid_loss


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


