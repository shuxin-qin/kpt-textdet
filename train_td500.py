import os
import numpy as np
import tensorflow as tf
import cv2
import math
import time
import shutil
import cfg_text_td500
from utils import data_reader_td500, dataset_text
from net.resnet import load_weights
from nets import model_n
from tensorflow.contrib import slim

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def tower_loss(in_imgs, batch_hm, batch_wh, batch_reg_mask, batch_ind, batch_hm_hp, 
                batch_kps, batch_kps_mask, batch_kps_width, batch_gt_text, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):

        #heads={'hm':1, 'wh':4, 'offset':2, 'hm_hp':7, 'hp_kp':14, 'hp_offset':2, 'kpwidth':7}
        #model = TextNet(in_imgs, heads, is_training=True)
        seg_maps_pred, hm, wh, hm_hp, hp_kp, kp_wid = model_n.model(in_imgs, is_training=True)

    #hm_loss, wh_loss, hm_hp_loss, kpt_loss, kpt_wid_loss, seg_loss = model.compute_loss(batch_hm, batch_wh, 
    #        batch_reg_mask, batch_ind, batch_hm_hp, batch_kps, batch_kps_mask, batch_kps_width, batch_gt_text)
    #det_loss = hm_loss + wh_loss + hm_hp_loss + kpt_loss + kpt_wid_loss

    seg_loss, det_loss, hm_loss, wh_loss, hm_hp_loss, kpt_loss, kpt_wid_loss = model_n.loss_n(
        seg_maps_pred, batch_gt_text, hm, wh, hm_hp, hp_kp, kp_wid, batch_hm, batch_wh, batch_reg_mask, batch_ind, batch_hm_hp, batch_kps, batch_kps_mask, batch_kps_width)
    total_loss = tf.add_n([det_loss + seg_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.scalar('det_loss', det_loss)
        tf.summary.scalar('seg_loss', seg_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, det_loss, seg_loss, hm_loss, wh_loss, hm_hp_loss, kpt_loss, kpt_wid_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def train(checkpoint=None):
    # define dataset
    configs = cfg_text_td500.Config()
    heads={'hm':1, 'wh':4, 'offset':2, 'hm_hp':7, 'hp_kp':14, 'hp_offset':2, 'kpwidth':7}

    img_dir = 'data/MSRA-TD500/train'
    data_source = data_reader_td500.DataReader(img_dir, config=configs)
    datasets = dataset_text.Dataset(data_source, batch_size=configs.BATCH_SIZE)

    in_imgs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='in_imgs')
    batch_hm = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='batch_hm')
    batch_wh = tf.placeholder(dtype=tf.float32, shape=[None, None, 4], name='batch_wh')
    batch_reg_mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='batch_reg_mask')
    batch_ind = tf.placeholder(dtype=tf.float32, shape=[None, None], name='batch_ind')

    batch_hm_hp = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 7], name='batch_hm_hp')
    batch_kps = tf.placeholder(dtype=tf.float32, shape=[None, None, 14], name='batch_kps')
    batch_kps_mask = tf.placeholder(dtype=tf.float32, shape=[None, None, 14], name='batch_kps_mask')
    batch_kps_width = tf.placeholder(dtype=tf.float32, shape=[None, None, 7], name='batch_kps_width')
    batch_gt_text = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 2], name='batch_gt_text')

    gpus = [0]

    global_step = tf.train.create_global_step()
    #learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=250, decay_rate=0.9, staircase=True)
    learning_rate = tf.train.piecewise_constant(global_step, [53000], [0.0001, 0.00001])
    opt = tf.train.AdamOptimizer(learning_rate)

    # split
    input_images_split = tf.split(in_imgs, len(gpus))
    input_seg_maps_split = tf.split(batch_gt_text, len(gpus))

    input_hm_split = tf.split(batch_hm, len(gpus))
    input_wh_split = tf.split(batch_wh, len(gpus))
    input_reg_mask_split = tf.split(batch_reg_mask, len(gpus))
    input_ind_split = tf.split(batch_ind, len(gpus))
    input_hm_hp_split = tf.split(batch_hm_hp, len(gpus))
    input_kps_split = tf.split(batch_kps, len(gpus))
    input_kps_mask_split = tf.split(batch_kps_mask, len(gpus))
    input_kps_width_split = tf.split(batch_kps_width, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                ihm = input_hm_split[i]
                iwh = input_wh_split[i]
                ireg_mask = input_reg_mask_split[i]
                iind = input_ind_split[i]
                ihm_hp = input_hm_hp_split[i]
                ikps = input_kps_split[i]
                ikps_mask = input_kps_mask_split[i]
                ikps_width = input_kps_width_split[i]
                isegs = input_seg_maps_split[i]

                total_loss, det_loss, seg_loss, hm_loss, wh_loss, hm_hp_loss, kpt_loss, kpt_wid_loss = tower_loss(
                    iis, ihm, iwh, ireg_mask, iind, ihm_hp, ikps, ikps_mask, ikps_width, isegs, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver  = tf.train.Saver(tf.global_variables(), max_to_keep=2)
    
    print(len(data_source))

    gpu_options=tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        with tf.name_scope('summary'):
            tf.summary.scalar("learning_rate", learning_rate)
            tf.summary.scalar("det_loss", det_loss)
            tf.summary.scalar("seg_loss", seg_loss)
            tf.summary.scalar("total_loss", total_loss)

            logdir = "./log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            write_op = tf.summary.merge_all()
            summary_writer  = tf.summary.FileWriter(logdir, graph=sess.graph)
        
        # train
        sess.run(tf.global_variables_initializer())
        if checkpoint == None:
            #load_weights(sess,'./pretrained_weights/resnet50.npy')
            variable_restore_op = slim.assign_from_checkpoint_fn('./pretrained_weights/resnet_v1_50.ckpt', 
                                                                     slim.get_trainable_variables(),
                                                                     ignore_missing_vars=True)
            variable_restore_op(sess)
            print('load pretrained weights resnet50!')
        else:
            saver.restore(sess, checkpoint)
            print("load model: ", checkpoint)

        print('Global Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.global_variables()]))
        print('Trainable Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        print('\n----------- start to train -----------\n')

        for epoch in range(1, 1+configs.epochs):
            epoch_loss = []
            epoch_seg_loss = []
            epoch_det_loss = []
            start = time.time()
            for data in datasets:
                imgs, gt_texts, hms, whs, reg_masks, inds, hm_hps, kpss, kps_masks, kps_width = data
                feed_dict = {in_imgs:imgs, 
                             batch_hm:hms,
                             batch_wh:whs,
                             batch_reg_mask:reg_masks, 
                             batch_ind:inds,
                             batch_hm_hp:hm_hps,
                             batch_kps:kpss,
                             batch_kps_mask:kps_masks,
                             batch_kps_width:kps_width,
                             batch_gt_text:gt_texts}
                _, summary, step_loss, det_los, seg_los, hm_los, wh_los, hm_hp_los, kpt_los, kpt_wid_los, step, lr = \
                    sess.run([train_op, write_op, total_loss, det_loss, seg_loss, hm_loss, wh_loss, hm_hp_loss, 
                              kpt_loss, kpt_wid_loss, global_step, learning_rate], feed_dict=feed_dict)

                epoch_loss.append(step_loss)
                epoch_seg_loss.append(seg_los)
                epoch_det_loss.append(det_los)
                if step % 10 == 0:
                    summary_writer.add_summary(summary, step)
                    print(('Epoch:{}, Step:{}, loss:{:.3f}, det_loss:{:.3f}, seg_loss:{:.3f}, hm_loss:{:.3f}, wh_loss:{:.3f}, hm_hp_loss:{:.3f}, kpt_loss:{:.3f}, wid_loss:{:.3f}, lr:{:.6f}'
                            ).format(epoch, step, step_loss, det_los, seg_los, hm_los, wh_los, hm_hp_los, kpt_los, kpt_wid_los, lr))

            epoch_loss = np.mean(epoch_loss)
            epoch_seg_loss = np.mean(epoch_seg_loss)
            epoch_det_loss = np.mean(epoch_det_loss)
            print('Epoch:{}, average loss:{:.3f}, det loss:{:.3f}, seg loss:{:.3f}, time:{:.2f}'
                .format(epoch, epoch_loss, epoch_det_loss, epoch_seg_loss, time.time()-start))
            
            if epoch % 10 == 0:
                saver.save(sess, "./checkpoint/td500/weight50_td500", global_step=global_step)


if __name__ == '__main__': 

    checkpoint = 'checkpoint/icdar/weight50_icdar-50000'
    train(checkpoint=checkpoint)