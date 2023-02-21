import os
import numpy as np
import tensorflow as tf
import cv2
import math
import time
import shutil
import cfg_text
from text_net import TextNet
from utils import data_reader_totaltext, dataset_text
from net.resnet import load_weights

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train(checkpoint=None):
    # define dataset
    configs = cfg_text.Config()
    heads={'hm':1, 'wh':4, 'offset':2, 'hm_hp':7, 'hp_kp':14, 'hp_offset':2, 'kpwidth':7}

    img_dir = 'data/total_text/train'
    data_source = data_reader_totaltext.DataReader(img_dir, config=configs)
    
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

    # define model and loss
    model = TextNet(in_imgs, heads, is_training=True)
    with tf.variable_scope('loss'):
        hm_loss, wh_loss, hm_hp_loss, kpt_loss, kpt_wid_loss, seg_loss = \
            model.compute_loss(batch_hm, batch_wh, batch_reg_mask, batch_ind, batch_hm_hp, 
                batch_kps, batch_kps_mask, batch_kps_width, batch_gt_text)
        det_loss = hm_loss + wh_loss + hm_hp_loss + kpt_loss + kpt_wid_loss
        total_loss = det_loss + seg_loss
        #total_loss = tf.add_n([total_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    global_step = tf.train.create_global_step()
    training_variables = tf.trainable_variables()
    #learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=250, decay_rate=0.9, staircase=True)
    learning_rate = tf.train.piecewise_constant(global_step, [10000, 20000], [0.001, 0.0001, 0.00001])
    optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grads_and_vars = optimizer.compute_gradients(total_loss, var_list=training_variables)
        clip_grad_var = [(g, v) if g is None else (tf.clip_by_norm(g, 10.), v) for g, v in grads_and_vars]
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step, name='train_op')
        #train_op = optimizer.minimize(total_loss, global_step=global_step)

    saver  = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    
    print(len(data_source))

    gpu_options=tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
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
            load_weights(sess,'./pretrained_weights/resnet50.npy')
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
                _, summary, step_loss, det_los, seg_los, step, lr = \
                    sess.run([train_op, write_op, total_loss, det_loss, seg_loss, global_step, learning_rate], feed_dict=feed_dict)

                epoch_loss.append(step_loss)
                epoch_seg_loss.append(seg_los)
                epoch_det_loss.append(det_los)
                if step % 10 == 0:
                    summary_writer.add_summary(summary, step)
                    print(('Epoch:{}, Step:{}, loss:{:.3f}, det_loss:{:.3f}, seg_loss:{:.3f}, lr:{:.6f}'
                            ).format(epoch, step, step_loss, det_los, seg_los, lr))

            epoch_loss = np.mean(epoch_loss)
            epoch_seg_loss = np.mean(epoch_seg_loss)
            epoch_det_loss = np.mean(epoch_det_loss)
            print('Epoch:{}, average loss:{:.3f}, det loss:{:.3f}, seg loss:{:.3f}, time:{:.2f}'
                .format(epoch, epoch_loss, epoch_det_loss, epoch_seg_loss, time.time()-start))
            
            if epoch % 10 == 0:
                saver.save(sess, "./checkpoint/weight50_totaltext", global_step=global_step)


if __name__ == '__main__': 

    #checkpoint = 'checkpoint/weight50_text-10000'
    train()