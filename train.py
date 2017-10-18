from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import numpy as np
import tensorflow as tf

import i3d 
from rmb_lib.action_dataset import *


_BATCH_SIZE = 8
_CLIP_SIZE = 16
_FRAME_SIZE = 224 
_LEARNING_RATE = 0.00001
_GLOBAL_EPOCH = 10


_CHECKPOINT_PATHS = {
    'rgb': './data/checkpoints/rgb_scratch/model.ckpt',
    'flow': './data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': './data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': './data/checkpoints/flow_imagenet/model.ckpt',
}

_CHANNEL = {
    'rgb': 3,
    'flow': 2,
}

_SCOPE = {
    'rgb': 'RGB',
    'flow': 'Flow',
}


def main(dataset_name, data_tag):
    assert data_tag in ['rgb', 'flow']
    train_info, test_info = split_data(
        os.path.join('./data', dataset_name, data_tag+'.txt'),
        os.path.join('./data', dataset_name, 'testlist01.txt'))
    train_data = Action_Dataset(dataset_name, data_tag, train_info)
    test_data = Action_Dataset(dataset_name, data_tag, test_info)
    with open(os.path.join('./data', dataset_name, 'label_map.txt')) as f:
        label_map = [x.strip() for x in f.readlines()]

    clip_holder = tf.placeholder(
        tf.float32, [None, None, _FRAME_SIZE, _FRAME_SIZE, _CHANNEL[train_data.tag]])
    label_holder = tf.placeholder(tf.int32, [None])
    dropout_holder = tf.placeholder(tf.float32)
    is_train_holder = tf.placeholder(tf.bool)
  
    with tf.variable_scope(_SCOPE[train_data.tag]):
        model = i3d.InceptionI3d(400, spatial_squeeze=True, final_endpoint='Logits')
        logits, _ = model(clip_holder, is_training=is_train_holder, dropout_keep_prob=1)
        logits_dropout = tf.nn.dropout(logits, dropout_holder)   
        fc_out = tf.layers.dense(logits_dropout, 101, tf.nn.relu, use_bias=True)

    variable_map = {}
    train_var = []
    for variable in tf.global_variables():
        tmp = variable.name.split('/')
        if tmp[1] == 'dense':
            train_var.append(variable)
        if tmp[0] == _SCOPE[train_data.tag] and tmp[1] != 'dense':
            variable_map[variable.name.replace(':0', '')] = variable
        if tmp[-1] == 'w:0':
            weight_l2 = tf.nn.l2_loss(variable)
            tf.add_to_collection('weight_l2', weight_l2)
    print(train_var)
    saver = tf.train.Saver(var_list=variable_map, reshape=True)
    top_k_op = tf.nn.in_top_k(fc_out, label_holder, 1)

    loss_weight = tf.add_n(tf.get_collection('weight_l2'), 'loss_weight')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_holder, logits=fc_out))
    total_loss = loss #+ 1e-7 * loss_weight
    tf.summary.scalar('total_loss', total_loss)

    per_epoch_step = int(np.ceil(train_data.size/_BATCH_SIZE))
    global_step = _GLOBAL_EPOCH * per_epoch_step
    decay_step = 3*per_epoch_step
    learning_rate = tf.train.exponential_decay(
        _LEARNING_RATE, global_step, decay_step, 0.1, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss, var_list=train_var)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, _CHECKPOINT_PATHS[train_data.tag+'_imagenet'])
    print('----Here we start!----')

    step = 0
    true_count = 0
    per_test_step = int(np.ceil(test_data.size/_BATCH_SIZE))
    while step <= global_step:
        step += 1
        start_time = time.time()
        clip, label = train_data.next_batch(_BATCH_SIZE, _CLIP_SIZE)
        clip = clip/255
        _, loss_now, predictions = sess.run([optimizer, total_loss, top_k_op],
                               feed_dict={clip_holder: clip,
                                          label_holder: label,
                                          dropout_holder: 0.7,
                                          is_train_holder: False})
        duration = time.time() - start_time
        true_count += np.sum(predictions)
        if step % 10 == 0:
            print('step: %-4d, loss: %-.4f (%.2f sec/batch)' % (step, loss_now, float(duration)))
            print(label)
            print(predictions)
        if step % per_epoch_step ==0:
            accuracy = true_count/ (per_test_step*_BATCH_SIZE)
            print('Epoch%d, train accuracy: %.3f' % (train_data.epoch_completed, accuracy))
        if step % decay_step == 0:
            true_count = 0
            for i in range(per_test_step):
                clip, label = test_data.next_batch(
                    _BATCH_SIZE, _CLIP_SIZE, shuffle=False, data_augment=False)
                clip = clip/255
                predictions = sess.run(top_k_op, feed_dict={clip_holder: clip,
                                                              label_holder: label,
                                                              dropout_holder: 1,
                                                              is_train_holder: False})
                true_count += np.sum(predictions)
            accuracy = true_count/ (per_test_step*_BATCH_SIZE)
            test_data.index_in_epoch = 0
            print('Epoch%d, test accuracy: %.3f' % (train_data.epoch_completed, accuracy))

if __name__ == '__main__':
    description = 'finetune on other dataset'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('dataset_name', type=str)
    p.add_argument('data_tag', type=str)
    main(**vars(p.parse_args()))
